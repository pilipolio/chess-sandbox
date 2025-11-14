"""
Training pipeline for concept probes.

Trains multi-label logistic regression classifiers to detect chess concepts
from LC0 layer activations.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path

import click
import numpy as np
from huggingface_hub import HfApi, hf_hub_download
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

from ...git import get_commit_sha
from ...logging_config import setup_logging
from .dataset import LabelledPosition, load_dataset_from_hf, rebalance_positions, split_positions
from .evaluation import calculate_multilabel_metrics, format_metrics_display
from .features import LczeroModel, extract_features_batch
from .inference import ConceptProbe
from .model_artefact import ModelTrainingOutput, generate_probe_name

logger = setup_logging(__name__)


def normalize_predict_proba(y_proba: np.ndarray | list[np.ndarray]) -> np.ndarray:
    """
    Normalize predict_proba output to a 2D array.

    DummyClassifier.predict_proba() returns a list of arrays for multilabel,
    while OneVsRestClassifier returns a 2D array. This function normalizes both.

    Args:
        y_proba: Either a 2D array (n_samples, n_classes) or a list of 2D arrays

    Returns:
        2D array of shape (n_samples, n_classes) with probabilities for class 1
    """
    if isinstance(y_proba, list):
        # DummyClassifier multilabel: list of (n_samples, 2) arrays
        # Extract probability of positive class (index 1) from each
        return np.column_stack([proba[:, 1] for proba in y_proba])
    else:
        # Already a proper 2D array (OneVsRestClassifier or multiclass)
        return y_proba


def save_dataset_split(positions: list[LabelledPosition], output_path: Path) -> Path:
    """
    Save dataset split to JSONL file.

    Args:
        positions: List of labeled positions
        output_path: Path to save JSONL file

    Returns:
        Path to saved file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for position in positions:
            json.dump(position.to_dict(), f)
            f.write("\n")
    return output_path


def upload_dataset_split(
    local_path: Path,
    repo_id: str,
    filename: str,
    revision: str | None = None,
) -> str:
    """
    Upload dataset split to HuggingFace Hub.

    Args:
        local_path: Local path to JSONL file
        repo_id: HuggingFace dataset repository ID
        filename: Filename in the repository
        revision: Optional git revision (branch/tag)

    Returns:
        Commit info string
    """
    api = HfApi()
    commit_message = f"Add {filename} split from training pipeline"
    if revision:
        commit_message += f" (revision: {revision})"

    result = api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=filename,
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        commit_message=commit_message,
    )
    return str(result)


def train_multiclass(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train_labels: list[list[str]],
    y_test_labels: list[list[str]],
    layer_name: str,
    random_seed: int,
    classifier_c: float = 1.0,
    classifier_class_weight: str = "none",
    verbose: bool = False,
    n_jobs: int = -1,
) -> ModelTrainingOutput:
    """
    Train multi-class concept probe (one concept per position).

    Takes multi-label format labels and flattens them internally (takes first concept).

    Args:
        X_train: Pre-extracted activation features for training
        X_test: Pre-extracted activation features for testing
        y_train_labels: List of concept label lists for training
        y_test_labels: List of concept label lists for testing
        layer_name: Layer name that activations were extracted from
        random_seed: Random seed for reproducibility
        verbose: Enable sklearn training progress output
        n_jobs: Number of parallel jobs (-1 = all cores)

    Returns:
        Trained ModelTrainingOutput
    """
    logger.info("\n[1/4] Preparing labels...")
    # Filter out positions without concepts (multi-class requires at least one concept)
    train_mask = [len(concepts) > 0 for concepts in y_train_labels]
    test_mask = [len(concepts) > 0 for concepts in y_test_labels]

    X_train_filtered = X_train[train_mask]
    X_test_filtered = X_test[test_mask]
    y_train_filtered = [y_train_labels[i] for i in range(len(y_train_labels)) if train_mask[i]]
    y_test_filtered = [y_test_labels[i] for i in range(len(y_test_labels)) if test_mask[i]]

    if not y_train_filtered:
        raise ValueError(
            "No positions with concepts found in training set! "
            "Multi-class mode requires at least one concept per position."
        )

    labels_flat = [concepts[0] for concepts in y_train_filtered]
    logger.info(f"Original: Train={X_train.shape[0]}, Test={X_test.shape[0]}")
    logger.info(f"Filtered (with concepts): Train={X_train_filtered.shape[0]}, Test={X_test_filtered.shape[0]}")
    logger.info("Flattened multi-label to multi-class (taking first concept)")

    logger.info("\n[2/4] Encoding labels...")
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(labels_flat)
    concept_list = list(encoder.classes_)
    logger.info(f"Label encoder classes: {encoder.classes_}")

    logger.info("\n[3/4] Training probe...")
    clf = LogisticRegression(
        C=classifier_c,
        class_weight="balanced" if classifier_class_weight == "balanced" else None,
        max_iter=10000,
        random_state=random_seed,
        verbose=verbose,
        n_jobs=n_jobs,
    )
    logger.info(f"Training multi-class classifier with {len(encoder.classes_)} classes")
    clf.fit(X_train_filtered, y_train_encoded)
    logger.info("Training complete!")

    baseline = DummyClassifier(strategy="stratified", random_state=random_seed)
    baseline.fit(X_train_filtered, y_train_encoded)

    logger.info("\n[4/4] Evaluating...")
    eval_encoder = MultiLabelBinarizer()
    eval_encoder.fit([concept_list])
    y_test_binary = eval_encoder.transform(y_test_filtered)

    y_score_baseline = baseline.predict_proba(X_test_filtered)
    y_score_probe = clf.predict_proba(X_test_filtered)

    baseline_metrics = calculate_multilabel_metrics(y_test_binary, y_score_baseline, concept_list)
    probe_metrics = calculate_multilabel_metrics(y_test_binary, y_score_probe, concept_list)

    logger.info(format_metrics_display(probe_metrics, baseline_metrics=baseline_metrics))

    probe = ConceptProbe(
        classifier=clf,
        concept_list=concept_list,
        layer_name=layer_name,
        label_encoder=encoder,
    )

    training_stats = {
        "baseline": baseline_metrics.model_dump(),
        "probe": probe_metrics.model_dump(),
        "training_samples": X_train_filtered.shape[0],
        "test_samples": X_test_filtered.shape[0],
        "random_seed": random_seed,
        "mode": "multi-class",
        "classifier_c": classifier_c,
        "classifier_class_weight": classifier_class_weight,
        "verbose": verbose,
        "n_jobs": n_jobs,
    }

    return ModelTrainingOutput(
        probe=probe,
        training_stats=training_stats,
        source_provenance=None,
        training_date=datetime.now().isoformat(),
    )


def train_multilabel(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train_labels: list[list[str]],
    y_test_labels: list[list[str]],
    layer_name: str,
    random_seed: int,
    classifier_c: float = 1.0,
    classifier_class_weight: str = "none",
    verbose: bool = False,
    n_jobs: int = -1,
) -> ModelTrainingOutput:
    """
    Train multi-label concept probe (multiple concepts per position).

    Args:
        X_train: Pre-extracted activation features for training
        X_test: Pre-extracted activation features for testing
        y_train_labels: List of concept label lists for training
        y_test_labels: List of concept label lists for testing
        layer_name: Layer name that activations were extracted from
        random_seed: Random seed for reproducibility
        verbose: Enable sklearn training progress output
        n_jobs: Number of parallel jobs (-1 = all cores)

    Returns:
        Trained ModelTrainingOutput
    """
    if not y_train_labels:
        raise ValueError("No validated concepts found! Cannot train without labels.")

    logger.info("\n[1/4] Encoding labels...")
    encoder = MultiLabelBinarizer()
    y_train = encoder.fit_transform(y_train_labels)
    y_test_binary = encoder.transform(y_test_labels)
    concept_list = list(encoder.classes_)
    logger.info(f"Multi-label binarizer classes: {encoder.classes_}")
    logger.info(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")
    logger.info(f"Binary label matrix shape: {y_train.shape}")
    logger.info(f"Total labels: {y_train.sum()} ({y_train.sum() / y_train.size * 100:.1f}% density)")

    logger.info("\n[2/4] Training probe...")
    base_clf = LogisticRegression(
        C=classifier_c,
        class_weight="balanced" if classifier_class_weight == "balanced" else None,
        max_iter=1000,
        random_state=random_seed,
        verbose=verbose,
    )
    clf = OneVsRestClassifier(base_clf, n_jobs=n_jobs, verbose=verbose)
    logger.info(f"Training multi-label classifier with {len(encoder.classes_)} concepts")
    clf.fit(X_train, y_train)
    logger.info("Training complete!")

    baseline = DummyClassifier(strategy="stratified", random_state=random_seed)
    baseline.fit(X_train, y_train)

    logger.info("\n[3/4] Evaluating...")

    logger.info("=" * 70)
    logger.info("TRAINING SET")
    logger.info("=" * 70)

    y_train_proba_baseline = normalize_predict_proba(baseline.predict_proba(X_train))
    y_train_proba_probe = clf.predict_proba(X_train)

    logger.info(
        format_metrics_display(
            calculate_multilabel_metrics(y_train, y_train_proba_probe, concept_list),
            baseline_metrics=calculate_multilabel_metrics(y_train, y_train_proba_baseline, concept_list),
        )
    )

    logger.info("=" * 70)
    logger.info("TEST SET")
    logger.info("=" * 70)

    y_pred_baseline = normalize_predict_proba(baseline.predict_proba(X_test))
    y_pred_probe = clf.predict_proba(X_test)

    baseline_metrics = calculate_multilabel_metrics(y_test_binary, y_pred_baseline, concept_list)
    probe_metrics = calculate_multilabel_metrics(y_test_binary, y_pred_probe, concept_list)

    logger.info(format_metrics_display(probe_metrics, baseline_metrics=baseline_metrics))

    probe = ConceptProbe(
        classifier=clf,
        concept_list=concept_list,
        layer_name=layer_name,
        label_encoder=encoder,
    )

    training_stats = {
        "baseline": baseline_metrics.model_dump(),
        "probe": probe_metrics.model_dump(),
        "training_samples": X_train.shape[0],
        "test_samples": X_test.shape[0],
        "classifier_mode": "multi-label",
        "classifier_c": classifier_c,
        "classifier_class_weight": classifier_class_weight,
        "random_seed": random_seed,
    }

    return ModelTrainingOutput(
        probe=probe,
        training_stats=training_stats,
        source_provenance=None,
        training_date=datetime.now().isoformat(),
    )


@click.command()
@click.option(
    "--dataset-repo-id",
    required=True,
    help="HuggingFace dataset repository ID (e.g., 'pilipolio/chess-concepts-async-100')",
)
@click.option(
    "--dataset-filename",
    default="data.jsonl",
    help="JSONL filename in the dataset repository",
)
@click.option(
    "--dataset-revision",
    default=None,
    help="Git revision for dataset (tag, branch, commit). Defaults to main",
)
@click.option(
    "--lc0-model-repo-id",
    required=True,
    help="HuggingFace model repository ID (e.g., 'lczerolens/maia-1500')",
)
@click.option(
    "--lc0-model-filename",
    default="model.onnx",
    help="Model filename in the repository (default: model.onnx)",
)
@click.option(
    "--layer-name",
    default="block3/conv2/relu",
    help="Layer name to extract activations from",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to save trained probe (auto-generated if not provided)",
)
@click.option(
    "--test-split",
    default=0.2,
    type=float,
    help="Fraction of data to use for testing",
)
@click.option(
    "--random-seed",
    default=42,
    type=int,
    help="Random seed for reproducibility",
)
@click.option(
    "--classifier-mode",
    default="multi-label",
    type=click.Choice(["multi-class", "multi-label"]),
    help="Training mode: 'multi-class' (one concept per position) or 'multi-label' (multiple concepts)",
)
@click.option(
    "--classifier-c",
    default=1.0,
    type=float,
    help="Inverse of regularization strength for LogisticRegression (lower values = stronger regularization)",
)
@click.option(
    "--classifier-class-weight",
    default="none",
    type=click.Choice(["none", "balanced"]),
    help=(
        "Class weight strategy: 'none' (default) or 'balanced' "
        "(adjust weights inversely proportional to class frequencies)"
    ),
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable verbose output from sklearn models (shows training progress)",
)
@click.option(
    "--n-jobs",
    default=-1,
    type=int,
    help="Number of parallel jobs for sklearn (-1 uses all cores, -2 leaves one free)",
)
@click.option(
    "--upload-to-hub",
    is_flag=True,
    help="Upload model output to HuggingFace Hub after training",
)
@click.option(
    "--output-repo-id",
    default=None,
    help="HuggingFace repository ID for upload (e.g., 'pilipolio/chess-positions-extractor')",
)
@click.option(
    "--output-revision",
    default=None,
    help="Optional revision/tag name for the upload (included in commit message)",
)
@click.option(
    "--save-splits",
    is_flag=True,
    help="Save train/test splits as train.jsonl and test.jsonl to the dataset repository",
)
@click.option(
    "--rebalance-training-positions",
    is_flag=False,
    help="Rebalance training positions to achieve 50/50 ratio of positions with/without validated concepts",
)
def train(
    dataset_repo_id: str,
    dataset_filename: str,
    dataset_revision: str | None,
    lc0_model_repo_id: str,
    lc0_model_filename: str,
    layer_name: str,
    output: Path | None,
    test_split: float,
    random_seed: int,
    classifier_mode: str,
    classifier_c: float,
    classifier_class_weight: str,
    verbose: bool,
    n_jobs: int,
    upload_to_hub: bool,
    output_repo_id: str | None,
    output_revision: str | None,
    save_splits: bool,
    rebalance_training_positions: bool,
) -> None:
    """
    Train concept probe from labeled positions.

    Example:
        python -m chess_sandbox.concept_extraction.model.train \\
            --dataset-repo-id pilipolio/chess-positions-concepts \\
            --dataset-filename data.jsonl \\
            --lc0-model-repo-id lczerolens/maia-1500 \\
            --layer-name block3/conv2/relu \\
            --classifier-mode multi-label \\
            --classifier-c 1.0 \\
            --classifier-class-weight none
    """
    logger.info("Training concept probe...")
    dataset_ref = f"{dataset_repo_id}/{dataset_filename}"
    if dataset_revision:
        dataset_ref += f"@{dataset_revision}"
    logger.info(f"  Dataset: {dataset_ref}")

    model_ref = f"{lc0_model_repo_id}/{lc0_model_filename}"
    logger.info(f"  Model: {model_ref}")

    logger.info(f"  Layer: {layer_name}")
    logger.info(f"  Classifier mode: {classifier_mode}")
    logger.info(f"  Classifier C: {classifier_c}")
    logger.info(f"  Classifier class_weight: {classifier_class_weight}")
    logger.info(f"  Parallel jobs (n_jobs): {n_jobs}")

    logger.info("\nLoading LC0 model from HuggingFace Hub...")
    lc0_model_path = hf_hub_download(repo_id=lc0_model_repo_id, filename=lc0_model_filename)
    model = LczeroModel.from_path(lc0_model_path)
    logger.info("Model loaded successfully!")

    logger.info("\nLoading training data from HuggingFace Hub...")
    positions = load_dataset_from_hf(dataset_repo_id, dataset_filename, dataset_revision)

    logger.info("\nSplitting dataset...")
    indices = np.arange(len(positions))
    train_and_test_indices: list[list[int]] = train_test_split(indices, test_size=test_split, random_state=random_seed)

    train_positions = [positions[i] for i in train_and_test_indices[0]]
    test_positions = [positions[i] for i in train_and_test_indices[1]]

    if rebalance_training_positions:
        train_positions = rebalance_positions(train_positions, random_state=random_seed)
    else:
        # by default not using any positions w/o validated concepts
        train_positions, _ = split_positions(train_positions)
        # test_positions, _ = split_positions(test_positions)

    logger.info(f"Train: {len(train_positions)} samples, Test: {len(test_positions)} samples")

    if save_splits:
        logger.info("\nSaving dataset splits...")
        train_path = save_dataset_split(train_positions, Path("data/splits/train.jsonl"))
        test_path = save_dataset_split(test_positions, Path("data/splits/test.jsonl"))
        logger.info(f"  Train split: {train_path}")
        logger.info(f"  Test split: {test_path}")

        logger.info("\nUploading splits to HuggingFace Hub...")
        train_commit = upload_dataset_split(
            train_path,
            dataset_repo_id,
            "train.jsonl",
            dataset_revision,
        )
        test_commit = upload_dataset_split(
            test_path,
            dataset_repo_id,
            "test.jsonl",
            dataset_revision,
        )
        logger.info(f"  Train commit: {train_commit}")
        logger.info(f"  Test commit: {test_commit}")

    # Extract validated concept names from positions
    y_train_labels = [[c.name for c in p.concepts if c.validated_by] for p in train_positions]
    y_test_labels = [[c.name for c in p.concepts if c.validated_by] for p in test_positions]

    logger.info("\nExtracting train activations...")
    train_fens = [p.fen for p in train_positions]
    X_train = extract_features_batch(
        train_fens,
        layer_name,
        model=model,
    )
    logger.info(f"Train activation matrix shape: {X_train.shape}")

    logger.info("\nExtracting test activations...")
    test_fens = [p.fen for p in test_positions]
    X_test = extract_features_batch(
        test_fens,
        layer_name,
        model=model,
    )
    logger.info(f"Test activation matrix shape: {X_test.shape}")

    if classifier_mode == "multi-class":
        training_output = train_multiclass(
            X_train=X_train,
            X_test=X_test,
            y_train_labels=y_train_labels,
            y_test_labels=y_test_labels,
            layer_name=layer_name,
            random_seed=random_seed,
            classifier_c=classifier_c,
            classifier_class_weight=classifier_class_weight,
            verbose=verbose,
            n_jobs=n_jobs,
        )
    else:
        training_output = train_multilabel(
            X_train=X_train,
            X_test=X_test,
            y_train_labels=y_train_labels,
            y_test_labels=y_test_labels,
            layer_name=layer_name,
            random_seed=random_seed,
            classifier_c=classifier_c,
            classifier_class_weight=classifier_class_weight,
            verbose=verbose,
            n_jobs=n_jobs,
        )

    training_output.source_provenance = {
        "source_model": {
            "repo_id": lc0_model_repo_id,
            "filename": lc0_model_filename,
        },
        "source_dataset": {
            "repo_id": dataset_repo_id,
            "filename": dataset_filename,
            "revision": dataset_revision,
            "hash": hashlib.sha256(dataset_repo_id.encode()).hexdigest(),
        },
        "training_code": {
            "repo": "pilipolio/chess-sandbox",
            "commit": get_commit_sha(),
        },
    }

    output_path = output or Path("data/models/concept_probes") / generate_probe_name(
        lc0_model_repo_id, layer_name, classifier_mode, dataset_repo_id
    )
    logger.info(f"  Output: {output_path}")

    training_output.save(output_path)

    if upload_to_hub:
        if not output_repo_id:
            raise ValueError("--output-repo-id is required when --upload-to-hub is set")
        commit = training_output.upload_to_hf(
            local_dir=output_path,
            repo_id=output_repo_id,
            revision=output_revision,
        )
        logger.info(f"Successfully uploaded: {commit}")


if __name__ == "__main__":
    train()
