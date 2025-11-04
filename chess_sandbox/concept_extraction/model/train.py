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

from .dataset import LabelledPosition, load_dataset_from_hf
from .evaluation import calculate_multilabel_metrics, format_metrics_display
from .features import LczeroModel, extract_features_batch
from .inference import ConceptProbe
from .model_artefact import ModelTrainingOutput, generate_probe_name


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
    split_name: str = "train",
) -> str:
    """
    Upload dataset split to HuggingFace Hub.

    Args:
        local_path: Local path to JSONL file
        repo_id: HuggingFace dataset repository ID
        filename: Filename in the repository
        revision: Optional git revision (branch/tag)
        split_name: Name of split for commit message (train/test)

    Returns:
        Commit info string
    """
    api = HfApi()
    commit_message = f"Add {split_name} split from training pipeline"
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
    print("\n[1/4] Preparing labels...")
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
    print(f"Original: Train={X_train.shape[0]}, Test={X_test.shape[0]}")
    print(f"Filtered (with concepts): Train={X_train_filtered.shape[0]}, Test={X_test_filtered.shape[0]}")
    print("Flattened multi-label to multi-class (taking first concept)")

    print("\n[2/4] Encoding labels...")
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(labels_flat)
    concept_list = list(encoder.classes_)
    print(f"Label encoder classes: {encoder.classes_}")

    print("\n[3/4] Training probe...")
    # NOTE: Consider adding class_weight='balanced' to handle remaining class imbalance
    # within concepts (e.g., mating_threat:11k vs zugzwang:194 samples). This would
    # automatically adjust weights inversely proportional to class frequencies.
    clf = LogisticRegression(
        max_iter=10000,
        random_state=random_seed,
        verbose=verbose,
        n_jobs=n_jobs,
    )
    print(f"Training multi-class classifier with {len(encoder.classes_)} classes")
    clf.fit(X_train_filtered, y_train_encoded)
    print("Training complete!")

    baseline = DummyClassifier(strategy="stratified", random_state=random_seed)
    baseline.fit(X_train_filtered, y_train_encoded)

    print("\n[4/4] Evaluating...")
    eval_encoder = MultiLabelBinarizer()
    eval_encoder.fit([concept_list])
    y_test_binary = eval_encoder.transform(y_test_filtered)

    y_pred_baseline = baseline.predict(X_test_filtered)
    y_pred_probe = clf.predict(X_test_filtered)

    y_pred_baseline_labels = [[concept_list[int(pred)]] for pred in y_pred_baseline]
    y_pred_probe_labels = [[concept_list[int(pred)]] for pred in y_pred_probe]
    y_pred_baseline_binary = eval_encoder.transform(y_pred_baseline_labels)
    y_pred_probe_binary = eval_encoder.transform(y_pred_probe_labels)

    baseline_metrics = calculate_multilabel_metrics(y_test_binary, y_pred_baseline_binary, concept_list)
    probe_metrics = calculate_multilabel_metrics(y_test_binary, y_pred_probe_binary, concept_list)

    print(format_metrics_display(probe_metrics, "Trained Probe", baseline_metrics=baseline_metrics))

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

    print("\n[1/4] Encoding labels...")
    encoder = MultiLabelBinarizer()
    y_train = encoder.fit_transform(y_train_labels)
    y_test_binary = encoder.transform(y_test_labels)
    concept_list = list(encoder.classes_)
    print(f"Multi-label binarizer classes: {encoder.classes_}")
    print(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")
    print(f"Binary label matrix shape: {y_train.shape}")
    print(f"Total labels: {y_train.sum()} ({y_train.sum() / y_train.size * 100:.1f}% density)")

    print("\n[2/4] Training probe...")
    # NOTE: Consider adding class_weight='balanced' to handle remaining class imbalance
    # within concepts (e.g., mating_threat:11k vs zugzwang:194 samples). This would
    # automatically adjust weights inversely proportional to class frequencies in each
    # binary classifier (OneVsRest trains one binary classifier per concept).
    base_clf = LogisticRegression(
        max_iter=10000,
        random_state=random_seed,
        verbose=verbose,
    )
    clf = OneVsRestClassifier(base_clf, n_jobs=n_jobs, verbose=verbose)
    print(f"Training multi-label classifier with {len(encoder.classes_)} concepts")
    clf.fit(X_train, y_train)
    print("Training complete!")

    baseline = DummyClassifier(strategy="stratified", random_state=random_seed)
    baseline.fit(X_train, y_train)

    print("\n[3/4] Evaluating...")
    y_pred_baseline = baseline.predict(X_test)
    y_pred_probe = clf.predict(X_test)

    baseline_metrics = calculate_multilabel_metrics(y_test_binary, y_pred_baseline, concept_list)
    probe_metrics = calculate_multilabel_metrics(y_test_binary, y_pred_probe, concept_list)

    print(format_metrics_display(probe_metrics, "Trained Probe", baseline_metrics=baseline_metrics))

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
        "random_seed": random_seed,
        "mode": "multi-label",
        "verbose": verbose,
        "n_jobs": n_jobs,
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
    "--batch-size",
    default=32,
    type=int,
    help="Batch size for activation extraction",
)
@click.option(
    "--mode",
    default="multi-label",
    type=click.Choice(["multi-class", "multi-label"]),
    help="Training mode: 'multi-class' (one concept per position) or 'multi-label' (multiple concepts)",
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
    batch_size: int,
    mode: str,
    verbose: bool,
    n_jobs: int,
    upload_to_hub: bool,
    output_repo_id: str | None,
    output_revision: str | None,
    save_splits: bool,
) -> None:
    """
    Train concept probe from labeled positions.

    Example:
        python -m chess_sandbox.concept_extraction.model.train \\
            --dataset-repo-id pilipolio/chess-concepts-async-100 \\
            --dataset-filename data.jsonl \\
            --lc0-model-repo-id lczerolens/maia-1500 \\
            --layer-name block3/conv2/relu \\
            --mode multi-label
    """
    print("Training concept probe...")
    dataset_ref = f"{dataset_repo_id}/{dataset_filename}"
    if dataset_revision:
        dataset_ref += f"@{dataset_revision}"
    print(f"  Dataset: {dataset_ref}")

    model_ref = f"{lc0_model_repo_id}/{lc0_model_filename}"
    print(f"  Model: {model_ref}")

    print(f"  Layer: {layer_name}")
    print(f"  Mode: {mode}")
    print(f"  Parallel jobs (n_jobs): {n_jobs}")

    print("\nLoading LC0 model from HuggingFace Hub...")
    lc0_model_path = hf_hub_download(repo_id=lc0_model_repo_id, filename=lc0_model_filename)
    model = LczeroModel.from_path(lc0_model_path)
    print("Model loaded successfully!")

    print("\nLoading training data from HuggingFace Hub...")
    positions, labels = load_dataset_from_hf(dataset_repo_id, dataset_filename, dataset_revision)

    print("\nSplitting dataset...")
    indices = np.arange(len(positions))
    train_indices, test_indices = train_test_split(indices, test_size=test_split, random_state=random_seed)

    train_positions = [positions[i] for i in train_indices]
    test_positions = [positions[i] for i in test_indices]
    y_train_labels = [labels[i] for i in train_indices]
    y_test_labels = [labels[i] for i in test_indices]
    print(f"Train: {len(train_positions)} samples, Test: {len(test_positions)} samples")

    if save_splits:
        print("\nSaving dataset splits...")
        train_path = save_dataset_split(train_positions, Path("data/splits/train.jsonl"))
        test_path = save_dataset_split(test_positions, Path("data/splits/test.jsonl"))
        print(f"  Train split: {train_path}")
        print(f"  Test split: {test_path}")

        print("\nUploading splits to HuggingFace Hub...")
        train_commit = upload_dataset_split(
            train_path,
            dataset_repo_id,
            "train.jsonl",
            dataset_revision,
            split_name="train",
        )
        test_commit = upload_dataset_split(
            test_path,
            dataset_repo_id,
            "test.jsonl",
            dataset_revision,
            split_name="test",
        )
        print(f"  Train commit: {train_commit}")
        print(f"  Test commit: {test_commit}")

    print("\nExtracting train activations...")
    train_fens = [p.fen for p in train_positions]
    X_train = extract_features_batch(
        train_fens,
        layer_name,
        model=model,
        batch_size=batch_size,
    )
    print(f"Train activation matrix shape: {X_train.shape}")

    print("\nExtracting test activations...")
    test_fens = [p.fen for p in test_positions]
    X_test = extract_features_batch(
        test_fens,
        layer_name,
        model=model,
        batch_size=batch_size,
    )
    print(f"Test activation matrix shape: {X_test.shape}")

    if mode == "multi-class":
        training_output = train_multiclass(
            X_train=X_train,
            X_test=X_test,
            y_train_labels=y_train_labels,
            y_test_labels=y_test_labels,
            layer_name=layer_name,
            random_seed=random_seed,
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
    }

    output_path = output or Path("data/models/concept_probes") / generate_probe_name(
        lc0_model_repo_id, layer_name, mode, dataset_repo_id
    )
    print(f"  Output: {output_path}")

    training_output.save(output_path)

    if upload_to_hub:
        if not output_repo_id:
            raise ValueError("--output-repo-id is required when --upload-to-hub is set")
        commit = training_output.upload_to_hf(
            local_dir=output_path,
            repo_id=output_repo_id,
            revision=output_revision,
        )
        print(f"Successfully uploaded: {commit}")


if __name__ == "__main__":
    train()
