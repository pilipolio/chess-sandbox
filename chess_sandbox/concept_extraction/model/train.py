"""
Training pipeline for concept probes.

Trains multi-label logistic regression classifiers to detect chess concepts
from LC0 layer activations.
"""

import hashlib
from datetime import datetime
from pathlib import Path

import click
import numpy as np
from huggingface_hub import hf_hub_download
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

from .dataset import load_dataset_from_hf
from .evaluation import calculate_multilabel_metrics, format_metrics_display
from .features import LczeroModel, extract_features_batch
from .inference import ConceptProbe
from .model_artefact import ModelTrainingOutput, generate_probe_name


def train_multiclass(
    labels: list[list[str]],
    activations: np.ndarray,
    layer_name: str,
    test_split: float,
    random_seed: int,
    verbose: bool = False,
    n_jobs: int = -1,
) -> ModelTrainingOutput:
    """
    Train multi-class concept probe (one concept per position).

    Takes multi-label format labels and flattens them internally (takes first concept).

    Args:
        labels: List of concept label lists (flattened internally to first concept)
        activations: Pre-extracted activation features
        layer_name: Layer name that activations were extracted from
        test_split: Fraction of data for testing
        random_seed: Random seed for reproducibility
        verbose: Enable sklearn training progress output
        n_jobs: Number of parallel jobs (-1 = all cores)

    Returns:
        Trained ModelTrainingOutput
    """
    print("\n[1/4] Splitting data...")
    indices = np.arange(len(labels))
    train_indices, test_indices = train_test_split(indices, test_size=test_split, random_state=random_seed)

    X_train = activations[train_indices]
    X_test = activations[test_indices]
    y_train_labels = [labels[i] for i in train_indices]
    y_test_labels = [labels[i] for i in test_indices]

    labels_flat = [concepts[0] for concepts in y_train_labels]
    print(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")
    print("Flattened multi-label to multi-class (taking first concept)")

    print("\n[2/4] Encoding labels...")
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(labels_flat)
    concept_list = list(encoder.classes_)
    print(f"Label encoder classes: {encoder.classes_}")

    print("\n[3/4] Training probe...")
    clf = LogisticRegression(
        max_iter=10000,
        random_state=random_seed,
        verbose=verbose,
        n_jobs=n_jobs,
    )
    print(f"Training multi-class classifier with {len(encoder.classes_)} classes")
    clf.fit(X_train, y_train_encoded)
    print("Training complete!")

    baseline = DummyClassifier(strategy="stratified", random_state=random_seed)
    baseline.fit(X_train, y_train_encoded)

    print("\n[4/4] Evaluating...")
    eval_encoder = MultiLabelBinarizer()
    eval_encoder.fit([concept_list])
    y_test_binary = eval_encoder.transform(y_test_labels)

    y_pred_baseline = baseline.predict(X_test)
    y_pred_probe = clf.predict(X_test)

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
        "training_samples": X_train.shape[0],
        "test_samples": X_test.shape[0],
        "test_split": test_split,
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
    labels: list[list[str]],
    activations: np.ndarray,
    layer_name: str,
    test_split: float,
    random_seed: int,
    verbose: bool = False,
    n_jobs: int = -1,
) -> ModelTrainingOutput:
    """
    Train multi-label concept probe (multiple concepts per position).

    Args:
        labels: List of concept label lists (multiple per position)
        activations: Pre-extracted activation features
        layer_name: Layer name that activations were extracted from
        test_split: Fraction of data for testing
        random_seed: Random seed for reproducibility
        verbose: Enable sklearn training progress output
        n_jobs: Number of parallel jobs (-1 = all cores)

    Returns:
        Trained ModelTrainingOutput
    """
    if not labels:
        raise ValueError("No validated concepts found! Cannot train without labels.")

    print("\n[1/4] Encoding labels...")
    encoder = MultiLabelBinarizer()
    label_matrix = encoder.fit_transform(labels)
    concept_list = list(encoder.classes_)
    print(f"Multi-label binarizer classes: {encoder.classes_}")
    print(f"Binary label matrix shape: {label_matrix.shape}")
    print(f"Total labels: {label_matrix.sum()} ({label_matrix.sum() / label_matrix.size * 100:.1f}% density)")

    print("\n[2/4] Splitting data...")
    indices = np.arange(len(labels))
    train_indices, test_indices = train_test_split(indices, test_size=test_split, random_state=random_seed)

    X_train = activations[train_indices]
    X_test = activations[test_indices]
    y_train = label_matrix[train_indices]
    y_test_labels = [labels[i] for i in test_indices]
    y_test_binary = encoder.transform(y_test_labels)
    print(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")

    print("\n[3/4] Training probe...")
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

    print("\n[4/4] Evaluating...")
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
        "test_split": test_split,
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

    print("\nExtracting activations...")
    fens = [p.fen for p in positions]
    activations = extract_features_batch(
        fens,
        layer_name,
        model=model,
        batch_size=batch_size,
    )
    print(f"Activation matrix shape: {activations.shape}")

    if mode == "multi-class":
        training_output = train_multiclass(
            labels=labels,
            activations=activations,
            layer_name=layer_name,
            test_split=test_split,
            random_seed=random_seed,
            verbose=verbose,
            n_jobs=n_jobs,
        )
    else:
        training_output = train_multilabel(
            labels=labels,
            activations=activations,
            layer_name=layer_name,
            test_split=test_split,
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
