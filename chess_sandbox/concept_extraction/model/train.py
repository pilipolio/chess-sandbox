"""
Training pipeline for concept probes.

Trains multi-label logistic regression classifiers to detect chess concepts
from LC0 layer activations.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import click
import numpy as np
from huggingface_hub import hf_hub_download
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

from ..labelling.labeller import LabelledPosition
from .evaluation import calculate_multilabel_metrics, format_metrics_display
from .features import LczeroModel, extract_features_batch
from .hub import upload_probe
from .inference import ConceptProbe


def generate_probe_name(model_repo_id: str, layer_name: str, mode: str, dataset_repo_id: str) -> str:
    """
    Generate a semantic probe name from training parameters.

    Format: {model_name}_{layer_safe}_{mode}_{dataset_hash}

    Args:
        model_repo_id: HuggingFace model repo ID (e.g., "pilipolio/maia-1500")
        layer_name: Layer name (e.g., "block3/conv2/relu")
        mode: Training mode ("multi-label" or "multi-class")
        dataset_repo_id: HuggingFace dataset repo ID

    Returns:
        Generated probe name (e.g., "maia1500_block3_conv2_relu_multilabel_a3f8c2d")
    """
    model_name = model_repo_id.split("/")[-1].replace("-", "").replace("_", "")
    layer_safe = layer_name.replace("/", "_").replace("-", "_")
    mode_safe = mode.replace("-", "")
    dataset_hash = hashlib.sha256(dataset_repo_id.encode()).hexdigest()[:8]
    return f"{model_name}_{layer_safe}_{mode_safe}_{dataset_hash}"


def load_lczero_model_from_hf(repo_id: str, filename: str = "model.onnx", revision: str | None = None) -> LczeroModel:
    """
    Load LC0 model from HuggingFace Hub.

    Args:
        repo_id: HuggingFace model repo ID (e.g., "pilipolio/maia-1500")
        filename: Model filename in the repo (default: "model.onnx")
        revision: Git revision (tag, branch, commit). Defaults to "main"

    Returns:
        Loaded LczeroModel
    """
    model_path = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)
    return LczeroModel.from_path(model_path)


def load_training_dataset_from_hf(
    repo_id: str, filename: str, revision: str | None = None
) -> tuple[list[LabelledPosition], list[list[str]]]:
    """
    Load training dataset from HuggingFace Hub.

    Downloads JSONL file from HF dataset repo and parses it.
    TODO: Consider using `datasets` library for better HF integration.

    Args:
        repo_id: HuggingFace dataset repo ID
        filename: JSONL filename in the repo
        revision: Git revision (tag, branch, commit). Defaults to "main"

    Returns:
        Tuple of (positions, labels)
        - positions: List of LabelledPosition objects
        - labels: list[list[str]] - each position has a list of concept names
    """
    data_path = Path(hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", revision=revision))
    return _parse_training_data(data_path)


def _parse_training_data(data_path: Path) -> tuple[list[LabelledPosition], list[list[str]]]:
    """
    Parse training data from JSONL file.

    Internal helper used by both HF and local loading paths.

    Args:
        data_path: Path to JSONL file with labeled positions

    Returns:
        Tuple of (positions, labels)
        - positions: List of position dictionaries
        - labels: list[list[str]] - each position has a list of concept names
    """
    all_positions: list[LabelledPosition] = []
    with data_path.open() as f:
        for line in f:
            all_positions.append(LabelledPosition.from_dict(json.loads(line)))

    positions_with_concepts = [p for p in all_positions if p.concepts]
    print(f"Loaded {len(all_positions)} positions, kept {len(positions_with_concepts)} with concepts")

    positions: list[LabelledPosition] = []
    labels: list[list[str]] = []

    for pos in positions_with_concepts:
        validated_concepts = [c.name for c in pos.concepts if c.validated_by is not None]
        if validated_concepts:
            positions.append(pos)
            labels.append(validated_concepts)

    print(f"Kept {len(positions)} positions with at least one validated concept")
    return positions, labels


def train_multiclass(
    labels: list[list[str]],
    activations: np.ndarray,
    layer_name: str,
    test_split: float,
    random_seed: int,
    model_version: str,
    source_provenance: dict[str, Any],
    verbose: bool = False,
    n_jobs: int = -1,
) -> ConceptProbe:
    """
    Train multi-class concept probe (one concept per position).

    Takes multi-label format labels and flattens them internally (takes first concept).

    Args:
        labels: List of concept label lists (flattened internally to first concept)
        activations: Pre-extracted activation features
        layer_name: Layer name that activations were extracted from
        test_split: Fraction of data for testing
        random_seed: Random seed for reproducibility
        model_version: Version identifier for the probe
        source_provenance: Provenance metadata (model/dataset repo IDs and filenames)
        verbose: Enable sklearn training progress output
        n_jobs: Number of parallel jobs (-1 = all cores)

    Returns:
        Trained ConceptProbe
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

    probe = ConceptProbe(  # pyright: ignore[reportReturnType]
        classifier=clf,
        concept_list=concept_list,
        layer_name=layer_name,
        training_metrics={
            "baseline": baseline_metrics,
            "probe": probe_metrics,
            "training_samples": X_train.shape[0],
            "test_samples": X_test.shape[0],
            "test_split": test_split,
            "random_seed": random_seed,
            "mode": "multi-class",
            "verbose": verbose,
            "n_jobs": n_jobs,
            **source_provenance,
        },
        training_date=datetime.now().isoformat(),
        model_version=model_version,
        label_encoder=encoder,
    )
    return probe


def train_multilabel(
    labels: list[list[str]],
    activations: np.ndarray,
    layer_name: str,
    test_split: float,
    random_seed: int,
    model_version: str,
    source_provenance: dict[str, Any],
    verbose: bool = False,
    n_jobs: int = -1,
) -> ConceptProbe:
    """
    Train multi-label concept probe (multiple concepts per position).

    Args:
        labels: List of concept label lists (multiple per position)
        activations: Pre-extracted activation features
        layer_name: Layer name that activations were extracted from
        test_split: Fraction of data for testing
        random_seed: Random seed for reproducibility
        model_version: Version identifier for the probe
        source_provenance: Provenance metadata (model/dataset repo IDs and filenames)
        verbose: Enable sklearn training progress output
        n_jobs: Number of parallel jobs (-1 = all cores)

    Returns:
        Trained ConceptProbe
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

    probe = ConceptProbe(  # pyright: ignore[reportReturnType]
        classifier=clf,
        concept_list=concept_list,
        layer_name=layer_name,
        training_metrics={
            "baseline": baseline_metrics,
            "probe": probe_metrics,
            "training_samples": X_train.shape[0],
            "test_samples": X_test.shape[0],
            "test_split": test_split,
            "random_seed": random_seed,
            "mode": "multi-label",
            "verbose": verbose,
            "n_jobs": n_jobs,
            **source_provenance,
        },
        training_date=datetime.now().isoformat(),
        model_version=model_version,
        label_encoder=encoder,
    )
    return probe


@click.command()
@click.option(
    "--dataset-repo-id",
    required=True,
    help="HuggingFace dataset repository ID (e.g., 'pilipolio/chess-concepts-async-100')",
)
@click.option(
    "--dataset-filename",
    required=True,
    help="JSONL filename in the dataset repository",
)
@click.option(
    "--dataset-revision",
    default=None,
    help="Git revision for dataset (tag, branch, commit). Defaults to main",
)
@click.option(
    "--model-repo-id",
    required=True,
    help="HuggingFace model repository ID (e.g., 'pilipolio/maia-1500')",
)
@click.option(
    "--model-filename",
    default="model.onnx",
    help="Model filename in the repository (default: model.onnx)",
)
@click.option(
    "--model-revision",
    default=None,
    help="Git revision for model (tag, branch, commit). Defaults to main",
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
    help="Upload trained probe to HuggingFace Hub after training",
)
def train(
    dataset_repo_id: str,
    dataset_filename: str,
    dataset_revision: str | None,
    model_repo_id: str,
    model_filename: str,
    model_revision: str | None,
    layer_name: str,
    output: Path | None,
    test_split: float,
    random_seed: int,
    batch_size: int,
    mode: str,
    verbose: bool,
    n_jobs: int,
    upload_to_hub: bool,
) -> None:
    """
    Train concept probe from labeled positions.

    Example:
        python -m chess_sandbox.concept_extraction.model.train \\
            --dataset-repo-id pilipolio/chess-concepts-async-100 \\
            --dataset-filename data.jsonl \\
            --model-repo-id pilipolio/maia-1500 \\
            --layer-name block3/conv2/relu \\
            --mode multi-label
    """
    print("Training concept probe...")
    dataset_ref = f"{dataset_repo_id}/{dataset_filename}"
    if dataset_revision:
        dataset_ref += f"@{dataset_revision}"
    print(f"  Dataset: {dataset_ref}")

    model_ref = f"{model_repo_id}/{model_filename}"
    if model_revision:
        model_ref += f"@{model_revision}"
    print(f"  Model: {model_ref}")

    print(f"  Layer: {layer_name}")
    print(f"  Mode: {mode}")
    print(f"  Parallel jobs (n_jobs): {n_jobs}")

    auto_generated_name = generate_probe_name(model_repo_id, layer_name, mode, dataset_repo_id)
    output_path = output or Path("data/models/concept_probes") / auto_generated_name
    print(f"  Output: {output_path}")

    print("\nLoading LC0 model from HuggingFace Hub...")
    model = load_lczero_model_from_hf(model_repo_id, model_filename, model_revision)
    print("Model loaded successfully!")

    print("\nLoading training data from HuggingFace Hub...")
    positions, labels = load_training_dataset_from_hf(dataset_repo_id, dataset_filename, dataset_revision)

    print("\nExtracting activations...")
    fens = [p.fen for p in positions]
    activations = extract_features_batch(
        fens,
        layer_name,
        model=model,
        batch_size=batch_size,
    )
    print(f"Activation matrix shape: {activations.shape}")

    source_provenance = {
        "source_model": {
            "repo_id": model_repo_id,
            "filename": model_filename,
            "revision": model_revision,
        },
        "source_dataset": {
            "repo_id": dataset_repo_id,
            "filename": dataset_filename,
            "revision": dataset_revision,
            "hash": hashlib.sha256(dataset_repo_id.encode()).hexdigest(),
        },
    }

    if mode == "multi-class":
        probe = train_multiclass(
            labels=labels,
            activations=activations,
            layer_name=layer_name,
            test_split=test_split,
            random_seed=random_seed,
            model_version=auto_generated_name,
            source_provenance=source_provenance,
            verbose=verbose,
            n_jobs=n_jobs,
        )
    else:
        probe = train_multilabel(
            labels=labels,
            activations=activations,
            layer_name=layer_name,
            test_split=test_split,
            random_seed=random_seed,
            model_version=auto_generated_name,
            source_provenance=source_provenance,
            verbose=verbose,
            n_jobs=n_jobs,
        )

    probe.save(output_path)
    if upload_to_hub:
        commit = upload_probe(output_path, model_name=auto_generated_name)
        print(f"Successfully uploaded: {commit}")


if __name__ == "__main__":
    train()
