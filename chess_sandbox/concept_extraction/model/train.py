"""
Training pipeline for concept probes.

Trains multi-label logistic regression classifiers to detect chess concepts
from LC0 layer activations.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import click
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, hamming_loss
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

from ..labelling.labeller import LabelledPosition
from .features import LczeroModel, extract_features_batch
from .hub import upload_probe
from .inference import ConceptProbe


def load_training_data(data_path: Path) -> tuple[list[LabelledPosition], list[list[str]]]:
    """
    Load positions and extract concept labels from JSONL file.

    Always returns labels as list of list of concepts (multi-label format).
    Positions with multiple validated concepts keep all of them.

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

    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, name: str) -> dict[str, Any]:
        hamming = float(hamming_loss(y_true, y_pred))
        exact_match = float(accuracy_score(y_true, y_pred))

        print(f"\n{'=' * 70}")
        print(f"{name.upper()}")
        print(f"{'=' * 70}")
        print("\nOverall:")
        print(f"  Hamming Loss: {hamming:.4f}")
        print(f"  Exact Match: {exact_match:.4f}")

        print("\nPer-Concept:")
        per_concept_metrics = {}
        for i, concept in enumerate(concept_list):
            y_true_i = y_true[:, i]
            y_pred_i = y_pred[:, i]
            support = int(y_true_i.sum())

            if support == 0:
                continue

            acc = float(accuracy_score(y_true_i, y_pred_i))
            f1 = float(f1_score(y_true_i, y_pred_i, zero_division="warn"))  # type: ignore[call-arg]

            per_concept_metrics[concept] = {
                "accuracy": acc,
                "f1": f1,
                "support": support,
            }

            print(f"  {concept}: acc={acc:.3f} f1={f1:.3f} (n={support})")

        return {
            "hamming_loss": hamming,
            "exact_match": exact_match,
            "per_concept": per_concept_metrics,
        }

    baseline_metrics = calculate_metrics(y_test_binary, y_pred_baseline_binary, "Random Baseline")
    probe_metrics = calculate_metrics(y_test_binary, y_pred_probe_binary, "Trained Probe")

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    print(f"Baseline Exact Match: {baseline_metrics['exact_match']:.1%}")
    print(f"Probe Exact Match:    {probe_metrics['exact_match']:.1%}")
    improvement = probe_metrics["exact_match"] / (baseline_metrics["exact_match"] + 1e-10)
    print(f"Improvement:          {improvement:.1f}x")
    print()
    print(f"Baseline Hamming:     {baseline_metrics['hamming_loss']:.1%}")
    print(f"Probe Hamming:        {probe_metrics['hamming_loss']:.1%}")
    error_reduction = (baseline_metrics["hamming_loss"] - probe_metrics["hamming_loss"]) / baseline_metrics[
        "hamming_loss"
    ]
    print(f"Error Reduction:      {error_reduction:.1%}")

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

    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, name: str) -> dict[str, Any]:
        hamming = float(hamming_loss(y_true, y_pred))
        exact_match = float(accuracy_score(y_true, y_pred))

        print(f"\n{'=' * 70}")
        print(f"{name.upper()}")
        print(f"{'=' * 70}")
        print("\nOverall:")
        print(f"  Hamming Loss: {hamming:.4f}")
        print(f"  Exact Match: {exact_match:.4f}")

        print("\nPer-Concept:")
        per_concept_metrics = {}
        for i, concept in enumerate(concept_list):
            y_true_i = y_true[:, i]
            y_pred_i = y_pred[:, i]
            support = int(y_true_i.sum())

            if support == 0:
                continue

            acc = float(accuracy_score(y_true_i, y_pred_i))
            f1 = float(f1_score(y_true_i, y_pred_i, zero_division="warn"))  # type: ignore[call-arg]

            per_concept_metrics[concept] = {
                "accuracy": acc,
                "f1": f1,
                "support": support,
            }

            print(f"  {concept}: acc={acc:.3f} f1={f1:.3f} (n={support})")

        return {
            "hamming_loss": hamming,
            "exact_match": exact_match,
            "per_concept": per_concept_metrics,
        }

    baseline_metrics = calculate_metrics(y_test_binary, y_pred_baseline, "Random Baseline")
    probe_metrics = calculate_metrics(y_test_binary, y_pred_probe, "Trained Probe")

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    print(f"Baseline Exact Match: {baseline_metrics['exact_match']:.1%}")
    print(f"Probe Exact Match:    {probe_metrics['exact_match']:.1%}")
    improvement = probe_metrics["exact_match"] / (baseline_metrics["exact_match"] + 1e-10)
    print(f"Improvement:          {improvement:.1f}x")
    print()
    print(f"Baseline Hamming:     {baseline_metrics['hamming_loss']:.1%}")
    print(f"Probe Hamming:        {probe_metrics['hamming_loss']:.1%}")
    error_reduction = (baseline_metrics["hamming_loss"] - probe_metrics["hamming_loss"]) / baseline_metrics[
        "hamming_loss"
    ]
    print(f"Error Reduction:      {error_reduction:.1%}")

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
        },
        training_date=datetime.now().isoformat(),
        model_version=model_version,
        label_encoder=encoder,
    )
    return probe


@click.command()
@click.option(
    "--data-path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to JSONL file with labeled positions",
)
@click.option(
    "--model-path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to LC0 model file (e.g., maia-1500.onnx)",
)
@click.option(
    "--layer-name",
    default="block3/conv2/relu",
    help="Layer name to extract activations from",
)
@click.option(
    "--output",
    required=True,
    type=click.Path(path_type=Path),
    help="Path to save trained probe",
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
    "--model-version",
    default="v1",
    help="Version identifier for the trained probe",
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
    data_path: Path,
    model_path: Path,
    layer_name: str,
    output: Path,
    test_split: float,
    random_seed: int,
    model_version: str,
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
            --data-path data/positions.jsonl \\
            --model-path models/maia-1500.onnx \\
            --output models/concept_probes/probe_v1.pkl \\
            --verbose --n-jobs -1
    """
    print("Training concept probe...")
    print(f"  Data: {data_path}")
    print(f"  Model: {model_path}")
    print(f"  Layer: {layer_name}")
    print(f"  Mode: {mode}")
    print(f"  Output: {output}")
    print(f"  Parallel jobs (n_jobs): {n_jobs}")

    print("\nLoading LC0 model...")
    model = LczeroModel.from_path(str(model_path))
    print("Model loaded successfully!")

    print("\nLoading training data...")
    positions, labels = load_training_data(data_path)

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
        probe = train_multiclass(
            labels=labels,
            activations=activations,
            layer_name=layer_name,
            test_split=test_split,
            random_seed=random_seed,
            model_version=model_version,
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
            model_version=model_version,
            verbose=verbose,
            n_jobs=n_jobs,
        )

    probe.save(output)
    if upload_to_hub:
        commit = upload_probe(output, model_name="chess-sandbox-concept-probes")
        print(f"Successfully uploaded: {commit}")


if __name__ == "__main__":
    train()
