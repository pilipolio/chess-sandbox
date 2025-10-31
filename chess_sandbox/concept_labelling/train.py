"""
Training pipeline for concept probes.

Trains multi-label logistic regression classifiers to detect chess concepts
from LC0 layer activations.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Any

import click
import numpy as np
from sklearn.dummy import DummyClassifier  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.metrics import accuracy_score, f1_score, hamming_loss  # type: ignore[import-untyped]
from sklearn.model_selection import train_test_split  # type: ignore[import-untyped]
from sklearn.multiclass import OneVsRestClassifier  # type: ignore[import-untyped]
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer  # type: ignore[import-untyped]

from .features import extract_features_batch
from .inference import create_probe


def load_training_data(
    data_path: Path, mode: str = "multi-label"
) -> tuple[list[dict[str, Any]], list[str] | list[list[str]], list[str]]:
    """
    Load positions and extract concept labels from JSONL file.

    Args:
        data_path: Path to JSONL file with labeled positions
        mode: "multi-class" (one concept per position) or "multi-label" (multiple concepts per position)

    Returns:
        Tuple of (positions, labels, concept_list)
        - positions: List of position dictionaries
        - labels: list[str] for multi-class, list[list[str]] for multi-label
        - concept_list: Sorted list of all unique concept names
    """
    if mode not in ("multi-class", "multi-label"):
        raise ValueError(f"Invalid mode: {mode}. Must be 'multi-class' or 'multi-label'")

    all_positions: list[dict[str, Any]] = []
    with data_path.open() as f:
        for line in f:
            all_positions.append(json.loads(line))

    positions_with_concepts = [p for p in all_positions if p.get("concepts")]
    print(f"Loaded {len(all_positions)} positions, kept {len(positions_with_concepts)} with concepts")

    all_concepts: set[str] = set()
    for pos in positions_with_concepts:
        concepts_data: Any = pos["concepts"]
        for concept_dict in concepts_data:
            validated_by: Any = concept_dict.get("validated_by")
            if validated_by is not None:
                concept_name_str: str = str(concept_dict["name"])
                all_concepts.add(concept_name_str)

    concept_list = sorted(all_concepts)
    print(f"Found {len(concept_list)} unique concepts: {concept_list}")

    positions: list[dict[str, Any]] = []
    labels: list[str] | list[list[str]]

    if mode == "multi-class":
        labels_single: list[str] = []
        for pos in positions_with_concepts:
            validated_concepts = [str(c["name"]) for c in pos["concepts"] if c.get("validated_by") is not None]
            if len(validated_concepts) >= 1:
                positions.append(pos)
                labels_single.append(validated_concepts[0])
        labels = labels_single
        print(f"Multi-class mode: kept {len(positions)} positions with at least one validated concept")
    else:
        labels_multi: list[list[str]] = []
        for pos in positions_with_concepts:
            validated_concepts = [str(c["name"]) for c in pos["concepts"] if c.get("validated_by") is not None]
            if validated_concepts:
                positions.append(pos)
                labels_multi.append(validated_concepts)
        labels = labels_multi
        print(f"Multi-label mode: kept {len(positions)} positions with at least one validated concept")

    label_counts: Counter[str] = Counter()
    if mode == "multi-class":
        label_counts.update(labels)  # type: ignore[arg-type]
    else:
        for concept_list_item in labels:  # type: ignore[union-attr]
            label_counts.update(concept_list_item)

    print("\nConcept distribution:")
    for concept_name, count in label_counts.most_common():
        print(f"  {concept_name}: {count}")

    return positions, labels, concept_list


def evaluate_classifier(
    clf: Any,
    X_test: np.ndarray,  # noqa: N803 (sklearn convention)
    y_test: np.ndarray,
    concept_list: list[str],
    name: str = "Classifier",
    mode: str = "multi-label",
) -> dict[str, Any]:
    """
    Evaluate classifier and print metrics.

    Args:
        clf: Trained classifier
        X_test: Test features
        y_test: Test labels (1D for multi-class, 2D for multi-label)
        concept_list: List of concept names
        name: Name for display
        mode: "multi-class" or "multi-label"

    Returns:
        Dictionary of metrics
    """
    y_pred: Any = clf.predict(X_test)

    if mode == "multi-class":
        y_test_eval = y_test.ravel() if y_test.ndim > 1 else y_test
        y_pred_eval = y_pred.ravel() if y_pred.ndim > 1 else y_pred
    else:
        y_test_eval = y_test
        y_pred_eval = y_pred

    print(f"\n{'=' * 70}")
    print(f"{name.upper()}")
    print(f"{'=' * 70}")

    if mode == "multi-class":
        accuracy = float(accuracy_score(y_test_eval, y_pred_eval))
        f1_macro = float(f1_score(y_test_eval, y_pred_eval, average="macro", zero_division="warn"))  # type: ignore[call-arg]
        f1_weighted = float(f1_score(y_test_eval, y_pred_eval, average="weighted", zero_division="warn"))  # type: ignore[call-arg]

        print("\nOverall:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 (macro): {f1_macro:.4f}")
        print(f"  F1 (weighted): {f1_weighted:.4f}")

        print("\nPer-Concept:")
        per_concept_metrics = {}
        for i, concept in enumerate(concept_list):
            mask = y_test_eval == i
            support = int(mask.sum())

            if support == 0:
                continue

            y_true_binary = (y_test_eval == i).astype(int)
            y_pred_binary = (y_pred_eval == i).astype(int)

            acc = float(accuracy_score(y_true_binary, y_pred_binary))
            f1 = float(f1_score(y_true_binary, y_pred_binary, zero_division="warn"))  # type: ignore[call-arg]

            per_concept_metrics[concept] = {
                "accuracy": acc,
                "f1": f1,
                "support": support,
            }

            print(f"  {concept}: acc={acc:.3f} f1={f1:.3f} (n={support})")

        return {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "per_concept": per_concept_metrics,
        }
    else:
        hamming = float(hamming_loss(y_test_eval, y_pred_eval))  # type: ignore[arg-type]
        exact_match = float(accuracy_score(y_test_eval, y_pred_eval))

        print("\nOverall:")
        print(f"  Hamming Loss: {hamming:.4f}")
        print(f"  Exact Match: {exact_match:.4f}")

        print("\nPer-Concept:")
        per_concept_metrics = {}
        for i, concept in enumerate(concept_list):
            y_true_i = y_test_eval[:, i]
            y_pred_i = y_pred_eval[:, i]
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
    help="Path to LC0 model file (e.g., maia-1500.pt)",
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
) -> None:
    """
    Train concept probe from labeled positions.

    Example:
        python -m chess_sandbox.concept_labelling.train \\
            --data-path data/positions.jsonl \\
            --model-path models/maia-1500.pt \\
            --output models/concept_probes/probe_v1.pkl
    """
    print("Training concept probe...")
    print(f"  Data: {data_path}")
    print(f"  Model: {model_path}")
    print(f"  Layer: {layer_name}")
    print(f"  Mode: {mode}")
    print(f"  Output: {output}")

    print("\n[1/6] Loading data...")
    positions, labels, concept_list = load_training_data(data_path, mode=mode)

    if not labels:
        print("\nWARNING: No validated concepts found! Cannot train without labels.")
        return

    print("\n[2/6] Extracting activations...")
    fens = [p["fen"] for p in positions]
    activations = extract_features_batch(
        fens,
        model_path,
        layer_name,
        batch_size=batch_size,
    )
    print(f"Activation matrix shape: {activations.shape}")

    print("\n[3/6] Encoding labels...")
    encoder: LabelEncoder | MultiLabelBinarizer
    label_matrix: np.ndarray

    if mode == "multi-class":
        encoder = LabelEncoder()
        label_matrix = encoder.fit_transform(labels).reshape(-1, 1)  # type: ignore[arg-type]
        print(f"Label encoder classes: {encoder.classes_}")  # type: ignore[reportUnknownMemberType]
        print(f"Encoded labels shape: {label_matrix.shape}")  # type: ignore[reportUnknownMemberType]
    else:
        encoder = MultiLabelBinarizer()
        label_matrix = encoder.fit_transform(labels)  # type: ignore[arg-type]
        print(f"Multi-label binarizer classes: {encoder.classes_}")
        print(f"Binary label matrix shape: {label_matrix.shape}")
        print(f"Total labels: {label_matrix.sum()} ({label_matrix.sum() / label_matrix.size * 100:.1f}% density)")

    print("\n[4/6] Splitting data...")
    X_train: Any  # noqa: N806 (sklearn convention)
    X_test: Any  # noqa: N806 (sklearn convention)
    y_train: Any
    y_test: Any
    X_train, X_test, y_train, y_test = train_test_split(  # noqa: N806 (sklearn convention)
        activations, label_matrix, test_size=test_split, random_state=random_seed
    )
    print(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")

    print("\n[5/6] Training probe...")
    base_clf = LogisticRegression(max_iter=10000, random_state=random_seed)

    if mode == "multi-class":
        clf: Any = base_clf
        y_train_fit = y_train.ravel()
        print(f"Training multi-class classifier with {len(encoder.classes_)} classes")  # type: ignore[attr-defined]
    else:
        clf = OneVsRestClassifier(base_clf)
        y_train_fit = y_train
        print(f"Training multi-label classifier with {len(encoder.classes_)} concepts")  # type: ignore[attr-defined]

    clf.fit(X_train, y_train_fit)
    print("Training complete!")

    print("\n[6/6] Evaluating...")
    baseline: Any = DummyClassifier(strategy="stratified", random_state=random_seed)
    baseline.fit(X_train, y_train_fit)

    baseline_metrics = evaluate_classifier(baseline, X_test, y_test, concept_list, "Random Baseline", mode=mode)
    probe_metrics = evaluate_classifier(clf, X_test, y_test, concept_list, "Trained Probe", mode=mode)

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    if mode == "multi-class":
        print(f"Baseline Accuracy:    {baseline_metrics['accuracy']:.1%}")
        print(f"Probe Accuracy:       {probe_metrics['accuracy']:.1%}")
        improvement = probe_metrics["accuracy"] / (baseline_metrics["accuracy"] + 1e-10)
        print(f"Improvement:          {improvement:.1f}x")
        print()
        print(f"Baseline F1 (macro):  {baseline_metrics['f1_macro']:.1%}")
        print(f"Probe F1 (macro):     {probe_metrics['f1_macro']:.1%}")
        print()
        print(f"Baseline F1 (wtd):    {baseline_metrics['f1_weighted']:.1%}")
        print(f"Probe F1 (wtd):       {probe_metrics['f1_weighted']:.1%}")
    else:
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

    print(f"\nSaving to {output}...")
    probe = create_probe(
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
            "data_path": str(data_path),
            "model_path": str(model_path),
            "mode": mode,
        },
        model_version=model_version,
        label_encoder=encoder,
    )
    probe.save(output)

    print("Done!")


if __name__ == "__main__":
    train()
