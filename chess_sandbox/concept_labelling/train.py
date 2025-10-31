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

from .features import extract_features_batch
from .inference import create_probe


def load_training_data(data_path: Path) -> tuple[list[dict[str, Any]], np.ndarray, list[str]]:
    """
    Load positions and build label matrix from JSONL file.

    Args:
        data_path: Path to JSONL file with labeled positions

    Returns:
        Tuple of (positions, label_matrix, concept_list)
    """
    all_positions: list[dict[str, Any]] = []
    with data_path.open() as f:
        for line in f:
            all_positions.append(json.loads(line))

    positions = [p for p in all_positions if p.get("concepts")]
    print(f"Loaded {len(all_positions)} positions, kept {len(positions)} with concepts")

    all_concepts: set[str] = set()
    for pos in positions:
        concepts_data: Any = pos["concepts"]
        for concept_dict in concepts_data:
            validated_by: Any = concept_dict.get("validated_by")
            if validated_by is not None:
                concept_name_str: str = str(concept_dict["name"])
                all_concepts.add(concept_name_str)

    concept_list = sorted(all_concepts)
    concept_to_idx = {c: i for i, c in enumerate(concept_list)}

    print(f"Found {len(concept_list)} unique concepts: {concept_list}")

    label_matrix = np.zeros((len(positions), len(concept_list)), dtype=int)
    for i, pos in enumerate(positions):
        concepts_data: Any = pos["concepts"]
        for concept_dict in concepts_data:
            validated_by: Any = concept_dict.get("validated_by")
            if validated_by is not None:
                concept_name: Any = concept_dict["name"]
                if concept_name in concept_to_idx:
                    label_matrix[i, concept_to_idx[concept_name]] = 1

    print(f"Label matrix shape: {label_matrix.shape}")
    print(f"Total labels: {label_matrix.sum()} ({label_matrix.sum() / label_matrix.size * 100:.1f}% density)")

    label_counts: Counter[str] = Counter()
    for i, concept_name in enumerate(concept_list):
        count = int(label_matrix[:, i].sum())
        label_counts[concept_name] = count

    print("\nConcept distribution:")
    for concept_name, count in label_counts.most_common():
        print(f"  {concept_name}: {count}")

    return positions, label_matrix, concept_list


def evaluate_classifier(
    clf: Any,
    X_test: np.ndarray,  # noqa: N803 (sklearn convention)
    y_test: np.ndarray,
    concept_list: list[str],
    name: str = "Classifier",
) -> dict[str, Any]:
    """
    Evaluate classifier and print metrics.

    Args:
        clf: Trained classifier
        X_test: Test features
        y_test: Test labels
        concept_list: List of concept names
        name: Name for display

    Returns:
        Dictionary of metrics
    """
    y_pred: Any = clf.predict(X_test)

    print(f"\n{'=' * 70}")
    print(f"{name.upper()}")
    print(f"{'=' * 70}")

    hamming = float(hamming_loss(y_test, y_pred))  # type: ignore[arg-type]
    exact_match = float(accuracy_score(y_test, y_pred))

    print("\nOverall:")
    print(f"  Hamming Loss: {hamming:.4f}")
    print(f"  Exact Match: {exact_match:.4f}")

    print("\nPer-Concept:")
    per_concept_metrics = {}
    for i, concept in enumerate(concept_list):
        y_true_i = y_test[:, i]
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
def train(
    data_path: Path,
    model_path: Path,
    layer_name: str,
    output: Path,
    test_split: float,
    random_seed: int,
    model_version: str,
    batch_size: int,
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
    print(f"  Output: {output}")

    print("\n[1/5] Loading data...")
    positions, label_matrix, concept_list = load_training_data(data_path)

    if label_matrix.sum() == 0:
        print("\nWARNING: No validated concepts found! Cannot train without labels.")
        return

    print("\n[2/5] Extracting activations...")
    fens = [p["fen"] for p in positions]
    activations = extract_features_batch(
        fens,
        model_path,
        layer_name,
        batch_size=batch_size,
        show_progress=True,
    )
    print(f"Activation matrix shape: {activations.shape}")

    print("\n[3/5] Splitting data...")
    X_train: Any  # noqa: N806 (sklearn convention)
    X_test: Any  # noqa: N806 (sklearn convention)
    y_train: Any
    y_test: Any
    X_train, X_test, y_train, y_test = train_test_split(  # noqa: N806 (sklearn convention)
        activations, label_matrix, test_size=test_split, random_state=random_seed
    )
    print(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")

    print("\n[4/5] Training probe...")
    clf: Any = OneVsRestClassifier(LogisticRegression(max_iter=10000, random_state=random_seed))
    clf.fit(X_train, y_train)
    print("Training complete!")

    print("\n[5/5] Evaluating...")
    baseline: Any = OneVsRestClassifier(DummyClassifier(strategy="stratified", random_state=random_seed))
    baseline.fit(X_train, y_train)

    baseline_metrics = evaluate_classifier(baseline, X_test, y_test, concept_list, "Random Baseline")
    probe_metrics = evaluate_classifier(clf, X_test, y_test, concept_list, "Trained Probe")

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
        },
        model_version=model_version,
    )
    probe.save(output)

    print("Done!")


if __name__ == "__main__":
    train()
