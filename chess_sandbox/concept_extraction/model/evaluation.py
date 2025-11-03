"""
Evaluation metrics for concept probes.

Provides consolidated metric calculation and formatting functions for
multi-label concept classification evaluation.
"""

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, precision_score, recall_score


def calculate_multilabel_metrics(y_true: np.ndarray, y_pred: np.ndarray, concept_list: list[str]) -> dict[str, Any]:
    """
    Calculate comprehensive multi-label classification metrics.

    Computes overall metrics (hamming loss, exact match) and per-concept
    metrics (accuracy, F1, precision, recall, support) for multi-label
    classification problems.

    Args:
        y_true: Binary label matrix of shape (n_samples, n_concepts)
        y_pred: Binary prediction matrix of shape (n_samples, n_concepts)
        concept_list: List of concept names corresponding to columns

    Returns:
        Dictionary containing:
            - hamming_loss: float - Fraction of wrong labels
            - exact_match: float - Fraction of perfectly matched samples
            - per_concept: dict - Per-concept metrics (accuracy, f1, precision, recall, support)

    Example:
        >>> import numpy as np
        >>> y_true = np.array([[1, 0, 1], [0, 1, 0]])
        >>> y_pred = np.array([[1, 0, 1], [0, 1, 0]])
        >>> concepts = ["fork", "pin", "skewer"]
        >>> metrics = calculate_multilabel_metrics(y_true, y_pred, concepts)
        >>> metrics["hamming_loss"]
        0.0
        >>> metrics["exact_match"]
        1.0
        >>> metrics["per_concept"]["fork"]["accuracy"]
        1.0
    """
    hamming = float(hamming_loss(y_true, y_pred))
    exact_match = float(accuracy_score(y_true, y_pred))

    per_concept_metrics = {}
    for i, concept in enumerate(concept_list):
        y_true_i = y_true[:, i]
        y_pred_i = y_pred[:, i]
        support = int(y_true_i.sum())

        if support == 0:
            continue

        acc = float(accuracy_score(y_true_i, y_pred_i))
        f1 = float(f1_score(y_true_i, y_pred_i, zero_division="warn"))  # type: ignore[call-arg]
        precision = float(precision_score(y_true_i, y_pred_i, zero_division="warn"))  # type: ignore[call-arg]
        recall = float(recall_score(y_true_i, y_pred_i, zero_division="warn"))  # type: ignore[call-arg]

        per_concept_metrics[concept] = {
            "accuracy": acc,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "support": support,
        }

    return {
        "hamming_loss": hamming,
        "exact_match": exact_match,
        "per_concept": per_concept_metrics,
    }


def format_metrics_display(metrics: dict[str, Any], name: str, baseline_metrics: dict[str, Any] | None = None) -> str:
    """
    Format evaluation metrics for console display.

    Args:
        metrics: Dictionary from calculate_multilabel_metrics()
        name: Name of the model being evaluated
        baseline_metrics: Optional baseline metrics for comparison

    Returns:
        Formatted string for console output. If baseline_metrics provided,
        includes baseline, probe, and summary comparison sections.

    Example:
        >>> metrics = {
        ...     "hamming_loss": 0.1234,
        ...     "exact_match": 0.8765,
        ...     "per_concept": {
        ...         "fork": {"accuracy": 0.95, "f1": 0.92, "precision": 0.90, "recall": 0.94, "support": 100}
        ...     }
        ... }
        >>> output = format_metrics_display(metrics, "Test Probe")
        >>> "TEST PROBE" in output
        True
        >>> "Hamming Loss: 0.1234" in output
        True
        >>> "fork:" in output
        True
    """
    lines: list[str] = []

    # If baseline provided, format both baseline and probe
    if baseline_metrics is not None:
        # Format baseline
        lines.append("=" * 70)
        lines.append("RANDOM BASELINE")
        lines.append("=" * 70)
        lines.append("\nOverall:")
        lines.append(f"  Hamming Loss: {baseline_metrics['hamming_loss']:.4f}")
        lines.append(f"  Exact Match: {baseline_metrics['exact_match']:.4f}")
        lines.append("\nPer-Concept:")
        for concept, concept_metrics in baseline_metrics["per_concept"].items():
            acc = concept_metrics["accuracy"]
            f1 = concept_metrics["f1"]
            precision = concept_metrics["precision"]
            recall = concept_metrics["recall"]
            support = concept_metrics["support"]
            lines.append(f"  {concept}: acc={acc:.3f} f1={f1:.3f} p={precision:.3f} r={recall:.3f} (n={support})")

        lines.append("")  # Blank line between sections

    # Format main metrics (probe or standalone)
    lines.append("=" * 70)
    lines.append(name.upper())
    lines.append("=" * 70)
    lines.append("\nOverall:")
    lines.append(f"  Hamming Loss: {metrics['hamming_loss']:.4f}")
    lines.append(f"  Exact Match: {metrics['exact_match']:.4f}")
    lines.append("\nPer-Concept:")
    for concept, concept_metrics in metrics["per_concept"].items():
        acc = concept_metrics["accuracy"]
        f1 = concept_metrics["f1"]
        precision = concept_metrics["precision"]
        recall = concept_metrics["recall"]
        support = concept_metrics["support"]
        lines.append(f"  {concept}: acc={acc:.3f} f1={f1:.3f} p={precision:.3f} r={recall:.3f} (n={support})")

    # Add summary comparison if baseline provided
    if baseline_metrics is not None:
        lines.append("")
        lines.append("=" * 70)
        lines.append("SUMMARY")
        lines.append("=" * 70)
        lines.append(f"Baseline Exact Match: {baseline_metrics['exact_match']:.1%}")
        lines.append(f"Probe Exact Match:    {metrics['exact_match']:.1%}")
        improvement = metrics["exact_match"] / (baseline_metrics["exact_match"] + 1e-10)
        lines.append(f"Improvement:          {improvement:.1f}x")
        lines.append("")
        lines.append(f"Baseline Hamming:     {baseline_metrics['hamming_loss']:.1%}")
        lines.append(f"Probe Hamming:        {metrics['hamming_loss']:.1%}")
        error_reduction = (baseline_metrics["hamming_loss"] - metrics["hamming_loss"]) / baseline_metrics[
            "hamming_loss"
        ]
        lines.append(f"Error Reduction:      {error_reduction:.1%}")

    return "\n".join(lines)
