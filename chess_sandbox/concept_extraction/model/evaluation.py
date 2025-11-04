"""
Evaluation metrics for concept probes.

Provides consolidated metric calculation and formatting functions for
multi-label concept classification evaluation.
"""

from pathlib import Path

import click
import numpy as np
from pydantic import BaseModel
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer

from .dataset import load_dataset_from_hf
from .inference import ConceptExtractor, hf_hub_options


class ConceptMetrics(BaseModel):
    """Per-concept evaluation metrics."""

    precision: float
    recall: float
    support: int


class MultiLabelMetrics(BaseModel):
    """Multi-label classification metrics with micro/macro averages."""

    micro_precision: float
    micro_recall: float
    macro_precision: float
    macro_recall: float
    per_concept: dict[str, ConceptMetrics]


def calculate_multilabel_metrics(y_true: np.ndarray, y_pred: np.ndarray, concept_list: list[str]) -> MultiLabelMetrics:
    """
    Calculate multi-label classification metrics with micro/macro averages.

    Computes micro and macro-averaged precision/recall, and per-concept
    precision, recall, and support for multi-label classification problems.

    Args:
        y_true: Binary label matrix of shape (n_samples, n_concepts)
        y_pred: Binary prediction matrix of shape (n_samples, n_concepts)
        concept_list: List of concept names corresponding to columns

    Returns:
        MultiLabelMetrics object containing:
            - micro_precision: Micro-averaged precision across all labels
            - micro_recall: Micro-averaged recall across all labels
            - macro_precision: Macro-averaged precision across all labels
            - macro_recall: Macro-averaged recall across all labels
            - per_concept: Per-concept precision, recall, and support

    Example:
        >>> import numpy as np
        >>> y_true = np.array([[1, 0, 1], [0, 1, 0]])
        >>> y_pred = np.array([[1, 0, 1], [0, 1, 0]])
        >>> concepts = ["fork", "pin", "skewer"]
        >>> metrics = calculate_multilabel_metrics(y_true, y_pred, concepts)
        >>> metrics.micro_precision
        1.0
        >>> metrics.per_concept["fork"].precision
        1.0
    """
    # Calculate micro and macro averages
    micro_precision = float(precision_score(y_true, y_pred, average="micro", zero_division=0.0))
    micro_recall = float(recall_score(y_true, y_pred, average="micro", zero_division=0.0))
    macro_precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0.0))
    macro_recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0.0))

    # Calculate per-concept metrics
    per_concept_metrics = {}
    for i, concept in enumerate(concept_list):
        y_true_i = y_true[:, i]
        y_pred_i = y_pred[:, i]
        support = int(y_true_i.sum())

        if support == 0:
            continue

        precision = float(precision_score(y_true_i, y_pred_i, zero_division=0.0))
        recall = float(recall_score(y_true_i, y_pred_i, zero_division=0.0))

        per_concept_metrics[concept] = ConceptMetrics(
            precision=precision,
            recall=recall,
            support=support,
        )

    return MultiLabelMetrics(
        micro_precision=micro_precision,
        micro_recall=micro_recall,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        per_concept=per_concept_metrics,
    )


def format_metrics_display(
    metrics: MultiLabelMetrics, name: str, baseline_metrics: MultiLabelMetrics | None = None
) -> str:
    """
    Format evaluation metrics for console display.

    Args:
        metrics: MultiLabelMetrics from calculate_multilabel_metrics()
        name: Name of the model being evaluated
        baseline_metrics: Optional baseline metrics for comparison

    Returns:
        Formatted string for console output. If baseline_metrics provided,
        includes baseline, probe, and summary comparison sections.
    """
    lines: list[str] = []

    # If baseline provided, format both baseline and probe
    if baseline_metrics is not None:
        # Format baseline
        lines.append("=" * 70)
        lines.append("RANDOM BASELINE")
        lines.append("=" * 70)
        lines.append("\nOverall:")
        lines.append(f"  Micro Precision: {baseline_metrics.micro_precision:.4f}")
        lines.append(f"  Micro Recall:    {baseline_metrics.micro_recall:.4f}")
        lines.append(f"  Macro Precision: {baseline_metrics.macro_precision:.4f}")
        lines.append(f"  Macro Recall:    {baseline_metrics.macro_recall:.4f}")
        lines.append("\nPer-Concept:")
        for concept, concept_metrics in baseline_metrics.per_concept.items():
            lines.append(
                f"  {concept}: p={concept_metrics.precision:.3f} "
                f"r={concept_metrics.recall:.3f} (n={concept_metrics.support})"
            )

        lines.append("")  # Blank line between sections

    # Format main metrics (probe or standalone)
    lines.append("=" * 70)
    lines.append(name.upper())
    lines.append("=" * 70)
    lines.append("\nOverall:")
    lines.append(f"  Micro Precision: {metrics.micro_precision:.4f}")
    lines.append(f"  Micro Recall:    {metrics.micro_recall:.4f}")
    lines.append(f"  Macro Precision: {metrics.macro_precision:.4f}")
    lines.append(f"  Macro Recall:    {metrics.macro_recall:.4f}")
    lines.append("\nPer-Concept:")
    for concept, concept_metrics in metrics.per_concept.items():
        lines.append(
            f"  {concept}: p={concept_metrics.precision:.3f} "
            f"r={concept_metrics.recall:.3f} (n={concept_metrics.support})"
        )

    # Add summary comparison if baseline provided
    if baseline_metrics is not None:
        lines.append("")
        lines.append("=" * 70)
        lines.append("SUMMARY")
        lines.append("=" * 70)
        lines.append(f"Baseline Micro Precision: {baseline_metrics.micro_precision:.1%}")
        lines.append(f"Probe Micro Precision:    {metrics.micro_precision:.1%}")
        precision_improvement = metrics.micro_precision / (baseline_metrics.micro_precision + 1e-10)
        lines.append(f"Improvement:              {precision_improvement:.1f}x")
        lines.append("")
        lines.append(f"Baseline Micro Recall:    {baseline_metrics.micro_recall:.1%}")
        lines.append(f"Probe Micro Recall:       {metrics.micro_recall:.1%}")
        recall_improvement = metrics.micro_recall / (baseline_metrics.micro_recall + 1e-10)
        lines.append(f"Improvement:              {recall_improvement:.1f}x")

    return "\n".join(lines)


def binarize_predictions(
    predictions_with_confidence: list[list[tuple[str, float]]],
    ground_truth_labels: list[list[str]],
    concept_list: list[str],
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert predictions with confidence scores and labels to binary matrices.

    Args:
        predictions_with_confidence: List of predictions per sample, each with (concept, confidence)
        ground_truth_labels: List of ground truth concept lists per sample
        concept_list: Complete list of all possible concepts
        threshold: Confidence threshold for considering a prediction positive

    Returns:
        Tuple of (y_true, y_pred) as dense numpy binary matrices
    """
    # Initialize MultiLabelBinarizer with all concepts
    mlb = MultiLabelBinarizer()
    mlb.fit([concept_list])

    # Transform ground truth labels to binary matrix
    y_true_transformed = mlb.transform(ground_truth_labels)

    # Filter predictions by threshold and extract concept names
    y_pred_concepts = [
        [name for name, confidence in predictions if confidence >= threshold]
        for predictions in predictions_with_confidence
    ]
    y_pred_transformed = mlb.transform(y_pred_concepts)

    # Convert sparse matrices to dense numpy arrays
    y_true = (
        np.asarray(y_true_transformed.toarray())
        if hasattr(y_true_transformed, "toarray")
        else np.asarray(y_true_transformed)
    )
    y_pred = (
        np.asarray(y_pred_transformed.toarray())
        if hasattr(y_pred_transformed, "toarray")
        else np.asarray(y_pred_transformed)
    )

    return y_true, y_pred


@click.group()
def cli() -> None:
    """Evaluation commands for concept probes."""
    pass


@cli.command()
@hf_hub_options
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
    "--sample-size",
    default=10,
    type=int,
    help="Number of sample predictions to display",
)
@click.option(
    "--batch-size",
    default=32,
    type=int,
    help="Batch size for activation extraction",
)
@click.option(
    "--random-seed",
    default=42,
    type=int,
    help="Random seed for reproducibility",
)
@click.option(
    "--show-samples/--no-show-samples",
    default=True,
    help="Show individual sample predictions",
)
@click.option(
    "--threshold",
    default=0.5,
    type=float,
    help="Confidence threshold for positive predictions",
)
def evaluate(
    model_repo_id: str,
    revision: str | None,
    lc0_repo_id: str,
    lc0_filename: str,
    cache_dir: Path | None,
    force_download: bool,
    token: str | None,
    dataset_repo_id: str,
    dataset_filename: str,
    dataset_revision: str | None,
    sample_size: int,
    batch_size: int,
    random_seed: int,
    show_samples: bool,
    threshold: float,
) -> None:
    """
    Evaluate trained concept probe on test data with comprehensive metrics.

    Loads a trained probe from HuggingFace Hub, evaluates it on test data,
    and displays both sample predictions and overall evaluation metrics including
    per-concept accuracy, F1, precision, and recall.

    Example:
        python -m chess_sandbox.concept_extraction.model.evaluation evaluate \\
            --model-repo-id pilipolio/chess-positions-extractor \\
            --dataset-repo-id pilipolio/chess-concepts-async-100 \\
            --dataset-filename data.jsonl \\
            --sample-size 10
    """
    # Display dataset reference
    dataset_ref = f"{dataset_repo_id}/{dataset_filename}"
    if dataset_revision:
        dataset_ref += f"@{dataset_revision}"
    print("Evaluating concept probe")
    print(f"  Model: {model_repo_id}")
    print(f"  Dataset: {dataset_ref}")

    print("\nLoading ConceptExtractor from HuggingFace Hub...")
    extractor = ConceptExtractor.from_hf(
        probe_repo_id=model_repo_id,
        model_repo_id=lc0_repo_id,
        model_filename=lc0_filename,
        revision=revision,
        cache_dir=cache_dir,
        force_download=force_download,
        token=token,
    )
    print(f"Loaded probe: {extractor.probe}")
    print(f"Concepts: {', '.join(extractor.probe.concept_list)}")

    print("\nLoading evaluation dataset from HuggingFace Hub...")
    positions, labels = load_dataset_from_hf(dataset_repo_id, dataset_filename, dataset_revision)

    if len(positions) == 0:
        print("No positions with validated concepts found!")
        return

    all_fens = [p.fen for p in positions]
    all_predictions_with_confidence = extractor.extract_concepts_with_confidence(all_fens)

    # Show sample predictions if requested
    if show_samples:
        n_samples = min(sample_size, len(positions))
        rng = np.random.RandomState(random_seed)
        sample_indices = rng.choice(len(positions), size=n_samples, replace=False)
        sample_fens = [all_fens[i] for i in sample_indices]
        sample_labels = [labels[i] for i in sample_indices]
        sample_predictions_with_confidence = [all_predictions_with_confidence[i] for i in sample_indices]

        print(f"\n{'=' * 70}")
        print(f"SAMPLE PREDICTIONS ({n_samples} examples)")
        print(f"{'=' * 70}\n")

        sample_correct = 0
        for pos, ground_truth, predicted_concepts in zip(
            sample_fens, sample_labels, sample_predictions_with_confidence, strict=True
        ):
            print(f"FEN: {pos}")

            above_threshold_extracted_concept = [c for c, score in predicted_concepts if score >= threshold]

            # Check if prediction matches
            match_marker = "✓" if set(ground_truth) == set(above_threshold_extracted_concept) else "✗"

            gt_str = ", ".join(ground_truth) if ground_truth else "(none)"
            pred_str = ", ".join(above_threshold_extracted_concept) if above_threshold_extracted_concept else "(none)"
            print(f"Ground Truth: {gt_str}")
            print(f"Prediction:   {pred_str} {match_marker}")
            print()

            if set(ground_truth) == set(above_threshold_extracted_concept):
                sample_correct += 1

        sample_accuracy = sample_correct / n_samples
        print(f"{'=' * 70}")
        print(f"Sample Exact Match Rate: {sample_accuracy:.1%} ({sample_correct}/{n_samples})")
        print(f"{'=' * 70}\n")

    # Calculate comprehensive metrics on full dataset
    print("\nCalculating comprehensive metrics on full dataset...")
    print(f"Extracting activations for {len(positions)} positions...")

    # Convert predictions to binary matrices for metric calculation
    y_true, y_pred = binarize_predictions(
        all_predictions_with_confidence, labels, extractor.probe.concept_list, threshold
    )

    # Calculate metrics
    metrics = calculate_multilabel_metrics(y_true, y_pred, extractor.probe.concept_list)

    # Display formatted metrics
    print("\n" + format_metrics_display(metrics, "Concept Probe Evaluation"))
    print()


if __name__ == "__main__":
    cli()
