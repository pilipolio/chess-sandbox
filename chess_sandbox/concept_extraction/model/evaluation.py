"""
Evaluation metrics for concept probes.

Provides consolidated metric calculation and formatting functions for
multi-label concept classification evaluation.
"""

from pathlib import Path

import click
import numpy as np
from pydantic import BaseModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import MultiLabelBinarizer

from chess_sandbox.lichess import get_analysis_url

from .dataset import load_dataset_from_hf
from .inference import ConceptExtractor, hf_hub_options


class ConceptMetrics(BaseModel):
    """Per-concept evaluation metrics."""

    precision: float
    recall: float
    support: int
    auc: float
    subset_accuracy: float


class MultiLabelMetrics(BaseModel):
    """Multi-label classification metrics with micro/macro averages."""

    micro_precision: float
    micro_recall: float
    macro_precision: float
    macro_recall: float
    subset_accuracy: float
    micro_auc: float
    macro_auc: float
    per_concept: dict[str, ConceptMetrics]


def calculate_multilabel_metrics(
    y_true: np.ndarray, y_score: np.ndarray, concept_list: list[str], threshold: float = 0.5
) -> MultiLabelMetrics:
    """
    Calculate multi-label classification metrics with micro/macro averages.

    Computes micro and macro-averaged precision/recall/AUC, and per-concept
    precision, recall, AUC and support for multi-label classification problems.

    Args:
        y_true: Binary label matrix of shape (n_samples, n_concepts)
        y_score: Probability score matrix of shape (n_samples, n_concepts)
        concept_list: List of concept names corresponding to columns
        threshold: Threshold for binarizing probability scores (default: 0.5)

    Returns:
        MultiLabelMetrics object containing:
            - micro_precision: Micro-averaged precision across all labels
            - micro_recall: Micro-averaged recall across all labels
            - macro_precision: Macro-averaged precision across all labels
            - macro_recall: Macro-averaged recall across all labels
            - micro_auc: Micro-averaged AUC across all labels
            - macro_auc: Macro-averaged AUC across all labels
            - per_concept: Per-concept precision, recall, AUC and support

    Example:
        >>> import numpy as np
        >>> y_true = np.array([[1, 0, 1], [0, 1, 0]])
        >>> y_score = np.array([[0.9, 0.1, 0.8], [0.2, 0.7, 0.3]])
        >>> concepts = ["fork", "pin", "skewer"]
        >>> metrics = calculate_multilabel_metrics(y_true, y_score, concepts)
        >>> metrics.micro_precision
        1.0
        >>> metrics.per_concept["fork"].precision
        1.0

    """
    micro_auc = float(roc_auc_score(y_true, y_score, average="micro"))
    macro_auc = float(roc_auc_score(y_true, y_score, average="macro"))
    per_concept_auc_scores: np.ndarray = roc_auc_score(y_true, y_score, average=None)  # type: ignore

    y_pred = (y_score >= threshold).astype(int)
    micro_precision = float(precision_score(y_true, y_pred, average="micro", zero_division=0.0))
    micro_recall = float(recall_score(y_true, y_pred, average="micro", zero_division=0.0))
    macro_precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0.0))
    macro_recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0.0))
    subset_accuracy = float(accuracy_score(y_true, y_pred))

    per_concept_metrics = {}
    for i, concept in enumerate(concept_list):
        y_true_i = y_true[:, i]
        y_pred_i = y_pred[:, i]
        support = int(y_true_i.sum())

        if support == 0:
            continue

        per_concept_metrics[concept] = ConceptMetrics(
            precision=float(precision_score(y_true_i, y_pred_i, zero_division=0.0)),
            recall=float(recall_score(y_true_i, y_pred_i, zero_division=0.0)),
            support=support,
            auc=float(per_concept_auc_scores[i]),
            subset_accuracy=float(accuracy_score(y_true_i, y_pred_i)),
        )

    # Calculate metrics for samples with no concepts
    no_concept_mask = y_true.sum(axis=1) == 0
    no_concept_count = int(no_concept_mask.sum())

    if no_concept_count > 0:
        y_true_no_concept = y_true[no_concept_mask]
        y_pred_no_concept = y_pred[no_concept_mask]
        y_score_no_concept = y_score[no_concept_mask]

        # Calculate micro-averaged metrics on this subset
        micro_precision_no_concept = float(
            precision_score(y_true_no_concept, y_pred_no_concept, average="micro", zero_division=0.0)
        )
        micro_recall_no_concept = float(
            recall_score(y_true_no_concept, y_pred_no_concept, average="micro", zero_division=0.0)
        )
        micro_auc_no_concept = float(roc_auc_score(y_true_no_concept, y_score_no_concept, average="micro"))

        per_concept_metrics["_no_concept"] = ConceptMetrics(
            precision=micro_precision_no_concept,
            recall=micro_recall_no_concept,
            support=no_concept_count,
            auc=micro_auc_no_concept,
            subset_accuracy=float(accuracy_score(y_true_no_concept, y_pred_no_concept)),
        )

    return MultiLabelMetrics(
        micro_precision=micro_precision,
        micro_recall=micro_recall,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        subset_accuracy=subset_accuracy,
        micro_auc=micro_auc,
        macro_auc=macro_auc,
        per_concept=per_concept_metrics,
    )


def format_metrics_display(metrics: MultiLabelMetrics, baseline_metrics: MultiLabelMetrics | None = None) -> str:
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

    lines.append("=" * 70)
    lines.append("SUMMARY (Micro/Macro)")
    lines.append("=" * 70)

    if baseline_metrics is not None:
        lines.append(f"Precision:              {metrics.micro_precision:.4} / {metrics.macro_precision:.4}")
        micro_precision_improvement = metrics.micro_precision / (baseline_metrics.micro_precision + 1e-10)
        macro_precision_improvement = metrics.macro_precision / (baseline_metrics.macro_precision + 1e-10)
        lines.append(
            f"Base Improvement:        {micro_precision_improvement:.1f}x / {macro_precision_improvement:.1f}x"
        )
        lines.append("")
        lines.append(f"Recall:                 {metrics.micro_recall:.4} / {metrics.macro_recall:.4}")
        micro_recall_improvement = metrics.micro_recall / (baseline_metrics.micro_recall + 1e-10)
        macro_recall_improvement = metrics.macro_recall / (baseline_metrics.macro_recall + 1e-10)
        lines.append(f"Base Improvement:        {micro_recall_improvement:.1f}x / {macro_recall_improvement:.1f}x")
        lines.append("")
        lines.append(f"AUC:                    {metrics.micro_auc:.4} / {metrics.macro_auc:.4}")
        micro_auc_improvement = metrics.micro_auc / (baseline_metrics.micro_auc + 1e-10)
        macro_auc_improvement = metrics.macro_auc / (baseline_metrics.macro_auc + 1e-10)
        lines.append(f"Base Improvement:        {micro_auc_improvement:.1f}x / {macro_auc_improvement:.1f}x")
        lines.append("")
        lines.append(f"Subset Accuracy:    {metrics.subset_accuracy:.4}")
        subset_accuracy_improvement = metrics.subset_accuracy / (baseline_metrics.subset_accuracy + 1e-10)
        lines.append(f"Base Improvement:        {subset_accuracy_improvement:.1f}x")
    else:
        lines.append(f"  Micro Precision:  {metrics.micro_precision:.4f}")
        lines.append(f"  Micro Recall:     {metrics.micro_recall:.4f}")
        lines.append(f"  Micro AUC:        {metrics.micro_auc:.4f}")
        lines.append(f"  Macro Precision:  {metrics.macro_precision:.4f}")
        lines.append(f"  Macro Recall:     {metrics.macro_recall:.4f}")
        lines.append(f"  Macro AUC:        {metrics.macro_auc:.4f}")
        lines.append(f"  Subset Accuracy:  {metrics.subset_accuracy:.4f}")

    lines.append("=" * 70)
    lines.append("PER-CONCEPT METRICS")
    lines.append("=" * 70)
    for concept, concept_metrics in metrics.per_concept.items():
        lines.append(
            f"  {concept}: p={concept_metrics.precision:.3f} "
            f"r={concept_metrics.recall:.2f} auc={concept_metrics.auc:.2f} "
            f"acc={concept_metrics.subset_accuracy:.2f} (n={concept_metrics.support})"
        )

    return "\n".join(lines)


def prepare_predictions(
    predictions_with_confidence: list[list[tuple[str, float]]],
    ground_truth_labels: list[list[str]],
    concept_list: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert predictions with confidence scores and labels to matrices.

    Args:
        predictions_with_confidence: List of predictions per sample, each with (concept, confidence)
        ground_truth_labels: List of ground truth concept lists per sample
        concept_list: Complete list of all possible concepts

    Returns:
        Tuple of (y_true, y_score) where:
            - y_true: Binary ground truth matrix (n_samples, n_concepts)
            - y_score: Probability score matrix (n_samples, n_concepts)
    """
    # Initialize MultiLabelBinarizer with all concepts
    mlb = MultiLabelBinarizer()
    mlb.fit([concept_list])

    # Transform ground truth labels to binary matrix
    y_true_transformed = mlb.transform(ground_truth_labels)

    # Convert sparse matrices to dense numpy arrays
    y_true = (
        np.asarray(y_true_transformed.toarray())
        if hasattr(y_true_transformed, "toarray")
        else np.asarray(y_true_transformed)
    )

    # Create probability score matrix
    y_score = np.zeros((len(predictions_with_confidence), len(concept_list)))
    for i, predictions in enumerate(predictions_with_confidence):
        for name, confidence in predictions:
            if name in mlb.classes_:
                j = list(mlb.classes_).index(name)
                y_score[i, j] = confidence

    return y_true, y_score


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
            --classifier-model-repo-id pilipolio/chess-positions-extractor \\
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
    labelled_positions = load_dataset_from_hf(dataset_repo_id, dataset_filename, dataset_revision)

    fens = [p.fen for p in labelled_positions]
    # Extract validated concept names from positions
    labels = [[c.name for c in p.concepts if c.validated_by] if p.concepts else [] for p in labelled_positions]
    all_predictions_with_confidence = extractor.extract_concepts_with_confidence(fens)

    if show_samples:
        n_samples = min(sample_size, len(labelled_positions))
        rng = np.random.RandomState(random_seed)
        sample_indices = list[int](rng.choice(len(labelled_positions), size=n_samples, replace=False))
        sample_fens = [fens[i] for i in sample_indices]
        sample_labels = [labels[i] for i in sample_indices]
        sample_predictions_with_confidence = [all_predictions_with_confidence[i] for i in sample_indices]

        print(f"\n{'=' * 70}")
        print(f"SAMPLE PREDICTIONS ({n_samples} examples)")
        print(f"{'=' * 70}\n")

        for pos, ground_truth, predicted_concepts in zip(
            sample_fens, sample_labels, sample_predictions_with_confidence, strict=True
        ):
            print(f"FEN: {pos}")
            print(f"Lichess: {get_analysis_url(pos)}")

            above_threshold_extracted_concept = [c for c, score in predicted_concepts if score >= threshold]

            match_marker = "✓" if set(ground_truth) == set(above_threshold_extracted_concept) else "✗"

            gt_str = ", ".join(ground_truth) if ground_truth else "(none)"
            pred_str = ", ".join(above_threshold_extracted_concept) if above_threshold_extracted_concept else "(none)"
            print(f"Ground Truth: {gt_str}")
            print(f"Prediction:   {pred_str} {match_marker}")
            print()

    print("\nCalculating comprehensive metrics on full dataset...")
    print(f"Extracting activations for {len(labelled_positions)} positions...")

    y_true, y_score = prepare_predictions(all_predictions_with_confidence, labels, extractor.probe.concept_list)
    metrics = calculate_multilabel_metrics(y_true, y_score, extractor.probe.concept_list, threshold)

    print("\n" + format_metrics_display(metrics))
    print()


if __name__ == "__main__":
    cli()
