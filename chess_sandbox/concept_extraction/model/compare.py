"""
Compare concept probe training runs from HuggingFace Hub.

Loads metadata from multiple model revisions and displays a comparison
of training parameters and evaluation metrics.
"""

from dataclasses import dataclass

import click
from huggingface_hub import HfApi
from rich.console import Console
from rich.table import Table

from .model_artefact import ModelTrainingOutput


@dataclass
class ModelSummary:
    """Summary of a single trained model for comparison."""

    revision: str
    classifier_mode: str
    classifier_c: float
    classifier_class_weight: str
    training_samples: int
    test_samples: int
    random_seed: int

    # Probe metrics
    probe_micro_precision: float
    probe_micro_recall: float
    probe_macro_precision: float
    probe_macro_recall: float
    probe_micro_auc: float
    probe_macro_auc: float
    probe_subset_accuracy: float

    # Baseline metrics
    baseline_micro_precision: float
    baseline_micro_recall: float
    baseline_macro_precision: float
    baseline_macro_recall: float
    baseline_micro_auc: float
    baseline_macro_auc: float
    baseline_subset_accuracy: float

    # Training metadata
    training_date: str
    dataset_repo: str
    dataset_revision: str | None

    @property
    def micro_auc_improvement(self) -> float:
        """Calculate improvement over baseline for micro AUC."""
        if self.baseline_micro_auc == 0:
            return 0.0
        return (self.probe_micro_auc - self.baseline_micro_auc) / self.baseline_micro_auc

    @property
    def macro_auc_improvement(self) -> float:
        """Calculate improvement over baseline for macro AUC."""
        if self.baseline_macro_auc == 0:
            return 0.0
        return (self.probe_macro_auc - self.baseline_macro_auc) / self.baseline_macro_auc


def load_model_summary(repo_id: str, revision: str | None = None) -> ModelSummary:
    """
    Load model metadata from HuggingFace Hub.

    Args:
        repo_id: HuggingFace model repository ID
        revision: Git revision (tag, branch, commit hash)

    Returns:
        ModelSummary with extracted metrics and parameters

    Note:
        Metrics are test set evaluations (not training set).
        The training_stats dict contains metrics computed on the test split.
    """
    model_output = ModelTrainingOutput.from_hf(repo_id, revision=revision)

    stats = model_output.training_stats
    # Note: probe_metrics and baseline_metrics are test set evaluations
    probe_metrics = stats["probe"]
    baseline_metrics = stats["baseline"]

    return ModelSummary(
        revision=revision or "main",
        classifier_mode=stats.get("classifier_mode", stats.get("mode", "unknown")),
        classifier_c=stats.get("classifier_c", 1.0),
        classifier_class_weight=stats.get("classifier_class_weight", "none"),
        training_samples=stats.get("training_samples", 0),
        test_samples=stats.get("test_samples", 0),
        random_seed=stats.get("random_seed", 42),
        probe_micro_precision=probe_metrics["micro_precision"],
        probe_micro_recall=probe_metrics["micro_recall"],
        probe_macro_precision=probe_metrics["macro_precision"],
        probe_macro_recall=probe_metrics["macro_recall"],
        probe_micro_auc=probe_metrics["micro_auc"],
        probe_macro_auc=probe_metrics["macro_auc"],
        probe_subset_accuracy=probe_metrics["subset_accuracy"],
        baseline_micro_precision=baseline_metrics["micro_precision"],
        baseline_micro_recall=baseline_metrics["micro_recall"],
        baseline_macro_precision=baseline_metrics["macro_precision"],
        baseline_macro_recall=baseline_metrics["macro_recall"],
        baseline_micro_auc=baseline_metrics["micro_auc"],
        baseline_macro_auc=baseline_metrics["macro_auc"],
        baseline_subset_accuracy=baseline_metrics["subset_accuracy"],
        training_date=model_output.training_date,
        dataset_repo=(model_output.source_provenance or {}).get("source_dataset", {}).get("repo_id", "unknown"),
        dataset_revision=(model_output.source_provenance or {}).get("source_dataset", {}).get("revision"),
    )


def create_comparison_table(
    summaries: list[ModelSummary],
    show_baseline: bool = False,
    sort_by: str | None = None,
) -> Table:
    """
    Create rich Table for model comparison.

    Args:
        summaries: List of ModelSummary objects to compare
        show_baseline: Include baseline metrics in table
        sort_by: Metric name to sort by (e.g., "micro_auc", "macro_auc")

    Returns:
        Rich Table object ready for display
    """
    if sort_by:
        reverse = True
        if sort_by.startswith("-"):
            sort_by = sort_by[1:]
            reverse = False

        summaries = sorted(
            summaries,
            key=lambda s: getattr(s, f"probe_{sort_by}", 0),
            reverse=reverse,
        )

    table = Table(
        title="Model Comparison",
        show_header=True,
        header_style="bold magenta",
        expand=False,
        show_lines=False,
        min_width=120,
    )

    # Basic columns
    table.add_column("Revision", style="cyan", no_wrap=True, overflow="fold")
    table.add_column("C", justify="right", no_wrap=True)
    table.add_column("Weight", justify="center", no_wrap=True)
    table.add_column("Train", justify="right", no_wrap=True)
    table.add_column("Test", justify="right", no_wrap=True)

    # Probe metrics
    table.add_column("μ-Prec", justify="right", style="yellow", no_wrap=True)
    table.add_column("μ-Recall", justify="right", style="yellow", no_wrap=True)
    table.add_column("μ-AUC", justify="right", style="bold yellow", no_wrap=True)
    table.add_column("M-AUC", justify="right", style="bold yellow", no_wrap=True)
    table.add_column("Δμ", justify="right", style="green", no_wrap=True)
    table.add_column("ΔM", justify="right", style="green", no_wrap=True)

    if show_baseline:
        table.add_column("Base μ-AUC", justify="right", style="dim", no_wrap=True)
        table.add_column("Base M-AUC", justify="right", style="dim", no_wrap=True)

    for summary in summaries:
        row = [
            summary.revision[:12],
            f"{summary.classifier_c:.2f}",
            summary.classifier_class_weight,
            f"{summary.training_samples:,}",
            f"{summary.test_samples:,}",
            f"{summary.probe_micro_precision:.3f}",
            f"{summary.probe_micro_recall:.3f}",
            f"{summary.probe_micro_auc:.3f}",
            f"{summary.probe_macro_auc:.3f}",
            f"{summary.micro_auc_improvement:+.1%}",
            f"{summary.macro_auc_improvement:+.1%}",
        ]

        if show_baseline:
            row.extend(
                [
                    f"{summary.baseline_micro_auc:.3f}",
                    f"{summary.baseline_macro_auc:.3f}",
                ]
            )

        table.add_row(*row)

    return table


def fetch_last_n_commits(repo_id: str, n: int) -> list[str]:
    """
    Fetch the last N commit IDs from HuggingFace Hub.

    Args:
        repo_id: HuggingFace model repository ID
        n: Number of recent commits to fetch

    Returns:
        List of commit IDs (hashes) in reverse chronological order
    """
    api = HfApi()
    commits = api.list_repo_commits(repo_id, repo_type="model")
    return [commit.commit_id for commit in commits[:n]]


def check_comparability(summaries: list[ModelSummary], console: Console) -> None:
    """
    Check if models are comparable based on train/test sample counts.

    Displays a warning if sample counts differ across models.

    Args:
        summaries: List of ModelSummary objects
        console: Rich Console for output
    """
    if not summaries:
        return

    train_counts = {s.training_samples for s in summaries}
    test_counts = {s.test_samples for s in summaries}

    if len(train_counts) > 1 or len(test_counts) > 1:
        console.print("\n[yellow]⚠️  Warning: Models have different dataset sizes![/yellow]")
        console.print("[yellow]Metrics may not be directly comparable.[/yellow]\n")

        warning_table = Table(
            title="Dataset Size Differences",
            show_header=True,
            header_style="bold yellow",
        )
        warning_table.add_column("Revision", style="cyan")
        warning_table.add_column("Train Samples", justify="right")
        warning_table.add_column("Test Samples", justify="right")

        for summary in summaries:
            warning_table.add_row(
                summary.revision[:12],
                f"{summary.training_samples:,}",
                f"{summary.test_samples:,}",
            )

        console.print(warning_table)
        console.print()


@click.command()
@click.option(
    "--repo-id",
    required=True,
    help="HuggingFace model repository ID (e.g., 'pilipolio/chess-sandbox-concept-probes')",
)
@click.option(
    "--revisions",
    default=None,
    help="Comma-separated list of revisions to compare (e.g., 'v1,v2,main')",
)
@click.option(
    "--last",
    default=None,
    type=int,
    help="Compare the last N commits from the repository",
)
@click.option(
    "--sort-by",
    default=None,
    help="Metric to sort by: micro_auc, macro_auc, micro_precision, etc. Prefix with - for ascending.",
)
@click.option(
    "--show-baseline",
    is_flag=True,
    help="Include baseline metrics in comparison table",
)
def compare(
    repo_id: str,
    revisions: str | None,
    last: int | None,
    sort_by: str | None,
    show_baseline: bool,
) -> None:
    """
    Compare concept probe models across different HuggingFace Hub revisions.

    Example (specify revisions):
        uv run python -m chess_sandbox.concept_extraction.model.compare \\
            --repo-id pilipolio/chess-sandbox-concept-probes \\
            --revisions v1.0,v1.1,v2.0 \\
            --sort-by micro_auc \\
            --show-baseline

    Example (last N commits):
        uv run python -m chess_sandbox.concept_extraction.model.compare \\
            --repo-id pilipolio/chess-sandbox-concept-probes \\
            --last 6 \\
            --sort-by micro_auc
    """
    console = Console(width=160, force_terminal=True)

    # Validate mutually exclusive options
    if revisions and last:
        console.print("[red]Error: Cannot use both --revisions and --last options[/red]")
        return
    if not revisions and not last:
        console.print("[red]Error: Must specify either --revisions or --last option[/red]")
        return

    # Get revision list
    if last:
        console.print(f"\n[bold]Fetching last {last} commits from {repo_id}...[/bold]")
        try:
            revision_list = fetch_last_n_commits(repo_id, last)
            console.print(f"Found {len(revision_list)} commits\n")
        except Exception as e:
            console.print(f"[red]Failed to fetch commits: {e}[/red]")
            return
    else:
        revision_list = [rev.strip() for rev in revisions.split(",")]  # type: ignore
        console.print(f"\n[bold]Loading models from {repo_id}...[/bold]")
        console.print(f"Revisions: {', '.join(revision_list)}\n")

    summaries: list[ModelSummary] = []
    for revision in revision_list:
        try:
            console.print(f"Loading {revision}...", style="dim")
            summary = load_model_summary(repo_id, revision)
            summaries.append(summary)
        except Exception as e:
            console.print(f"[red]Failed to load {revision}: {e}[/red]")

    if not summaries:
        console.print("[red]No models loaded successfully![/red]")
        return

    console.print(f"\n[green]Loaded {len(summaries)} models[/green]\n")

    # Check if models are comparable
    check_comparability(summaries, console)

    table = create_comparison_table(summaries, show_baseline=show_baseline, sort_by=sort_by)
    console.print(table)

    # Summary statistics
    if len(summaries) > 1:
        best_micro = max(summaries, key=lambda s: s.probe_micro_auc)
        best_macro = max(summaries, key=lambda s: s.probe_macro_auc)

        console.print("\n[bold]Best Models:[/bold]")
        console.print(f"  Micro AUC: {best_micro.revision[:12]} ({best_micro.probe_micro_auc:.3f})")
        console.print(f"  Macro AUC: {best_macro.revision[:12]} ({best_macro.probe_macro_auc:.3f})")


if __name__ == "__main__":
    compare()
