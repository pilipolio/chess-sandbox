"""Main CLI pipeline for parsing PGNs and labeling concepts."""

import json
import time
from collections import Counter
from pathlib import Path

import click

from .batch_refiner import BatchRefiner
from .labeller import label_positions
from .parser import parse_pgn_directory
from .refiner import Refiner


@click.command()
@click.option(
    "--input-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Directory containing PGN files to process",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    required=True,
    help="Output JSONL file for labeled positions",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Limit number of PGN files to process (for testing)",
)
@click.option(
    "--refine-with-llm",
    is_flag=True,
    default=False,
    help="Use LLM to validate concepts and extract temporal context",
)
@click.option(
    "--llm-model",
    type=str,
    default="gpt-5-nano",
    help="LLM model to use for refinement",
)
@click.option(
    "--use-batch",
    is_flag=True,
    default=False,
    help="Use batch API for cost-effective LLM refinement (50-92% cost savings)",
)
@click.option(
    "--resume-batch",
    type=str,
    default=None,
    help="Resume processing a previously submitted batch job by batch_id",
)
def main(
    input_dir: Path,
    output: Path,
    limit: int | None,
    refine_with_llm: bool,
    llm_model: str,
    use_batch: bool,
    resume_batch: str | None,
) -> None:
    """Parse PGN files and label chess positions with detected concepts.

    This combines PGN parsing and concept labeling into a single pipeline.
    Optionally refines labels using LLM for higher precision.
    """
    click.echo(f"Parsing PGN files from: {input_dir}")
    if limit:
        click.echo(f"Processing only first {limit} files")

    positions = parse_pgn_directory(input_dir, limit=limit)
    click.echo(f"Extracted {len(positions)} positions with comments")

    labeled_positions = label_positions(positions)
    labeled_count = sum(1 for p in labeled_positions if p.concepts)
    click.echo(f"Labeled {labeled_count}/{len(labeled_positions)} positions with concepts")

    concept_counts = Counter[str](concept.name for position in labeled_positions for concept in position.concepts)
    click.echo("\nTop concepts (Phase 1 - Regex):")
    for concept, count in concept_counts.most_common(10):
        click.echo(f"  {concept}: {count}")

    # Handle batch resumption
    if resume_batch:
        click.echo(f"\nResuming batch job: {resume_batch}")
        batch_refiner = BatchRefiner.create(model=llm_model)

        # Load original positions from batch metadata
        metadata = batch_refiner.get_batch_metadata(resume_batch)
        click.echo(f"Batch submitted at: {metadata['submitted_at']}")
        click.echo(f"Current status: {metadata['status']}")

        # Poll for completion
        while True:
            status_info = batch_refiner.poll_status(resume_batch)
            status = status_info["status"]

            if status == "completed":
                click.echo("\nBatch completed! Downloading results...")
                results_file = batch_refiner.download_results(resume_batch)
                labeled_positions = batch_refiner.process_results(results_file, labeled_positions)
                break
            elif status in ["failed", "expired", "cancelled"]:
                raise RuntimeError(f"Batch {status}")

            # Show progress
            counts = status_info["request_counts"]
            pct = (counts["completed"] / counts["total"]) * 100 if counts["total"] > 0 else 0
            click.echo(f"Progress: {pct:.1f}% ({counts['completed']}/{counts['total']})")
            time.sleep(60)

    # Handle refinement
    elif refine_with_llm and use_batch:
        click.echo(f"\nRefining labels with Batch API ({llm_model})...")
        batch_refiner = BatchRefiner.create(model=llm_model)

        positions_to_refine = [p for p in labeled_positions if p.concepts]
        click.echo(f"Preparing {len(positions_to_refine)} positions for batch validation...")

        # Prepare and submit batch
        batch_file = batch_refiner.prepare_batch_input(positions_to_refine)
        click.echo("Submitting batch job...")
        batch_id = batch_refiner.submit_batch(batch_file)

        click.echo(f"\nBatch submitted: {batch_id}")
        click.echo("Processing (this may take up to 24 hours, often completes faster)...")
        click.echo(f"To resume later: --resume-batch {batch_id}")

        # Poll for completion
        while True:
            status_info = batch_refiner.poll_status(batch_id)
            status = status_info["status"]

            if status == "completed":
                click.echo("\nBatch completed! Downloading results...")
                results_file = batch_refiner.download_results(batch_id)
                labeled_positions = batch_refiner.process_results(results_file, positions_to_refine)
                break
            elif status in ["failed", "expired", "cancelled"]:
                raise RuntimeError(f"Batch {status}")

            # Show progress
            counts = status_info["request_counts"]
            pct = (counts["completed"] / counts["total"]) * 100 if counts["total"] > 0 else 0
            click.echo(f"Progress: {pct:.1f}% ({counts['completed']}/{counts['total']})")
            time.sleep(60)

    elif refine_with_llm:
        click.echo(f"\nRefining labels with LLM ({llm_model})...")
        refiner = Refiner.create({"llm_model": llm_model})

        positions_to_refine = [p for p in labeled_positions if p.concepts]
        click.echo(f"Refining {len(positions_to_refine)} positions with detected concepts...")

        false_positive_counts = Counter[str]()
        temporal_counts = Counter[str]()

        with click.progressbar(positions_to_refine, label="Refining") as bar:
            for position in bar:
                try:
                    # Functional approach: get new concepts list
                    position.concepts = refiner.refine(position)

                    # Collect statistics from refined concepts
                    for concept in position.concepts:
                        if concept.validated_by is None:
                            false_positive_counts[concept.name] += 1
                        elif concept.temporal:
                            temporal_counts[concept.temporal] += 1

                except Exception as e:
                    click.echo(f"\nWarning: Failed to refine position {position.game_id}: {e}", err=True)

        click.echo("\nRefinement results:")
        validated_count = sum(1 for p in labeled_positions if p.validated_concepts)
        click.echo(f"  Validated: {validated_count} positions")

        if false_positive_counts:
            click.echo("\n  False positives detected:")
            for concept, count in false_positive_counts.most_common(5):
                click.echo(f"    {concept}: {count}")

        if temporal_counts:
            click.echo("\n  Temporal distribution:")
            for context, count in temporal_counts.most_common():
                click.echo(f"    {context}: {count}")

    # Print refinement statistics for batch processing
    if refine_with_llm and use_batch or resume_batch:
        false_positive_counts = Counter[str]()
        temporal_counts = Counter[str]()

        for position in labeled_positions:
            for concept in position.concepts:
                if concept.validated_by is None:
                    false_positive_counts[concept.name] += 1
                elif concept.temporal:
                    temporal_counts[concept.temporal] += 1

        click.echo("\nRefinement results:")
        validated_count = sum(1 for p in labeled_positions if p.validated_concepts)
        click.echo(f"  Validated: {validated_count} positions")

        if false_positive_counts:
            click.echo("\n  False positives detected:")
            for concept, count in false_positive_counts.most_common(5):
                click.echo(f"    {concept}: {count}")

        if temporal_counts:
            click.echo("\n  Temporal distribution:")
            for context, count in temporal_counts.most_common():
                click.echo(f"    {context}: {count}")

    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w") as f:
        for position in labeled_positions:
            if position.concepts:
                json.dump(position.to_dict(), f)
                f.write("\n")
    click.echo(f"\nWrote labeled positions to: {output}")


if __name__ == "__main__":
    main()
