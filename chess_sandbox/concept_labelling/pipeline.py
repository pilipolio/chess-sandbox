"""Main CLI pipeline for parsing PGNs and labeling concepts."""

import asyncio
import json
import time
from collections import Counter
from pathlib import Path

import click
from tqdm.asyncio import tqdm

from .labeller import label_positions
from .models import LabelledPosition
from .parser import parse_pgn_directory
from .refiner import Refiner


async def refine_positions_parallel(
    positions_to_refine: list[LabelledPosition],
    refiner: Refiner,
    semaphore_limit: int = 10,
) -> None:
    """Refine positions in parallel with semaphore-based rate limiting.

    Args:
        positions_to_refine: List of LabelledPosition objects with concepts to validate
        refiner: Refiner instance for validation
        semaphore_limit: Maximum concurrent API calls (default: 10)

    Returns:
        Tuple of (refined_positions, false_positive_counts, temporal_counts)
    """
    semaphore = asyncio.Semaphore(semaphore_limit)

    async def refine_with_semaphore(position: LabelledPosition) -> None:
        async with semaphore:
            try:
                position.concepts = await refiner.refine(position)
            except Exception as e:
                tqdm.write(f"Warning: Failed to refine position {position.game_id}: {e}")

    await tqdm.gather(  # type: ignore[reportUnknownMemberType]
        *[refine_with_semaphore(p) for p in positions_to_refine],
        desc="Refining positions",
        unit="pos",
        mininterval=60,  # Update progress every minute
    )


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
    default="gpt-4.1-mini",
    help="LLM model to use for refinement."
    "gpt-4.1-mini: Offering a good balance of speed and accuracy"
    "(gpt5-nano: Too slow, while 4.1-nano marks too many concepts as false positives)",
)
def main(input_dir: Path, output: Path, limit: int | None, refine_with_llm: bool, llm_model: str) -> None:
    """Parse PGN files and label chess positions with detected concepts.

    This combines PGN parsing and concept labeling into a single pipeline.
    Optionally refines labels using LLM for higher precision.
    """
    start_time = time.time()

    click.echo(f"Parsing PGN files from: {input_dir}")
    if limit:
        click.echo(f"Processing only first {limit} files")

    positions = parse_pgn_directory(input_dir, limit=limit)
    elapsed_time = time.time() - start_time
    click.echo(f"Extracted {len(positions)} positions with comments in: {elapsed_time:.2f} seconds")

    labelled_positions = label_positions(positions)
    labeled_count = sum(1 for p in labelled_positions if p.concepts)
    click.echo(f"Labeled {labeled_count}/{len(labelled_positions)} positions with concepts")

    concept_counts = Counter[str](concept.name for position in labelled_positions for concept in position.concepts)
    click.echo("\nTop concepts (Phase 1 - Regex):")
    for concept, count in concept_counts.most_common(10):
        click.echo(f"  {concept}: {count}")

    if refine_with_llm:
        start_time = time.time()

        click.echo(f"\nRefining labels with LLM ({llm_model})...")
        refiner = Refiner.create({"llm_model": llm_model})

        positions_to_refine = [p for p in labelled_positions if p.concepts]
        if len(positions_to_refine) == 0:
            click.echo("No positions to refine")
            return

        click.echo(f"Refining {len(positions_to_refine)} positions with detected concepts (parallel processing)...")

        asyncio.run(refine_positions_parallel(positions_to_refine, refiner))
        elapsed_time = time.time() - start_time
        total_concepts = sum(concept_counts.values())
        elapsed_minutes = elapsed_time / 60
        time_per_100 = (elapsed_minutes / total_concepts) * 100

        click.echo("\nRefinement results:")
        click.echo(
            f"  Analyzed {total_concepts} concepts in {elapsed_minutes:.2f} minutes ({time_per_100:.2f} mins / 100)"
        )
        validated_concepts_count = sum(1 for p in labelled_positions for _ in p.validated_concepts)

        click.echo(" ")
        click.echo(f"  Validated: {validated_concepts_count}/{total_concepts} concepts")

        false_positive_counts = Counter[str](
            concept.name
            for position in labelled_positions
            for concept in position.concepts
            if concept.validated_by is None
        )
        temporal_counts = Counter[str](
            concept.temporal for position in labelled_positions for concept in position.concepts if concept.temporal
        )

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
        for position in labelled_positions:
            json.dump(position.to_dict(), f)
            f.write("\n")
    click.echo(f"\nWrote labeled positions to: {output}")


if __name__ == "__main__":
    main()
