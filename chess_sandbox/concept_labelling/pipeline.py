"""Main CLI pipeline for parsing PGNs and labeling concepts."""

import asyncio
import json
import os
from collections import Counter
from pathlib import Path

import click

from .labeller import label_positions
from .models import LabelledPosition
from .parser import parse_pgn_directory
from .refiner import Refiner


async def refine_positions_parallel(
    positions_to_refine: list[LabelledPosition],
    refiner: Refiner,
    semaphore_limit: int = 10,
) -> tuple[list[LabelledPosition], Counter[str], Counter[str]]:
    """Refine positions in parallel with semaphore-based rate limiting.

    Args:
        positions_to_refine: List of LabelledPosition objects with concepts to validate
        refiner: Refiner instance for validation
        semaphore_limit: Maximum concurrent API calls (default: 10)

    Returns:
        Tuple of (refined_positions, false_positive_counts, temporal_counts)
    """
    semaphore = asyncio.Semaphore(semaphore_limit)
    false_positive_counts = Counter[str]()
    temporal_counts = Counter[str]()

    async def refine_with_semaphore(position: LabelledPosition) -> None:
        async with semaphore:
            try:
                position.concepts = await refiner.refine(position)

                for concept in position.concepts:
                    if concept.validated_by is None:
                        false_positive_counts[concept.name] += 1
                    elif concept.temporal:
                        temporal_counts[concept.temporal] += 1

            except Exception as e:
                click.echo(f"\nWarning: Failed to refine position {position.game_id}: {e}", err=True)

    await asyncio.gather(*[refine_with_semaphore(p) for p in positions_to_refine])

    return positions_to_refine, false_positive_counts, temporal_counts


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
def main(input_dir: Path, output: Path, limit: int | None, refine_with_llm: bool, llm_model: str) -> None:
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

    if refine_with_llm:
        if not os.environ.get("OPENAI_API_KEY"):
            click.echo("Error: OPENAI_API_KEY environment variable not set", err=True)
            return

        click.echo(f"\nRefining labels with LLM ({llm_model})...")
        refiner = Refiner.create({"llm_model": llm_model})

        positions_to_refine = [p for p in labeled_positions if p.concepts]
        click.echo(f"Refining {len(positions_to_refine)} positions with detected concepts (parallel processing)...")

        _, false_positive_counts, temporal_counts = asyncio.run(refine_positions_parallel(positions_to_refine, refiner))

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
            json.dump(position.to_dict(), f)
            f.write("\n")
    click.echo(f"\nWrote labeled positions to: {output}")


if __name__ == "__main__":
    main()
