"""Main CLI pipeline for parsing PGNs and labeling concepts."""

import json
from collections import Counter
from pathlib import Path

import click

from .labeller import label_positions
from .parser import parse_pgn_directory


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
def main(input_dir: Path, output: Path, limit: int | None) -> None:
    """Parse PGN files and label chess positions with detected concepts.

    This combines PGN parsing and concept labeling into a single pipeline.
    """
    click.echo(f"Parsing PGN files from: {input_dir}")
    if limit:
        click.echo(f"Processing only first {limit} files")

    positions = parse_pgn_directory(input_dir, limit=limit)
    click.echo(f"Extracted {len(positions)} positions with comments")

    labeled_positions = label_positions(positions)
    labeled_count = sum(1 for p in labeled_positions if p.concepts)
    click.echo(f"Labeled {labeled_count}/{len(labeled_positions)} positions with concepts")

    concept_counts = Counter[str](concept for position in labeled_positions for concept in position.concepts)
    click.echo("\nTop concepts:")
    for concept, count in concept_counts.most_common(10):
        click.echo(f"  {concept}: {count}")

    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w") as f:
        for position in labeled_positions:
            json.dump(position.to_dict(), f)
            f.write("\n")
    click.echo(f"Wrote labeled positions to: {output}")


if __name__ == "__main__":
    main()
