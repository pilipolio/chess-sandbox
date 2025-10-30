"""Export labeled positions to Lichess study format."""

import json
import random
from collections import defaultdict
from pathlib import Path

import click
import httpx

from chess_sandbox.config import settings

from .models import LabelledPosition


def load_labeled_positions(jsonl_path: Path) -> list[LabelledPosition]:
    """Load labeled positions from JSONL file.

    Handles both old format (concepts_validated, temporal_context) and new format
    (concepts with Concept objects).

    >>> import tempfile
    >>> temp_file = Path(tempfile.mktemp(suffix='.jsonl'))
    >>> fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    >>> data1 = {
    ...     "fen": fen, "move_number": 1, "side_to_move": "white", "comment": "Pin",
    ...     "game_id": "g1", "move_san": "e4", "previous_fen": fen,
    ...     "concepts": [{"name": "pin", "validated_by": None, "temporal": None}]
    ... }
    >>> data2 = {
    ...     "fen": fen, "move_number": 2, "side_to_move": "black", "comment": "Fork",
    ...     "game_id": "g2", "move_san": "Nf3", "previous_fen": fen,
    ...     "concepts": [{"name": "fork", "validated_by": None, "temporal": None}]
    ... }
    >>> with temp_file.open('w') as f:
    ...     _ = f.write(json.dumps(data1) + '\\n' + json.dumps(data2) + '\\n')
    >>> positions = load_labeled_positions(temp_file)
    >>> len(positions)
    2
    >>> positions[0].concepts[0].name
    'pin'
    >>> temp_file.unlink()
    """
    positions: list[LabelledPosition] = []
    with jsonl_path.open() as f:
        for line in f:
            data = json.loads(line)

            # Convert old format to new format if needed
            if "concepts" in data and data["concepts"] and isinstance(data["concepts"][0], str):
                # Old format: concepts is list of strings
                concept_names = data["concepts"]
                concepts_validated = set(data.get("concepts_validated", []))
                temporal_context = data.get("temporal_context", {})

                # Convert to new format
                data["concepts"] = [
                    {
                        "name": name,
                        "validated_by": "llm" if name in concepts_validated else None,
                        "temporal": temporal_context.get(name),
                    }
                    for name in concept_names
                ]

            position = LabelledPosition.from_dict(data)
            positions.append(position)
    return positions


def sample_positions(
    positions: list[LabelledPosition],
    n_samples: int,
    strategy: str = "balanced",
    use_validated: bool = False,
) -> list[LabelledPosition]:
    """Sample positions from labeled dataset.

    Args:
        positions: List of labeled positions
        n_samples: Number of samples to select
        strategy: "balanced" (equal per concept) or "random"
        use_validated: If True, use only validated concepts

    >>> from chess_sandbox.concept_labelling.models import Concept, LabelledPosition
    >>> positions = [
    ...     LabelledPosition("fen1", 1, "white", "pin1", "g1", "e4", "fen0", [Concept(name="pin")]),
    ...     LabelledPosition("fen2", 2, "white", "pin2", "g2", "e5", "fen1", [Concept(name="pin")]),
    ...     LabelledPosition("fen3", 3, "white", "fork1", "g3", "Nf3", "fen2", [Concept(name="fork")]),
    ...     LabelledPosition("fen4", 4, "white", "fork2", "g4", "Nc6", "fen3", [Concept(name="fork")]),
    ... ]
    >>> random.seed(42)
    >>> sampled = sample_positions(positions, 2, strategy="balanced")
    >>> len(sampled)
    2
    >>> sampled = sample_positions(positions, 2, strategy="random")
    >>> len(sampled)
    2
    """
    if strategy == "balanced":
        # Group by concepts
        by_concept: dict[str, list[LabelledPosition]] = defaultdict(list)
        for pos in positions:
            concepts = pos.validated_concepts if use_validated else pos.concepts
            for concept in concepts:
                by_concept[concept.name].append(pos)

        # Sample evenly from each concept
        sampled: list[LabelledPosition] = []
        concepts = list(by_concept.keys())
        per_concept = max(1, n_samples // len(concepts)) if concepts else 0

        for concept in concepts:
            concept_positions = by_concept[concept]
            n = min(per_concept, len(concept_positions))
            sampled.extend(random.sample(concept_positions, n))

        # If we need more samples, add random ones
        if len(sampled) < n_samples:
            remaining = [p for p in positions if p not in sampled]
            additional = min(n_samples - len(sampled), len(remaining))
            sampled.extend(random.sample(remaining, additional))

        return sampled[:n_samples]
    else:  # random strategy
        if use_validated:
            labeled_positions = [p for p in positions if p.validated_concepts]
        else:
            labeled_positions = [p for p in positions if p.concepts]
        n = min(n_samples, len(labeled_positions))
        return random.sample(labeled_positions, n)


def position_to_pgn(position: LabelledPosition, use_validated: bool = False) -> str:
    """Convert a labeled position to PGN format with tags and comment.

    Args:
        position: The labeled position to convert
        use_validated: If True, use only validated concepts and include temporal context

    >>> from chess_sandbox.concept_labelling.models import Concept, LabelledPosition
    >>> pos = LabelledPosition(
    ...     fen="r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    ...     move_number=3,
    ...     side_to_move="white",
    ...     comment="Pin that knight",
    ...     game_id="gameknot_1160",
    ...     move_san="Nc6",
    ...     previous_fen="r1bqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
    ...     concepts=[Concept(name="pin", validated_by="llm", temporal="actual")]
    ... )
    >>> pgn = position_to_pgn(pos)
    >>> '2... Nc6' in pgn
    True
    >>> 'Pin that knight' in pgn
    True
    """
    # Always show all concepts for diversity, regardless of validation status
    concepts = position.concepts
    concepts_str = ", ".join(c.name for c in concepts) if concepts else "unlabeled"

    # Build PGN header - use previous_fen as starting position
    pgn_lines = [
        f'[Event "{concepts_str}"]',
        f'[Site "{position.game_id}"]',
        f'[FEN "{position.previous_fen}"]',
        "",
    ]

    # Add the move with comment
    # side_to_move indicates who moves NEXT, not who just moved
    # If white is to move, black just moved on previous move number
    # If black is to move, white just moved on current move number
    if position.side_to_move == "white":
        move_prefix = f"{position.move_number - 1}..."
    else:
        move_prefix = f"{position.move_number}."
    move_line = f"{move_prefix} {position.move_san}"

    # Add original comment
    if position.comment:
        move_line += f" {{ {position.comment} }}"

    pgn_lines.append(move_line)

    # Add concept details with validation and temporal info
    for concept in concepts:
        validated = "yes" if concept.validated_by else "no"
        temporal = concept.temporal if concept.temporal else "none"
        pgn_lines.append(f"{{ Concept: {concept.name} [validated: {validated}, temporal: {temporal}] }}")

    pgn_lines.append("")
    return "\n".join(pgn_lines)


def import_pgn_to_lichess(study_id: str, pgn_content: str) -> dict[str, str]:
    """Import PGN content to a Lichess study.

    Args:
        study_id: The Lichess study ID to import into
        pgn_content: The PGN content to import

    Returns:
        Response from Lichess API

    Raises:
        httpx.HTTPStatusError: If the API request fails
    """
    if not settings.LICHESS_API_TOKEN:
        msg = "LICHESS_API_TOKEN not set in environment"
        raise ValueError(msg)

    url = f"https://lichess.org/api/study/{study_id}/import-pgn"
    headers = {"Authorization": f"Bearer {settings.LICHESS_API_TOKEN}"}
    data = {"pgn": pgn_content}

    response = httpx.post(url, headers=headers, data=data, timeout=30)
    response.raise_for_status()

    return response.json()


@click.command()
@click.option(
    "--input",
    "input_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Input JSONL file with labeled positions",
)
@click.option(
    "--study-id",
    type=str,
    required=True,
    help="Lichess study ID to import PGN into (requires LICHESS_API_TOKEN)",
)
@click.option(
    "--n-samples",
    type=int,
    default=64,
    help="Number of positions to sample (default: 64)",
)
@click.option(
    "--strategy",
    type=click.Choice(["balanced", "random"]),
    default="balanced",
    help="Sampling strategy: balanced (equal per concept) or random",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Random seed for reproducible sampling",
)
@click.option(
    "--use-validated",
    is_flag=True,
    default=False,
    help="Use LLM-validated concepts (concepts_validated) instead of regex concepts",
)
@click.option(
    "--filter-temporal",
    type=click.Choice(["actual", "threat", "hypothetical", "past"]),
    default=None,
    help="Filter to positions with specific temporal context (requires --use-validated)",
)
def main(
    input_path: Path,
    study_id: str,
    n_samples: int,
    strategy: str,
    seed: int | None,
    use_validated: bool,
    filter_temporal: str | None,
) -> None:
    """Export labeled positions directly to Lichess study via API.

    Requires LICHESS_API_TOKEN to be set in environment.

    Examples:
        # Export regex-labeled positions
        python -m chess_sandbox.concept_labelling.lichess_export \\
            --input positions.jsonl --study-id ABC123 --n-samples 64

        # Export only LLM-validated positions with temporal context
        python -m chess_sandbox.concept_labelling.lichess_export \\
            --input positions_refined.jsonl --study-id ABC123 \\
            --use-validated --n-samples 32

        # Export only positions where concepts ACTUALLY exist (not threats)
        python -m chess_sandbox.concept_labelling.lichess_export \\
            --input positions_refined.jsonl --study-id ABC123 \\
            --use-validated --filter-temporal actual --n-samples 16
    """
    if seed is not None:
        random.seed(seed)

    if filter_temporal and not use_validated:
        click.echo("Error: --filter-temporal requires --use-validated", err=True)
        raise SystemExit(1)

    click.echo(f"Loading positions from: {input_path}")
    positions = load_labeled_positions(input_path)
    click.echo(f"Loaded {len(positions)} total positions")

    # Filter positions based on flags
    if use_validated:
        labeled_positions = [p for p in positions if p.validated_concepts]
        click.echo(f"Found {len(labeled_positions)} positions with validated concept labels")

        if filter_temporal:
            # Filter to positions where at least one concept has the specified temporal context
            filtered = [
                p for p in labeled_positions if any(c.temporal == filter_temporal for c in p.validated_concepts)
            ]
            click.echo(f"Filtered to {len(filtered)} positions with temporal context: {filter_temporal}")
            labeled_positions = filtered
    else:
        labeled_positions = [p for p in positions if p.concepts]
        click.echo(f"Found {len(labeled_positions)} positions with concept labels")

    if not labeled_positions:
        click.echo("No positions match the criteria", err=True)
        raise SystemExit(1)

    click.echo(f"Sampling {n_samples} positions using {strategy} strategy")
    sampled = sample_positions(labeled_positions, n_samples, strategy, use_validated)
    click.echo(f"Sampled {len(sampled)} positions")

    # Convert to PGN
    pgn_content = ""
    for position in sampled:
        pgn = position_to_pgn(position, use_validated)
        pgn_content += pgn + "\n"

    # Upload to Lichess
    click.echo(f"\nUploading to Lichess study: {study_id}")
    try:
        result = import_pgn_to_lichess(study_id, pgn_content)
        click.echo(f"Successfully imported to Lichess: {result}")
        click.echo(f"View study at: https://lichess.org/study/{study_id}")
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        click.echo("Please set LICHESS_API_TOKEN in your .env file", err=True)
        raise SystemExit(1) from e
    except httpx.HTTPStatusError as e:
        click.echo(f"Lichess API error: {e}", err=True)
        click.echo(f"Response: {e.response.text if e.response else 'N/A'}", err=True)
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
