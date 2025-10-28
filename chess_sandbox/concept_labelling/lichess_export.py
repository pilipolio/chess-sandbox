"""Export labeled positions to Lichess study format."""

import json
import random
from collections import defaultdict
from pathlib import Path

import click
import requests

from chess_sandbox.config import settings

from .models import LabelledPosition


def load_labeled_positions(jsonl_path: Path) -> list[LabelledPosition]:
    """Load labeled positions from JSONL file.

    >>> import tempfile
    >>> temp_file = Path(tempfile.mktemp(suffix='.jsonl'))
    >>> data1 = {
    ...     "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    ...     "move_number": 1, "side_to_move": "white", "comment": "Pin",
    ...     "game_id": "g1", "concepts": ["pin"]
    ... }
    >>> data2 = {
    ...     "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    ...     "move_number": 2, "side_to_move": "black", "comment": "Fork",
    ...     "game_id": "g2", "concepts": ["fork"]
    ... }
    >>> with temp_file.open('w') as f:
    ...     _ = f.write(json.dumps(data1) + '\\n' + json.dumps(data2) + '\\n')
    >>> positions = load_labeled_positions(temp_file)
    >>> len(positions)
    2
    >>> positions[0].concepts
    ['pin']
    >>> temp_file.unlink()
    """
    positions: list[LabelledPosition] = []
    with jsonl_path.open() as f:
        for line in f:
            data = json.loads(line)
            position = LabelledPosition.from_dict(data)
            positions.append(position)
    return positions


def sample_positions(
    positions: list[LabelledPosition], n_samples: int, strategy: str = "balanced"
) -> list[LabelledPosition]:
    """Sample positions from labeled dataset.

    >>> from chess_sandbox.concept_labelling.models import LabelledPosition
    >>> positions = [
    ...     LabelledPosition("fen1", 1, "white", "pin1", "g1", ["pin"]),
    ...     LabelledPosition("fen2", 2, "white", "pin2", "g2", ["pin"]),
    ...     LabelledPosition("fen3", 3, "white", "fork1", "g3", ["fork"]),
    ...     LabelledPosition("fen4", 4, "white", "fork2", "g4", ["fork"]),
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
            for concept in pos.concepts:
                by_concept[concept].append(pos)

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
        labeled_positions = [p for p in positions if p.concepts]
        n = min(n_samples, len(labeled_positions))
        return random.sample(labeled_positions, n)


def position_to_pgn(position: LabelledPosition) -> str:
    """Convert a labeled position to PGN format with tags and comment.

    >>> from chess_sandbox.concept_labelling.models import LabelledPosition
    >>> pos = LabelledPosition(
    ...     fen="r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    ...     move_number=3,
    ...     side_to_move="white",
    ...     comment="Pin that knight",
    ...     game_id="gameknot_1160",
    ...     concepts=["pin"]
    ... )
    >>> pgn = position_to_pgn(pos)
    >>> '[Event "Concept: pin"]' in pgn
    True
    >>> '[Site "gameknot_1160"]' in pgn
    True
    >>> '[FEN "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"]' in pgn
    True
    >>> '{ Pin that knight }' in pgn
    True
    """
    concepts_str = ", ".join(position.concepts) if position.concepts else "unlabeled"
    pgn_lines = [
        f'[Event "Concept: {concepts_str}"]',
        f'[Site "{position.game_id}"]',
        f'[FEN "{position.fen}"]',
        "",
        f"{{ {position.comment} }}",
        "",
    ]
    return "\n".join(pgn_lines)


def import_pgn_to_lichess(study_id: str, pgn_content: str) -> dict[str, str]:
    """Import PGN content to a Lichess study.

    Args:
        study_id: The Lichess study ID to import into
        pgn_content: The PGN content to import

    Returns:
        Response from Lichess API

    Raises:
        requests.HTTPError: If the API request fails
    """
    if not settings.LICHESS_API_TOKEN:
        msg = "LICHESS_API_TOKEN not set in environment"
        raise ValueError(msg)

    url = f"https://lichess.org/api/study/{study_id}/import-pgn"
    headers = {"Authorization": f"Bearer {settings.LICHESS_API_TOKEN}"}
    data = {"pgn": pgn_content}

    response = requests.post(url, headers=headers, data=data, timeout=30)
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
def main(
    input_path: Path,
    study_id: str,
    n_samples: int,
    strategy: str,
    seed: int | None,
) -> None:
    """Export labeled positions directly to Lichess study via API.

    Requires LICHESS_API_TOKEN to be set in environment.
    """
    if seed is not None:
        random.seed(seed)

    click.echo(f"Loading positions from: {input_path}")
    positions = load_labeled_positions(input_path)
    click.echo(f"Loaded {len(positions)} total positions")

    labeled_positions = [p for p in positions if p.concepts]
    click.echo(f"Found {len(labeled_positions)} positions with concept labels")

    click.echo(f"Sampling {n_samples} positions using {strategy} strategy")
    sampled = sample_positions(labeled_positions, n_samples, strategy)
    click.echo(f"Sampled {len(sampled)} positions")

    # Convert to PGN
    pgn_content = ""
    for position in sampled:
        pgn = position_to_pgn(position)
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
    except requests.HTTPError as e:
        click.echo(f"Lichess API error: {e}", err=True)
        click.echo(f"Response: {e.response.text if e.response else 'N/A'}", err=True)
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
