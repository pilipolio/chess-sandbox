"""Prepare chess puzzle solution dataset from Lichess puzzles.

This script:
1. Loads the Lichess/chess-puzzles dataset
2. Filters by popularity (minimum number of plays, default: 80)
3. Filters by rating (optional max difficulty)
4. Optionally filters by theme(s) (e.g., mateIn1, mateIn2)
5. Generates board images (120x120 PNG)
6. Extracts the first move of the puzzle solution (in UCI notation)
7. Creates train/test splits
8. Pushes to HuggingFace Hub (pilipolio/lichess-puzzles-solutions)
"""

import io
from collections.abc import Iterator

import cairosvg
import chess
import chess.svg
import click
import numpy as np
from datasets import Dataset, DatasetDict, Features, Sequence, Value, load_dataset
from datasets import Image as HFImage
from PIL import Image
from tqdm import tqdm


def generate_board_image(fen: str, size: int = 120) -> Image.Image:
    """Generate PNG board image from FEN position."""
    board = chess.Board(fen)
    svg_content = chess.svg.board(board, size=size)
    png_bytes = cairosvg.svg2png(bytestring=svg_content.encode())
    return Image.open(io.BytesIO(png_bytes))


def get_first_solution_move(puzzle_solution: str) -> str:
    """Extract the first move from the puzzle solution string.

    Puzzle solutions are in UCI format: "g6f5 h3g5 f6g5 c1g5"
    We want just the first move: "g6f5"
    """
    moves = puzzle_solution.strip().split()
    if not moves:
        return ""
    return moves[0]


def validate_move(fen: str, uci_move: str) -> bool:
    """Validate that a UCI move is legal in the given position."""
    try:
        board = chess.Board(fen)
        move = chess.Move.from_uci(uci_move)
        return move in board.legal_moves
    except Exception:
        return False


def format_prompt(fen: str, rating: int, themes: list[str]) -> dict[str, str]:
    """Format the chess puzzle as an instruction-following example."""
    theme_str = ", ".join(themes) if themes else "general"
    return {
        "question": f"Find the best first move for this chess puzzle:\n{fen}",
        "context": f"Themes: {theme_str} | Rating: {rating}",
    }


def stream_and_sample_puzzles(
    dataset: Dataset,
    sample_size: int,
    theme_filter: tuple[str, ...] | None = None,
    min_popularity: int = 80,
    max_rating: int | None = None,
) -> list[int]:
    """Stream through dataset once and collect samples that meet criteria.

    TODO: Implement reservoir sampling for uniform random sampling across entire dataset
    while still maintaining single-pass efficiency.
    """
    themes_lower = [t.lower() for t in theme_filter] if theme_filter else None
    sampled_indices: list[int] = []

    with tqdm(total=sample_size, desc="Sampling puzzles") as pbar:
        for idx, example in enumerate(cast_to_iterator(dataset)):
            if example["Popularity"] < min_popularity:
                continue

            if max_rating and example["Rating"] > max_rating:
                continue

            if themes_lower:
                puzzle_themes = example.get("Themes", [])
                if not any(pt.lower() in themes_lower for pt in puzzle_themes):
                    continue

            sampled_indices.append(idx)
            pbar.update(1)

            if len(sampled_indices) >= sample_size:
                break

    if len(sampled_indices) < sample_size:
        click.echo(f"\nWarning: Only found {len(sampled_indices)} puzzles matching criteria (requested {sample_size})")
    else:
        click.echo(f"\nFound {len(sampled_indices)} puzzles matching criteria")

    return sampled_indices


def cast_to_iterator(dataset: Dataset) -> Iterator[dict[str, object]]:
    """Cast dataset to iterator for type checking."""
    return iter(dataset)  # type: ignore[return-value]


def process_dataset(
    sample_size: int = 1000,
    test_split: float = 0.1,
    seed: int = 42,
    themes: tuple[str, ...] | None = None,
    min_popularity: int = 80,
    max_rating: int | None = None,
    image_size: int = 120,
) -> DatasetDict:
    """Process Lichess puzzles into puzzle solution dataset with images."""
    np.random.seed(seed)

    click.echo("Loading Lichess chess-puzzles dataset...")
    lichess_dataset: Dataset = load_dataset("Lichess/chess-puzzles", split="train")  # type: ignore[assignment]
    click.echo(f"Loaded {len(lichess_dataset)} puzzles")

    if themes:
        click.echo(f"Filtering by themes: {', '.join(themes)}")
    if max_rating:
        click.echo(f"Filtering by max rating: {max_rating}")
    click.echo(f"Scanning for {sample_size} puzzles with popularity >= {min_popularity}...")

    sampled_indices = stream_and_sample_puzzles(
        lichess_dataset,
        sample_size,
        theme_filter=themes,
        min_popularity=min_popularity,
        max_rating=max_rating,
    )

    if not sampled_indices:
        raise ValueError("No puzzles found matching the specified criteria")

    sampled_data = lichess_dataset.select(sampled_indices)

    click.echo("Processing puzzles and generating images...")
    processed_examples: list[dict[str, object]] = []
    skipped = 0

    for example in tqdm(cast_to_iterator(sampled_data), desc="Processing puzzles", total=len(sampled_data)):
        fen = str(example["FEN"])
        puzzle_solution = str(example["Moves"])

        first_move = get_first_solution_move(puzzle_solution)

        if not first_move:
            skipped += 1
            continue

        if not validate_move(fen, first_move):
            skipped += 1
            continue

        prompt_data = format_prompt(
            fen=fen,
            rating=int(example["Rating"]),  # type: ignore[arg-type]
            themes=list(example.get("Themes", [])),  # type: ignore[arg-type]
        )

        board_image = generate_board_image(fen, size=image_size)

        processed_examples.append(
            {
                "image": board_image,
                "fen": fen,
                "puzzle_solution": puzzle_solution,
                "first_move": first_move,
                "rating": int(example["Rating"]),  # type: ignore[arg-type]
                "themes": list(example.get("Themes", [])),  # type: ignore[arg-type]
                "question": prompt_data["question"],
                "context": prompt_data["context"],
                "answer": first_move,
            }
        )

    click.echo(f"\nSuccessfully processed {len(processed_examples)} positions")
    click.echo(f"Skipped {skipped} puzzles due to errors")

    test_size = int(len(processed_examples) * test_split)
    np.random.shuffle(processed_examples)

    train_examples = processed_examples[:-test_size] if test_size > 0 else processed_examples
    test_examples = processed_examples[-test_size:] if test_size > 0 else []

    click.echo(f"Split into {len(train_examples)} train and {len(test_examples)} test examples")

    features = Features(
        {
            "image": HFImage(),
            "fen": Value("string"),
            "puzzle_solution": Value("string"),
            "first_move": Value("string"),
            "rating": Value("int32"),
            "themes": Sequence(Value("string")),
            "question": Value("string"),
            "context": Value("string"),
            "answer": Value("string"),
        }
    )

    train_dataset = Dataset.from_list(train_examples, features=features)
    test_dataset = (
        Dataset.from_list(test_examples, features=features)
        if test_examples
        else Dataset.from_list([], features=features)
    )

    return DatasetDict({"train": train_dataset, "test": test_dataset})


@click.command()
@click.option("--sample-size", type=int, default=1000, help="Number of positions to sample")
@click.option("--test-split", type=float, default=0.1, help="Fraction for test set")
@click.option("--seed", type=int, default=42, help="Random seed")
@click.option("--themes", type=str, multiple=True, help="Filter by theme(s)")
@click.option("--min-popularity", type=int, default=80, help="Min popularity score (number of plays)")
@click.option("--max-rating", type=int, default=None, help="Max puzzle rating difficulty")
@click.option("--image-size", type=int, default=120, help="Board image size in pixels")
@click.option("--push-to-hub", is_flag=True, help="Push to HuggingFace Hub")
@click.option("--hub-repo", type=str, default="pilipolio/lichess-puzzles-solutions", help="Hub repo name")
@click.option("--save-local", type=str, default=None, help="Local save path")
def main(
    sample_size: int,
    test_split: float,
    seed: int,
    themes: tuple[str, ...],
    min_popularity: int,
    max_rating: int | None,
    image_size: int,
    push_to_hub: bool,
    hub_repo: str,
    save_local: str | None,
) -> None:
    """Prepare chess puzzle solution dataset with board images."""
    dataset_dict = process_dataset(
        sample_size=sample_size,
        test_split=test_split,
        seed=seed,
        themes=themes if themes else None,
        min_popularity=min_popularity,
        max_rating=max_rating,
        image_size=image_size,
    )

    click.echo("\nDataset Statistics:")
    click.echo(f"Train set: {len(dataset_dict['train'])} examples")
    click.echo(f"Test set: {len(dataset_dict['test'])} examples")

    click.echo("\nExample from train set:")
    example = dataset_dict["train"][0]
    click.echo(f"FEN: {example['fen']}")
    click.echo(f"Rating: {example['rating']}")
    click.echo(f"Themes: {example['themes']}")
    click.echo(f"Full solution: {example['puzzle_solution']}")
    click.echo(f"First move (answer): {example['answer']}")
    click.echo(f"Image size: {example['image'].size}")  # type: ignore[union-attr]

    if save_local:
        click.echo(f"\nSaving to {save_local}...")
        dataset_dict.save_to_disk(save_local)
        click.echo("Saved successfully")

    if push_to_hub:
        click.echo(f"\nPushing to HuggingFace Hub: {hub_repo}...")
        dataset_dict.push_to_hub(hub_repo)
        click.echo("Pushed successfully!")
        click.echo(f"\nDataset available at: https://huggingface.co/datasets/{hub_repo}")


if __name__ == "__main__":
    main()
