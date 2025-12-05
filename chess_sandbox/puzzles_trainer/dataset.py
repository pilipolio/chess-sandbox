"""Chess puzzle dataset loading and formatting with multiple task types."""

import random
from collections.abc import Iterator
from typing import Literal

import chess
from datasets import Dataset, DatasetDict, Features, Value, load_dataset
from datasets import Image as HFImage
from tqdm import tqdm

from chess_sandbox.puzzles_trainer.helper import generate_board_image
from chess_sandbox.puzzles_trainer.prompts import (
    build_ascii_board_prompt,
    build_concept_detection_prompt,
    build_legal_captures_prompt,
    build_legal_moves_prompt,
    build_piece_captures_prompt,
    build_piece_positions_prompt,
    build_puzzle_prompt,
)

DATASET_ID = "Lichess/chess-puzzles"

TaskType = Literal[
    "puzzle", "ascii_board", "legal_moves", "legal_captures", "piece_captures", "concept_detection", "piece_positions"
]


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


def _cast_to_iterator(dataset: Dataset) -> Iterator[dict[str, object]]:
    """Cast dataset to iterator for type checking."""
    return iter(dataset)  # type: ignore[return-value]


def stream_and_sample_puzzles(
    dataset: Dataset,
    sample_size: int,
    theme_filter: tuple[str, ...] | None = None,
    min_popularity: int = 80,
    max_rating: int | None = None,
) -> list[int]:
    """Stream through dataset once and collect samples that meet criteria."""
    themes_lower = [t.lower() for t in theme_filter] if theme_filter else None
    sampled_indices: list[int] = []

    with tqdm(total=sample_size, desc="Sampling puzzles") as pbar:
        for idx, example in enumerate(_cast_to_iterator(dataset)):
            if example["Popularity"] < min_popularity:  # pyright: ignore[reportOperatorIssue]
                continue

            if max_rating and example["Rating"] > max_rating:  # pyright: ignore[reportOperatorIssue]
                continue

            if themes_lower:
                puzzle_themes = example.get("Themes", [])
                if not any(pt.lower() in themes_lower for pt in puzzle_themes):  # pyright: ignore[reportUnknownArgumentType,reportUnknownMemberType]
                    continue

            sampled_indices.append(idx)
            pbar.update(1)

            if len(sampled_indices) >= sample_size:
                break

    if len(sampled_indices) < sample_size:
        print(f"\nWarning: Only found {len(sampled_indices)} puzzles matching criteria (requested {sample_size})")
    else:
        print(f"\nFound {len(sampled_indices)} puzzles matching criteria")

    return sampled_indices


def normalize_lichess_example(example: dict) -> dict:
    """Normalize Lichess dataset example to internal format.

    Lichess schema: FEN, Moves, Rating, Themes, Popularity, PuzzleId
    Internal schema: fen, answer, themes, lichess_url
    """
    fen = str(example["FEN"])
    moves = str(example["Moves"])
    first_move = get_first_solution_move(moves)
    themes = list(example.get("Themes", []))
    puzzle_id = str(example["PuzzleId"])

    return {
        "fen": fen,
        "answer": first_move,
        "themes": themes,
        "lichess_url": f"https://lichess.org/training/{puzzle_id}",
    }


def format_puzzle(example: dict) -> dict:
    """Format puzzle as chat messages for SFT with few-shot examples."""
    fen = example["fen"]
    prompt = build_puzzle_prompt(fen)

    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": example["answer"]},
        ],
        "task_type": "puzzle",
        "fen": fen,
        "question": prompt,
        "answer": example["answer"],
        "lichess_url": example["lichess_url"],
    }


def format_ascii_board(example: dict) -> dict:
    """Format ASCII board task with one-shot example."""
    fen = example["fen"]
    board = chess.Board(fen)
    ascii_output = str(board)
    prompt = build_ascii_board_prompt(fen)

    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": ascii_output},
        ],
        "task_type": "ascii_board",
        "fen": fen,
        "question": prompt,
        "answer": ascii_output,
        "lichess_url": example["lichess_url"],
    }


def format_legal_moves(example: dict) -> dict | None:
    """Format legal moves task for a random piece.

    Returns None if no suitable piece is found.
    """
    fen = example["fen"]
    board = chess.Board(fen)

    piece_squares = [
        sq
        for sq in chess.SQUARES
        if board.piece_at(sq) and board.piece_at(sq).color == board.turn  # pyright: ignore[reportOptionalMemberAccess]
    ]

    if not piece_squares:
        return None

    random.shuffle(piece_squares)

    for square in piece_squares:
        moves = [m for m in board.legal_moves if m.from_square == square]
        if moves:
            square_name = chess.square_name(square)
            san_moves = ", ".join(board.san(m) for m in moves)
            prompt = build_legal_moves_prompt(fen, square_name)

            return {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": san_moves},
                ],
                "task_type": "legal_moves",
                "fen": fen,
                "square": square_name,
                "question": prompt,
                "answer": san_moves,
                "lichess_url": example["lichess_url"],
            }

    return None


def format_legal_captures(example: dict) -> dict | None:
    """Format legal captures task listing all captures for side to move.

    Returns None if no captures are available.
    """
    fen = example["fen"]
    board = chess.Board(fen)

    captures = [m for m in board.legal_moves if board.is_capture(m)]

    if not captures:
        return None

    san_captures = ", ".join(board.san(m) for m in captures)
    prompt = build_legal_captures_prompt(fen)

    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": san_captures},
        ],
        "task_type": "legal_captures",
        "fen": fen,
        "question": prompt,
        "answer": san_captures,
        "lichess_url": example["lichess_url"],
    }


def format_piece_captures(example: dict) -> dict | None:
    """Format piece captures task - captures for a specific piece.

    Returns None if no piece has captures available.
    """
    fen = example["fen"]
    board = chess.Board(fen)

    piece_squares = [
        sq
        for sq in chess.SQUARES
        if board.piece_at(sq) and board.piece_at(sq).color == board.turn  # pyright: ignore[reportOptionalMemberAccess]
    ]

    if not piece_squares:
        return None

    random.shuffle(piece_squares)

    for square in piece_squares:
        captures = [m for m in board.legal_moves if m.from_square == square and board.is_capture(m)]

        if captures:
            square_name = chess.square_name(square)
            san_captures = ", ".join(board.san(m) for m in captures)
            prompt = build_piece_captures_prompt(fen, square_name)

            return {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": san_captures},
                ],
                "task_type": "piece_captures",
                "fen": fen,
                "square": square_name,
                "question": prompt,
                "answer": san_captures,
                "lichess_url": example["lichess_url"],
            }

    return None


def format_concept_detection(example: dict) -> dict:
    """Format concept detection task using dataset themes."""
    fen = example["fen"]
    themes = example.get("themes", [])
    themes_str = ", ".join(themes) if themes else "none"
    prompt = build_concept_detection_prompt(fen)

    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": themes_str},
        ],
        "task_type": "concept_detection",
        "fen": fen,
        "question": prompt,
        "answer": themes_str,
        "lichess_url": example["lichess_url"],
    }


def format_piece_positions(example: dict) -> dict:
    """Format piece positions task listing all pieces by color."""
    fen = example["fen"]
    board = chess.Board(fen)

    white_pieces: list[str] = []
    black_pieces: list[str] = []

    for square, piece in board.piece_map().items():
        piece_str = piece.symbol().upper() + chess.square_name(square)
        if piece.color == chess.WHITE:
            white_pieces.append(piece_str)
        else:
            black_pieces.append(piece_str)

    white_pieces.sort(key=lambda p: ("KQRBNP".index(p[0]), p[1:]))
    black_pieces.sort(key=lambda p: ("KQRBNP".index(p[0]), p[1:]))

    output = f"White: {', '.join(white_pieces)}\nBlack: {', '.join(black_pieces)}"
    prompt = build_piece_positions_prompt(fen)

    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": output},
        ],
        "task_type": "piece_positions",
        "fen": fen,
        "question": prompt,
        "answer": output,
        "lichess_url": example["lichess_url"],
    }


def create_mixed_dataset(examples: list[dict]) -> list[dict]:
    """Create mixed dataset with all task types from normalized puzzle examples."""
    result: list[dict] = []

    for example in examples:
        result.append(format_puzzle(example))
        result.append(format_ascii_board(example))
        # concept_detection disabled: labels are untrustworthy
        # result.append(format_concept_detection(example))
        result.append(format_piece_positions(example))

        legal_moves_example = format_legal_moves(example)
        if legal_moves_example:
            result.append(legal_moves_example)

        legal_captures_example = format_legal_captures(example)
        if legal_captures_example:
            result.append(legal_captures_example)

        piece_captures_example = format_piece_captures(example)
        if piece_captures_example:
            result.append(piece_captures_example)

    random.shuffle(result)
    return result


def load_puzzle_dataset(
    sample_size: int = 1000,
    test_split: float = 0.1,
    seed: int = 42,
    min_popularity: int = 80,
    max_rating: int | None = None,
    themes: tuple[str, ...] | None = None,
) -> tuple[Dataset, Dataset]:
    """Load and format the chess puzzles dataset with mixed tasks.

    Args:
        sample_size: Number of source puzzles to sample.
        test_split: Fraction of data for test set.
        seed: Random seed for reproducibility.
        min_popularity: Minimum popularity score (default 80).
        max_rating: Maximum puzzle rating (None for no limit).
        themes: Filter by theme(s) (None for all themes).

    Returns:
        Tuple of (train_dataset, test_dataset).
    """
    random.seed(seed)

    print(f"Loading dataset: {DATASET_ID}")
    lichess_dataset: Dataset = load_dataset(DATASET_ID, split="train")  # pyright: ignore[reportAssignmentType]
    print(f"Loaded {len(lichess_dataset)} puzzles")

    print(f"Sampling {sample_size} puzzles with popularity >= {min_popularity}...")
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

    print("Normalizing and validating puzzles...")
    normalized_examples: list[dict] = []
    skipped = 0

    for example in tqdm(_cast_to_iterator(sampled_data), desc="Processing", total=len(sampled_data)):
        normalized = normalize_lichess_example(dict(example))  # pyright: ignore[reportUnknownArgumentType]

        if not normalized["answer"]:
            skipped += 1
            continue

        if not validate_move(normalized["fen"], normalized["answer"]):
            skipped += 1
            continue

        normalized_examples.append(normalized)

    print(f"Processed {len(normalized_examples)} puzzles, skipped {skipped}")

    # Split into train/test
    test_size = int(len(normalized_examples) * test_split)
    random.shuffle(normalized_examples)

    train_examples = normalized_examples[:-test_size] if test_size > 0 else normalized_examples
    test_examples = normalized_examples[-test_size:] if test_size > 0 else []

    print("Creating mixed dataset with all task types...")
    train_mixed = create_mixed_dataset(train_examples)
    test_mixed = create_mixed_dataset(test_examples)

    train_dataset = Dataset.from_list(train_mixed)
    test_dataset = Dataset.from_list(test_mixed)

    print(f"Train examples: {len(train_dataset)}")
    print(f"Test examples: {len(test_dataset)}")

    task_counts: dict[str, int] = {}
    for ex in train_mixed[:1000]:
        task_type = ex.get("task_type", "unknown")
        task_counts[task_type] = task_counts.get(task_type, 0) + 1
    print(f"Task distribution (first 1000): {task_counts}")

    return train_dataset, test_dataset


def materialize_task_dataset(
    sample_size: int = 1000,
    test_split: float = 0.1,
    seed: int = 42,
    image_size: int = 240,
    min_popularity: int = 80,
    max_rating: int | None = None,
    themes: tuple[str, ...] | None = None,
) -> DatasetDict:
    """Create task dataset with board images ready for HF Hub.

    Args:
        sample_size: Number of source puzzles to sample.
        test_split: Fraction of data for test set.
        seed: Random seed for reproducibility.
        image_size: Board image size in pixels.
        min_popularity: Minimum popularity score (default 80).
        max_rating: Maximum puzzle rating (None for no limit).
        themes: Filter by theme(s) (None for all themes).

    Returns:
        DatasetDict with train and test splits.
    """
    random.seed(seed)

    print(f"Loading dataset: {DATASET_ID}")
    lichess_dataset: Dataset = load_dataset(DATASET_ID, split="train")  # pyright: ignore[reportAssignmentType]
    print(f"Loaded {len(lichess_dataset)} puzzles")

    print(f"Sampling {sample_size} puzzles with popularity >= {min_popularity}...")
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

    print("Normalizing and validating puzzles...")
    normalized_examples: list[dict] = []
    skipped = 0

    for example in tqdm(_cast_to_iterator(sampled_data), desc="Processing", total=len(sampled_data)):
        normalized = normalize_lichess_example(dict(example))  # pyright: ignore[reportUnknownArgumentType]

        if not normalized["answer"]:
            skipped += 1
            continue

        if not validate_move(normalized["fen"], normalized["answer"]):
            skipped += 1
            continue

        normalized_examples.append(normalized)

    print(f"Processed {len(normalized_examples)} puzzles, skipped {skipped}")

    # Split into train/test
    test_size = int(len(normalized_examples) * test_split)
    random.shuffle(normalized_examples)

    train_examples = normalized_examples[:-test_size] if test_size > 0 else normalized_examples
    test_examples = normalized_examples[-test_size:] if test_size > 0 else []

    print("Creating mixed dataset with all task types...")
    train_mixed = create_mixed_dataset(train_examples)
    test_mixed = create_mixed_dataset(test_examples)

    def add_images(examples: list[dict], desc: str) -> list[dict]:
        """Add board images to examples."""
        for ex in tqdm(examples, desc=desc):
            ex["image"] = generate_board_image(ex["fen"], size=image_size)
        return examples

    print("Generating board images...")
    train_mixed = add_images(train_mixed, "Train images")
    test_mixed = add_images(test_mixed, "Test images")

    features = Features(
        {
            "image": HFImage(),
            "fen": Value("string"),
            "task_type": Value("string"),
            "question": Value("string"),
            "answer": Value("string"),
            "lichess_url": Value("string"),
            "messages": [
                {
                    "role": Value("string"),
                    "content": Value("string"),
                }
            ],
        }
    )

    def normalize_example(ex: dict) -> dict:
        """Ensure all examples have consistent fields."""
        return {
            "image": ex["image"],
            "fen": ex["fen"],
            "task_type": ex["task_type"],
            "question": ex["question"],
            "answer": ex["answer"],
            "lichess_url": ex["lichess_url"],
            "messages": ex["messages"],
        }

    train_mixed = [normalize_example(ex) for ex in train_mixed]
    test_mixed = [normalize_example(ex) for ex in test_mixed]

    train_dataset = Dataset.from_list(train_mixed, features=features)
    test_dataset = Dataset.from_list(test_mixed, features=features)

    print(f"Train examples: {len(train_dataset)}")
    print(f"Test examples: {len(test_dataset)}")

    task_counts: dict[str, int] = {}
    for ex in train_mixed[:1000]:
        task_type = ex.get("task_type", "unknown")
        task_counts[task_type] = task_counts.get(task_type, 0) + 1
    print(f"Task distribution (first 1000): {task_counts}")

    return DatasetDict({"train": train_dataset, "test": test_dataset})
