"""Chess puzzle dataset loading and formatting with multiple task types."""

import random
from typing import Literal

import chess
from datasets import Dataset, DatasetDict, load_dataset

from chess_sandbox.puzzles_trainer.prompts import build_puzzle_prompt

DATASET_ID = "pilipolio/lichess-puzzles-solutions"

TaskType = Literal["puzzle", "ascii_board", "legal_moves", "concept_detection", "piece_positions"]

ASCII_BOARD_EXAMPLE_FEN = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
ASCII_BOARD_EXAMPLE_OUTPUT = """r n b q k b n r
p p p p p p p p
. . . . . . . .
. . . . . . . .
. . . . P . . .
. . . . . . . .
P P P P . P P P
R N B Q K B N R"""


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
    }


def format_ascii_board(example: dict) -> dict:
    """Format ASCII board task with one-shot example."""
    fen = example["fen"]
    board = chess.Board(fen)
    ascii_output = str(board)

    prompt = f"""Generate an ASCII representation of this chess position.

Example:
FEN: {ASCII_BOARD_EXAMPLE_FEN}
Output:
{ASCII_BOARD_EXAMPLE_OUTPUT}

Now generate for:
FEN: {fen}"""

    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": ascii_output},
        ],
        "task_type": "ascii_board",
        "fen": fen,
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

            prompt = f"""List all legal moves for the piece on {square_name} in this position.
FEN: {fen}"""

            return {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": san_moves},
                ],
                "task_type": "legal_moves",
                "fen": fen,
            }

    return None


def format_concept_detection(example: dict) -> dict:
    """Format concept detection task using dataset themes."""
    fen = example["fen"]
    themes = example.get("themes", [])
    themes_str = ", ".join(themes) if themes else "none"

    prompt = f"""What chess concepts or themes are present in this position?
FEN: {fen}"""

    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": themes_str},
        ],
        "task_type": "concept_detection",
        "fen": fen,
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

    prompt = f"""List all pieces and their positions from this FEN.
FEN: {fen}"""

    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": output},
        ],
        "task_type": "piece_positions",
        "fen": fen,
    }


def create_mixed_dataset(dataset: Dataset) -> list[dict]:
    """Create mixed dataset with all task types from puzzles."""
    examples: list[dict] = []

    for example in dataset:
        example_dict = dict(example)  # pyright: ignore[reportUnknownArgumentType]

        examples.append(format_puzzle(example_dict))
        examples.append(format_ascii_board(example_dict))
        examples.append(format_concept_detection(example_dict))
        examples.append(format_piece_positions(example_dict))

        legal_moves_example = format_legal_moves(example_dict)
        if legal_moves_example:
            examples.append(legal_moves_example)

    random.shuffle(examples)
    return examples


def load_puzzle_dataset() -> tuple[Dataset, Dataset]:
    """Load and format the chess puzzles dataset with mixed tasks."""
    print(f"Loading dataset: {DATASET_ID}")
    dataset: DatasetDict = load_dataset(DATASET_ID)  # pyright: ignore[reportAssignmentType]

    print("Creating mixed dataset with all task types...")
    train_examples = create_mixed_dataset(dataset["train"])
    test_examples = create_mixed_dataset(dataset["test"])

    train_dataset = Dataset.from_list(train_examples)
    test_dataset = Dataset.from_list(test_examples)

    print(f"Train examples: {len(train_dataset)}")
    print(f"Test examples: {len(test_dataset)}")

    task_counts = {}
    for ex in train_examples[:1000]:
        task_type = ex.get("task_type", "unknown")
        task_counts[task_type] = task_counts.get(task_type, 0) + 1
    print(f"Task distribution (first 1000): {task_counts}")

    return train_dataset, test_dataset
