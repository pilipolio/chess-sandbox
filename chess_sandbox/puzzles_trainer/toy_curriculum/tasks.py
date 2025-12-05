"""Task formatters for toy chess curriculum."""

import random
from typing import Any, Literal

import chess

from chess_sandbox.puzzles_trainer.toy_curriculum.formats import (
    fen_to_editor_url,
    parse_piece_list,
)
from chess_sandbox.puzzles_trainer.toy_curriculum.generators import (
    GeneratedPosition,
    PieceType,
    generate_capture_exercise,
    generate_movement_exercise,
)
from chess_sandbox.puzzles_trainer.toy_curriculum.prompts import (
    build_capture_sequence_prompt,
    build_fen_to_piece_list_prompt,
    build_legal_moves_uci_prompt,
    build_movement_path_prompt,
    build_piece_list_to_fen_prompt,
)

ToyTaskType = Literal[
    "toy_capture_sequence",
    "toy_movement_path",
    "toy_fen_to_piece_list",
    "toy_piece_list_to_fen",
    "toy_fen_to_legal_moves_uci",
    "toy_piece_list_to_legal_moves_uci",
]


def format_capture_sequence(position: GeneratedPosition) -> dict[str, Any]:
    """Format capture exercise as SFT example.

    >>> from chess_sandbox.puzzles_trainer.toy_curriculum.generators import generate_capture_exercise
    >>> pos = generate_capture_exercise("knight", num_targets=1, seed=42)
    >>> task = format_capture_sequence(pos)
    >>> task["task_type"]
    'toy_capture_sequence'
    >>> "messages" in task
    True
    """
    prompt = build_capture_sequence_prompt(position.fen, position.piece_list)
    answer = ", ".join(position.solution_moves)

    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
        "task_type": "toy_capture_sequence",
        "fen": position.fen,
        "piece_list": position.piece_list,
        "question": prompt,
        "answer": answer,
        "difficulty": position.difficulty,
        "source": "synthetic",
        "source_url": fen_to_editor_url(position.fen),
    }


def format_movement_path(position: GeneratedPosition) -> dict[str, Any]:
    """Format movement path exercise as SFT example.

    >>> from chess_sandbox.puzzles_trainer.toy_curriculum.generators import generate_movement_exercise
    >>> pos = generate_movement_exercise("knight", min_moves=2, max_moves=3, seed=42)
    >>> task = format_movement_path(pos)
    >>> task["task_type"]
    'toy_movement_path'
    """
    # Extract start square from piece list
    pieces = parse_piece_list(position.piece_list)
    start_square = chess.square_name(pieces[0].square) if pieces else "a1"
    target_square = chess.square_name(position.target_square) if position.target_square else "h8"

    prompt = build_movement_path_prompt(position.piece_list, start_square, target_square)
    answer = ", ".join(position.solution_moves)

    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
        "task_type": "toy_movement_path",
        "fen": position.fen,
        "piece_list": position.piece_list,
        "question": prompt,
        "answer": answer,
        "difficulty": position.difficulty,
        "target_square": target_square,
        "source": "synthetic",
        "source_url": fen_to_editor_url(position.fen),
    }


def format_fen_to_piece_list(position: GeneratedPosition) -> dict[str, Any]:
    """Format FEN to piece list conversion task."""
    prompt = build_fen_to_piece_list_prompt(position.fen)
    answer = position.piece_list

    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
        "task_type": "toy_fen_to_piece_list",
        "fen": position.fen,
        "question": prompt,
        "answer": answer,
        "source": "synthetic",
        "source_url": fen_to_editor_url(position.fen),
    }


def format_piece_list_to_fen(position: GeneratedPosition) -> dict[str, Any]:
    """Format piece list to FEN conversion task."""
    prompt = build_piece_list_to_fen_prompt(position.piece_list)
    answer = position.fen

    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
        "task_type": "toy_piece_list_to_fen",
        "fen": position.fen,
        "piece_list": position.piece_list,
        "question": prompt,
        "answer": answer,
        "source": "synthetic",
        "source_url": fen_to_editor_url(position.fen),
    }


def format_legal_moves_uci(position: GeneratedPosition) -> dict[str, Any]:
    """Format legal moves (UCI) task."""
    board = chess.Board(position.fen)
    legal_moves = sorted([m.uci() for m in board.legal_moves])
    answer = ", ".join(legal_moves)

    prompt = build_legal_moves_uci_prompt(position.fen, position.piece_list)

    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
        "task_type": "toy_fen_to_legal_moves_uci",
        "fen": position.fen,
        "piece_list": position.piece_list,
        "question": prompt,
        "answer": answer,
        "source": "synthetic",
        "source_url": fen_to_editor_url(position.fen),
    }


def format_piece_list_legal_moves_uci(position: GeneratedPosition) -> dict[str, Any]:
    """Format piece list to legal moves (UCI) task - uses piece list as input."""
    board = chess.Board(position.fen)
    legal_moves = sorted([m.uci() for m in board.legal_moves])
    answer = ", ".join(legal_moves)

    # Use piece list in prompt instead of FEN
    prompt = f"""List all legal moves for the white piece in UCI notation, comma-separated.

Position (pieces): {position.piece_list}
Answer:"""

    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
        "task_type": "toy_piece_list_to_legal_moves_uci",
        "fen": position.fen,
        "piece_list": position.piece_list,
        "question": prompt,
        "answer": answer,
        "source": "synthetic",
        "source_url": fen_to_editor_url(position.fen),
    }


def format_toy_representation_tasks(position: GeneratedPosition) -> list[dict[str, Any]]:
    """Generate all representation task variations for a position.

    Returns 4 task dicts:
    - FEN -> piece list
    - piece list -> FEN
    - FEN -> legal moves (UCI)
    - piece list -> legal moves (UCI)
    """
    return [
        format_fen_to_piece_list(position),
        format_piece_list_to_fen(position),
        format_legal_moves_uci(position),
        format_piece_list_legal_moves_uci(position),
    ]


def create_toy_curriculum(
    capture_exercises: int = 100,
    movement_exercises: int = 100,
    include_representation: bool = True,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Create toy curriculum dataset with all task types.

    >>> curriculum = create_toy_curriculum(
    ...     capture_exercises=5, movement_exercises=5, include_representation=False, seed=42
    ... )
    >>> len(curriculum) >= 5
    True
    >>> "toy_capture_sequence" in set(t["task_type"] for t in curriculum)
    True

    >>> curriculum = create_toy_curriculum(
    ...     capture_exercises=2, movement_exercises=2, include_representation=True, seed=42
    ... )
    >>> len(curriculum) > 4
    True
    """
    random.seed(seed)
    tasks: list[dict[str, Any]] = []

    attacker_types: list[PieceType] = ["knight", "rook", "bishop", "queen"]
    movement_types: list[PieceType] = ["knight", "rook", "bishop", "queen"]

    # Generate capture exercises
    for i in range(capture_exercises):
        attacker = attacker_types[i % len(attacker_types)]
        num_targets = (i % 3) + 1  # Cycle through 1, 2, 3 targets

        try:
            position = generate_capture_exercise(attacker, num_targets, seed=seed + i)
            tasks.append(format_capture_sequence(position))

            if include_representation:
                tasks.extend(format_toy_representation_tasks(position))
        except ValueError:
            continue  # Skip if generation fails

    # Generate movement exercises
    for i in range(movement_exercises):
        piece = movement_types[i % len(movement_types)]
        min_moves = 1 if piece != "knight" else 2
        max_moves = 2 if piece != "knight" else 4

        try:
            position = generate_movement_exercise(piece, min_moves, max_moves, seed=seed + capture_exercises + i)
            tasks.append(format_movement_path(position))

            if include_representation:
                tasks.extend(format_toy_representation_tasks(position))
        except ValueError:
            continue

    random.shuffle(tasks)
    return tasks
