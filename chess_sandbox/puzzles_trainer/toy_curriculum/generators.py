"""Synthetic position generation for toy chess exercises."""

import random
from dataclasses import dataclass
from typing import Literal

import chess

from chess_sandbox.puzzles_trainer.toy_curriculum.formats import (
    PieceSpec,
    fen_to_piece_list,
    piece_list_to_fen,
)
from chess_sandbox.puzzles_trainer.toy_curriculum.paths import find_shortest_path

PieceType = Literal["knight", "rook", "bishop", "queen", "pawn"]

PIECE_TYPE_MAP: dict[PieceType, chess.PieceType] = {
    "knight": chess.KNIGHT,
    "rook": chess.ROOK,
    "bishop": chess.BISHOP,
    "queen": chess.QUEEN,
    "pawn": chess.PAWN,
}


@dataclass
class GeneratedPosition:
    """Result of synthetic position generation."""

    fen: str
    piece_list: str
    solution_moves: list[str]  # UCI format
    difficulty: int  # 1-3 scale
    target_square: chess.Square | None = None  # For movement exercises


def _get_piece_attacks(piece_type: chess.PieceType, square: chess.Square) -> chess.SquareSet:
    """Get squares attacked by a piece on an empty board."""
    board = chess.Board.empty()
    board.set_piece_at(square, chess.Piece(piece_type, chess.WHITE))
    return board.attacks(square)


def _find_capture_sequence(
    board: chess.Board,
    attacker_square: chess.Square,
    targets: list[chess.Square],
) -> list[str] | None:
    """Find a sequence of moves to capture all targets.

    Uses greedy approach: capture closest reachable target each time.
    Returns None if not all targets can be captured.

    Note: This simulates consecutive captures by White, ignoring turn order.
    """
    if not targets:
        return []

    board = board.copy()
    remaining = set(targets)
    moves: list[str] = []
    current_square = attacker_square

    while remaining:
        # Ensure it's White's turn for our attacker
        board.turn = chess.WHITE

        legal_captures = [m for m in board.legal_moves if m.from_square == current_square and m.to_square in remaining]

        if not legal_captures:
            return None

        # Pick the first legal capture (could be optimized)
        move = legal_captures[0]
        moves.append(move.uci())

        remaining.remove(move.to_square)
        board.push(move)
        current_square = move.to_square

    return moves


def generate_capture_exercise(
    attacker_type: PieceType,
    num_targets: int = 1,
    seed: int | None = None,
) -> GeneratedPosition:
    """Generate single-piece capture exercise.

    Creates position with one White attacker and 1-3 Black pawns as targets.

    >>> pos = generate_capture_exercise("knight", num_targets=1, seed=42)
    >>> len(pos.solution_moves)
    1
    >>> pos.difficulty
    1

    >>> pos = generate_capture_exercise("rook", num_targets=2, seed=123)
    >>> pos.solution_moves
    ['a5a6', 'a6a2']

    >>> pos = generate_capture_exercise("queen", num_targets=3, seed=456)
    >>> pos.solution_moves
    ['b6h6', 'h6e3', 'e3b3']
    """
    if seed is not None:
        random.seed(seed)

    num_targets = max(1, min(3, num_targets))
    chess_piece_type = PIECE_TYPE_MAP[attacker_type]

    for _ in range(100):  # Max retries
        # Place attacker (avoid edges for pawns)
        if attacker_type == "pawn":
            attacker_file = random.randint(0, 7)
            attacker_rank = random.randint(1, 5)  # Ranks 2-6 for pawn mobility
        else:
            attacker_file = random.randint(0, 7)
            attacker_rank = random.randint(0, 7)

        attacker_square = chess.square(attacker_file, attacker_rank)

        # Build board with just the attacker
        piece_symbol_map = {"knight": "N", "rook": "R", "bishop": "B", "queen": "Q", "pawn": "P"}
        pieces = [PieceSpec(piece_symbol_map[attacker_type], attacker_square, chess.WHITE)]

        # Find squares the attacker can reach
        attacks = _get_piece_attacks(chess_piece_type, attacker_square)
        valid_target_squares = [sq for sq in attacks if chess.square_rank(sq) not in (0, 7)]  # No pawns on 1st/8th

        if len(valid_target_squares) < num_targets:
            continue

        # For multi-target captures, we need to ensure targets can be captured in sequence
        # For rooks: all targets on same rank or same file work best
        # For knights/bishops/queens: need BFS approach
        if num_targets == 1:
            target_squares = random.sample(valid_target_squares, 1)
        elif attacker_type == "rook" and num_targets > 1:
            # Group targets by rank and file
            attacker_file_sq = chess.square_file(attacker_square)
            attacker_rank_sq = chess.square_rank(attacker_square)
            same_file = [sq for sq in valid_target_squares if chess.square_file(sq) == attacker_file_sq]
            same_rank = [sq for sq in valid_target_squares if chess.square_rank(sq) == attacker_rank_sq]

            # Pick from the group with enough targets
            if len(same_file) >= num_targets:
                target_squares = random.sample(same_file, num_targets)
            elif len(same_rank) >= num_targets:
                target_squares = random.sample(same_rank, num_targets)
            else:
                continue  # Can't find enough targets on same line
        else:
            # For other pieces, try random sampling and check if sequence works
            target_squares = random.sample(valid_target_squares, num_targets)

        for sq in target_squares:
            pieces.append(PieceSpec("P", sq, chess.BLACK))

        fen = piece_list_to_fen(pieces, turn=chess.WHITE)
        board = chess.Board(fen)

        # Find capture sequence
        solution = _find_capture_sequence(board, attacker_square, target_squares)

        if solution and len(solution) == num_targets:
            return GeneratedPosition(
                fen=fen,
                piece_list=fen_to_piece_list(fen),
                solution_moves=solution,
                difficulty=num_targets,
            )

    raise ValueError("Could not generate valid capture exercise after 100 retries")


def generate_movement_exercise(
    piece_type: PieceType,
    min_moves: int = 2,
    max_moves: int = 4,
    seed: int | None = None,
) -> GeneratedPosition:
    """Generate piece movement path exercise.

    Creates empty board with single piece and finds a target requiring min-max moves.

    >>> pos = generate_movement_exercise("knight", min_moves=2, max_moves=3, seed=42)
    >>> 2 <= len(pos.solution_moves) <= 3
    True

    >>> pos = generate_movement_exercise("rook", min_moves=1, max_moves=2, seed=123)
    >>> 1 <= len(pos.solution_moves) <= 2
    True

    >>> pos = generate_movement_exercise("bishop", min_moves=1, max_moves=2, seed=456)
    >>> 1 <= len(pos.solution_moves) <= 2
    True
    """
    if seed is not None:
        random.seed(seed)

    if piece_type == "pawn":
        raise ValueError("Pawn movement exercises not supported (complex rules)")

    chess_piece_type = PIECE_TYPE_MAP[piece_type]

    for _ in range(100):
        start_square = chess.square(random.randint(0, 7), random.randint(0, 7))

        # Find all reachable squares at each distance via BFS
        candidates: list[tuple[chess.Square, list[str]]] = []

        for target_square in chess.SQUARES:
            if target_square == start_square:
                continue

            path = find_shortest_path(chess_piece_type, start_square, target_square)
            if path is not None and min_moves <= len(path) <= max_moves:
                candidates.append((target_square, path))

        if candidates:
            target_square, solution = random.choice(candidates)

            piece_symbol_map = {"knight": "N", "rook": "R", "bishop": "B", "queen": "Q"}
            pieces = [PieceSpec(piece_symbol_map[piece_type], start_square, chess.WHITE)]
            fen = piece_list_to_fen(pieces, turn=chess.WHITE)

            return GeneratedPosition(
                fen=fen,
                piece_list=fen_to_piece_list(fen),
                solution_moves=solution,
                difficulty=len(solution),
                target_square=target_square,
            )

    raise ValueError("Could not generate valid movement exercise after 100 retries")
