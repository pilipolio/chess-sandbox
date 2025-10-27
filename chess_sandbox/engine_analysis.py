#!/usr/bin/env python3
"""
Deterministic Chess Engine Analysis Module

Provides Stockfish-based position analysis and text formatting utilities.
"""

import chess
import chess.engine
import click
from chess.engine import InfoDict
from pydantic import BaseModel

from chess_sandbox.config import settings


class PrincipalVariation(BaseModel):
    score: float | None
    san_moves: list[str]


def info_to_pv(info: InfoDict, board: chess.Board) -> PrincipalVariation | None:
    score = info.get("score")

    pv = info.get("pv", [])

    temp_board = board.copy()
    san_moves: list[str] = []
    for move in pv:
        san_moves.append(temp_board.san(move))
        temp_board.push(move)

    score_value = score.relative.score() if score else None

    return PrincipalVariation(score=score_value / 100 if score_value is not None else None, san_moves=san_moves)


def analyze_position(board: chess.Board, num_lines: int = 5, depth: int = 20) -> list[PrincipalVariation]:
    """Analyze position with Stockfish, returning top principal variations.

    >>> board = chess.Board()  # Starting position
    >>> results = analyze_position(board, num_lines=1, depth=1)
    >>> results
    [PrincipalVariation(score=0.17, san_moves=['e4'])]
    """
    if num_lines == 0:
        return []

    engine = chess.engine.SimpleEngine.popen_uci(settings.STOCKFISH_PATH)
    analysis_results = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=num_lines)
    engine.quit()

    principal_variations: list[PrincipalVariation] = []
    for info in analysis_results:
        pv = info_to_pv(info, board)
        if pv is not None:
            principal_variations.append(pv)

    return principal_variations


def format_as_text(
    board: chess.Board, principal_variations: list[PrincipalVariation], max_display_moves: int = 8
) -> str:
    lines: list[str] = []
    lines.append("POSITION:")
    lines.append(str(board))
    lines.append(f"\nFEN: {board.fen()}")
    lines.append(f"Turn: {'White' if board.turn == chess.WHITE else 'Black'} to move\n")

    if len(principal_variations) > 0:
        lines.append("PRINCIPAL VARIATIONS:\n")

        for i, pv in enumerate[PrincipalVariation](principal_variations, 1):
            eval_str = f"{pv.score:+.2f}"
            displayed_moves = pv.san_moves[:max_display_moves]
            move_sequence = " ".join(displayed_moves)

            lines.append(f"Line {i}: Eval {eval_str}")
            lines.append(f"  Moves: {move_sequence}")

            if len(pv.san_moves) > max_display_moves:
                lines.append(f"  ... and {len(pv.san_moves) - max_display_moves} more moves")
            lines.append("")

    return "\n".join(lines)


@click.command()
@click.argument("fen")
@click.option("--next-move", required=False)
@click.option("--depth", default=20, help="Analysis depth (default: 20)")
@click.option("--num-lines", default=5, help="Number of lines to analyze (default: 5)")
def main(fen: str, next_move: str, depth: int, num_lines: int):
    """Analyze a chess position given in FEN notation."""
    board = chess.Board(fen)

    if next_move:
        board.push_san(next_move)

    analysis_results = analyze_position(board, num_lines=num_lines, depth=depth)
    print(format_as_text(board, analysis_results))


if __name__ == "__main__":
    main()
