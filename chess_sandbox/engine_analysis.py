#!/usr/bin/env python3
"""
Deterministic Chess Engine Analysis Module

Provides Stockfish-based position analysis and text formatting utilities.
"""

from collections.abc import Sequence

import chess
import chess.engine
import click
from chess.engine import InfoDict

STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"


def format_score(score: chess.engine.PovScore) -> str:
    score_value = score.relative.score()
    if score_value is not None:
        return f"{score_value/100:+.2f}"
    return "0.00"


def analyze_position(
    board: chess.Board, stockfish_path: str = STOCKFISH_PATH, num_lines: int = 5, depth: int = 20
) -> Sequence[InfoDict]:
    if num_lines == 0:
        return []

    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    analysis_results = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=num_lines)
    engine.quit()
    return analysis_results


def format_as_text(board: chess.Board, analysis_results: Sequence[InfoDict]) -> str:
    lines: list[str] = []
    lines.append("POSITION:")
    lines.append(str(board))
    lines.append(f"\nFEN: {board.fen()}")
    lines.append(f"Turn: {'White' if board.turn == chess.WHITE else 'Black'} to move\n")

    if len(analysis_results) > 0:
        lines.append("TOP ENGINE LINES:\n")

        for i, info in enumerate(analysis_results, 1):
            score = info.get("score")
            pv = info.get("pv", [])
            depth = info.get("depth", "?")

            temp_board = board.copy()
            san_moves: list[str] = []
            for move in pv[:8]:
                san_moves.append(temp_board.san(move))
                temp_board.push(move)
            move_sequence = " ".join(san_moves)

            eval_str = format_score(score) if score is not None else "N/A"
            lines.append(f"Line {i} (Depth {depth}): Eval {eval_str}")
            lines.append(f"  Moves: {move_sequence}")
            if len(pv) > 8:
                lines.append(f"  ... and {len(pv) - 8} more moves")
            lines.append("")

    return "\n".join(lines)


@click.command()
@click.argument("fen")
@click.option("--next-move", required=False)
@click.option("--depth", default=20, help="Analysis depth (default: 20)")
@click.option("--num-lines", default=5, help="Number of lines to analyze (default: 5)")
@click.option("--stockfish-path", default=STOCKFISH_PATH, help="Path to Stockfish binary")
def main(fen: str, next_move: str, depth: int, num_lines: int, stockfish_path: str):
    """Analyze a chess position given in FEN notation."""
    board = chess.Board(fen)

    if next_move:
        board.push_san(next_move)

    analysis_results = analyze_position(board, stockfish_path=stockfish_path, num_lines=num_lines, depth=depth)
    print(format_as_text(board, analysis_results))


if __name__ == "__main__":
    main()
