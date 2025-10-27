#!/usr/bin/env python3
"""
Deterministic Chess Engine Analysis Module

Provides Stockfish-based position analysis and text formatting utilities.
"""

import os
from collections.abc import Sequence

import chess
import chess.engine
import click
from chess.engine import InfoDict


def get_stockfish_path() -> str:
    """Get Stockfish path from environment variable.
    TODO - replace with dotenv or pydantic settings

    Returns:
        Path to Stockfish binary

    Raises:
        RuntimeError: If STOCKFISH_PATH environment variable is not set
    """
    path = os.environ.get("STOCKFISH_PATH")
    if not path:
        raise RuntimeError(
            "STOCKFISH_PATH environment variable not set. " "Please set it to the path of your Stockfish binary."
        )
    return path


def format_score(score: chess.engine.PovScore) -> str:
    score_value = score.relative.score()
    if score_value is not None:
        return f"{score_value/100:+.2f}"
    return "0.00"


def analyze_position(
    board: chess.Board, stockfish_path: str | None = None, num_lines: int = 5, depth: int = 20
) -> Sequence[InfoDict]:
    if num_lines == 0:
        return []

    if stockfish_path is None:
        stockfish_path = get_stockfish_path()

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
@click.option("--stockfish-path", default=None, help="Path to Stockfish binary (defaults to STOCKFISH_PATH env var)")
def main(fen: str, next_move: str, depth: int, num_lines: int, stockfish_path: str | None):
    """Analyze a chess position given in FEN notation."""
    board = chess.Board(fen)

    if next_move:
        board.push_san(next_move)

    analysis_results = analyze_position(board, stockfish_path=stockfish_path, num_lines=num_lines, depth=depth)
    print(format_as_text(board, analysis_results))


if __name__ == "__main__":
    main()


# --- Tests ---


def test_analyze_starting_position():
    """Test that Stockfish can analyze the starting position.

    This is a happy path test that verifies:
    - Stockfish binary is accessible via STOCKFISH_PATH
    - Engine communication works
    - Analysis returns expected data structure
    """
    board = chess.Board()  # Starting position

    results = analyze_position(board, num_lines=3, depth=15)

    # Should return 3 lines of analysis
    assert len(results) == 3

    # Each result should have core fields
    for info in results:
        assert "score" in info
        assert "pv" in info  # Principal variation (move sequence)
        assert "depth" in info
        assert info["depth"] >= 15  # Should reach requested depth

        # PV should be non-empty
        assert len(info["pv"]) > 0
