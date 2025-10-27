"""Integration tests requiring Stockfish engine."""

import chess
import pytest

from chess_sandbox.engine_analysis import analyze_position


@pytest.mark.integration
def test_analyze_starting_position():
    """Test analysis of starting chess position with Stockfish."""
    board = chess.Board()  # Starting position
    results = analyze_position(board, num_lines=1, depth=1)

    assert len(results) == 1
    assert results[0].score == 0.17
    assert results[0].san_moves == ["e4"]


@pytest.mark.integration
def test_analyze_position_multiple_lines():
    """Test analysis with multiple principal variations."""
    board = chess.Board()
    results = analyze_position(board, num_lines=3, depth=1)

    assert len(results) == 3
    # All results should have scores and moves
    for pv in results:
        assert pv.score is not None
        assert len(pv.san_moves) > 0


@pytest.mark.integration
def test_analyze_position_zero_lines():
    """Test that num_lines=0 returns empty list."""
    board = chess.Board()
    results = analyze_position(board, num_lines=0, depth=1)

    assert results == []
