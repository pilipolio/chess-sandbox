"""Integration tests for Maia human move prediction."""

import pytest

from chess_sandbox.engine.maia import MaiaConfig, get_human_moves


@pytest.mark.integration
def test_maia_predicts_human_move():
    """Test that Maia predicts Qxb6 as top human move in puzzle position.

    Position where Qxb6 captures the queen (human-like) but allows checkmate,
    while Qxd8+ is objectively better. Maia should predict Qxb6 as most likely.
    """
    fen = "3r2k1/1p3ppp/pq1Q1b2/8/8/1P3N2/P4PPP/3R2K1 w - - 3 28"

    moves = get_human_moves(fen)

    assert len(moves) > 0
    assert moves[0].san_move == "Qxb6"
    assert moves[0].policy > 50  # Should be dominant move


@pytest.mark.integration
def test_maia_returns_all_legal_moves():
    """Test that Maia returns policy for multiple legal moves."""
    fen = "3r2k1/1p3ppp/pq1Q1b2/8/8/1P3N2/P4PPP/3R2K1 w - - 3 28"

    moves = get_human_moves(fen)

    # Should return policy for multiple moves
    assert len(moves) >= 10

    # All policies should be valid percentages
    for move in moves:
        assert 0 <= move.policy <= 100


@pytest.mark.integration
def test_maia_config_custom_weights():
    """Test MaiaConfig with custom weights path."""
    config = MaiaConfig.default()

    assert config.weights_path.endswith(".pb.gz")
    assert config.nodes == 1


@pytest.mark.integration
def test_maia_starting_position():
    """Test Maia on starting position returns sensible results."""
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    moves = get_human_moves(fen)

    assert len(moves) > 0
    # Top move should have non-trivial policy
    assert moves[0].policy > 1.0
    # Common opening moves should appear
    move_names = [m.san_move for m in moves]
    assert any(m in move_names for m in ["e4", "d4", "Nf3", "c4"])
