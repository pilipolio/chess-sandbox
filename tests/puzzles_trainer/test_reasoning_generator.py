"""Tests for reasoning trace generation."""

import chess

from chess_sandbox.puzzles_trainer.reasoning_generator import (
    RefutedLine,
    build_lines_exploration_pgn,
)
from chess_sandbox.puzzles_trainer.reasoning_verifier import validate_pgn_lines


class TestBuildLinesExplorationPgn:
    """Tests for PGN generation with solution and refuted lines."""

    def test_solution_only_white_to_move(self):
        """Test PGN generation with only solution moves, White to move."""
        fen = "r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2b1/PqP3PP/7K w - - 0 25"
        board = chess.Board(fen)
        solution = ["Rxe7", "Qb1+", "Nc1", "Qxc1+", "Qxc1"]

        pgn = build_lines_exploration_pgn(board, solution)

        assert "25. Rxe7!" in pgn
        assert "Qb1+" in pgn
        assert "26. Nc1" in pgn

        # Validate the generated PGN
        valid, illegal = validate_pgn_lines(fen, pgn)
        assert valid, f"Generated PGN is invalid: {illegal}"

    def test_solution_only_black_to_move(self):
        """Test PGN generation with only solution moves, Black to move."""
        fen = "8/5R2/1p2P3/p4r2/P6p/1P3Pk1/4K3/8 b - - 2 64"
        board = chess.Board(fen)
        solution = ["Re5+", "Kf1", "Rxe6"]

        pgn = build_lines_exploration_pgn(board, solution)

        assert "64... Re5+!" in pgn
        assert "65. Kf1" in pgn
        assert "Rxe6" in pgn

        # Validate the generated PGN
        valid, illegal = validate_pgn_lines(fen, pgn)
        assert valid, f"Generated PGN is invalid: {illegal}"

    def test_with_refuted_lines_white_to_move(self):
        """Test PGN generation with refuted alternatives, White to move."""
        fen = "r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2b1/PqP3PP/7K w - - 0 25"
        board = chess.Board(fen)
        solution = ["Rxe7", "Qb1+", "Nc1"]

        refuted = [
            RefutedLine(
                human_move="Rxf6",
                human_policy=30.0,
                refutation_line=["Re1+", "Rf1"],
                score=None,
                mate_in=2,
                explanation="leads to forced mate",
            ),
            RefutedLine(
                human_move="hxg3",
                human_policy=20.0,
                refutation_line=["Rxe6", "Qh4"],
                score=-300.0,
                mate_in=None,
                explanation="loses significant material",
            ),
        ]

        pgn = build_lines_exploration_pgn(board, solution, refuted)

        # Main line present
        assert "25. Rxe7!" in pgn
        # Variations present with ? annotation
        assert "(25. Rxf6?" in pgn
        assert "(25. hxg3?" in pgn
        # Explanations in braces
        assert "{leads to forced mate}" in pgn
        assert "{loses significant material}" in pgn

        # Validate the generated PGN
        valid, illegal = validate_pgn_lines(fen, pgn)
        assert valid, f"Generated PGN is invalid: {illegal}"

    def test_with_refuted_lines_black_to_move(self):
        """Test PGN generation with refuted alternatives, Black to move.

        This is a regression test for a bug where variations were placed at the end
        of the PGN (after the main line), causing the validator to check them against
        the wrong position.
        """
        fen = "8/5R2/1p2P3/p4r2/P6p/1P3Pk1/4K3/8 b - - 2 64"
        board = chess.Board(fen)
        solution = ["Re5+", "Kf1", "Rxe6"]

        refuted = [
            RefutedLine(
                human_move="Rxf3",
                human_policy=30.0,
                refutation_line=["Rxf3+", "Kg4"],
                score=None,
                mate_in=2,
                explanation="leads to forced mate",
            ),
        ]

        pgn = build_lines_exploration_pgn(board, solution, refuted)

        # Main line present
        assert "64... Re5+!" in pgn
        # Variation present with correct move number prefix for Black
        assert "(64... Rxf3?" in pgn
        # Explanation in braces
        assert "{leads to forced mate}" in pgn

        # CRITICAL: Validate the generated PGN
        # This was failing before the fix because the variation was placed
        # after the main line, so the validator tried to check Rxf3 from
        # the position after Re5+ Kf1 Rxe6 instead of the starting position
        valid, illegal = validate_pgn_lines(fen, pgn)
        assert valid, f"Generated PGN is invalid: {illegal}"

    def test_variation_placement_immediately_after_first_move(self):
        """Verify variations are placed right after the first move, not at the end."""
        fen = "8/5R2/1p2P3/p4r2/P6p/1P3Pk1/4K3/8 b - - 2 64"
        board = chess.Board(fen)
        solution = ["Re5+", "Kf1", "Rxe6"]

        refuted = [
            RefutedLine(
                human_move="Rxf3",
                human_policy=30.0,
                refutation_line=["Rxf3+", "Kg4"],
                score=None,
                mate_in=2,
                explanation="leads to forced mate",
            ),
        ]

        pgn = build_lines_exploration_pgn(board, solution, refuted)

        # The variation should come BEFORE the continuation of the main line
        # Structure should be: first_move (variation) rest_of_main_line
        re5_pos = pgn.find("Re5+!")
        variation_pos = pgn.find("(64... Rxf3?")
        kf1_pos = pgn.find("Kf1")

        assert re5_pos < variation_pos < kf1_pos, (
            f"Variation should be between first move and continuation. "
            f"Got: Re5+ at {re5_pos}, variation at {variation_pos}, Kf1 at {kf1_pos}"
        )

    def test_empty_solution(self):
        """Test handling of empty solution."""
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        board = chess.Board(fen)

        pgn = build_lines_exploration_pgn(board, [])

        assert pgn == ""

    def test_single_move_solution(self):
        """Test solution with only one move."""
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        board = chess.Board(fen)
        solution = ["e4"]

        pgn = build_lines_exploration_pgn(board, solution)

        assert "1. e4!" in pgn
        valid, illegal = validate_pgn_lines(fen, pgn)
        assert valid, f"Generated PGN is invalid: {illegal}"
