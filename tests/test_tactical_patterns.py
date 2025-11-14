"""Tests for tactical pattern detection."""

import chess

from chess_sandbox.commentary.tactical_patterns import (
    TacticalPatternDetector,
)


class TestPinDetection:
    """Tests for pin detection."""

    def test_absolute_pin_to_king(self):
        """Test detection of absolute pin to king."""
        # Bishop on c1 pinning knight on d2 to king on e3
        board = chess.Board("8/8/8/8/8/4k3/3n4/2B1K3 b - - 0 1")
        detector = TacticalPatternDetector(board)
        pins = detector.detect_pins()

        # Black to move, so we detect black's pinned pieces
        assert len(pins) == 1
        pin = pins[0]
        assert chess.square_name(pin.pinned_square) == "d2"
        assert pin.pinned_piece.piece_type == chess.KNIGHT
        assert chess.square_name(pin.pinner_square) == "c1"
        assert pin.king_square is not None
        assert chess.square_name(pin.king_square) == "e3"

    def test_no_pins_in_starting_position(self):
        """Test that no pins are detected in starting position."""
        board = chess.Board()
        detector = TacticalPatternDetector(board)
        pins = detector.detect_pins()
        assert len(pins) == 0

    def test_rook_pin(self):
        """Test detection of rook pin."""
        # Rook on d1 pinning knight on d3 to king on d8
        board = chess.Board("3k4/8/8/8/8/3n4/8/3R1K2 b - - 0 1")
        detector = TacticalPatternDetector(board)
        pins = detector.detect_pins()

        # This position should have rook pinning d3 knight to d8 king
        assert len(pins) == 1
        pin = pins[0]
        assert pin.pinned_piece.piece_type == chess.KNIGHT
        assert chess.square_name(pin.pinned_square) == "d3"
        assert chess.square_name(pin.pinner_square) == "d1"

    def test_pin_description(self):
        """Test that pin description is human-readable."""
        board = chess.Board("8/8/8/8/8/4k3/3n4/2B1K3 b - - 0 1")
        detector = TacticalPatternDetector(board)
        pins = detector.detect_pins()

        assert len(pins) == 1
        description = pins[0].describe()
        assert "knight" in description.lower()
        assert "d2" in description.lower()
        assert "pinned" in description.lower()

    def test_real_game_position_with_diagonal_pin(self):
        """Test detection in a real game position with diagonal pins.

        Position has two pins by the same bishop on d4:
        1. f2 pawn pinned to g1 king (absolute pin)
        2. Nc3 knight pinned to a1 rook (relative pin)
        """
        board = chess.Board("r2qk2r/p1p2p2/p2p1n1p/3Pp1p1/1P1bP3/P1N2QBP/2P2PP1/R4RK1 w kq - 2 15")
        detector = TacticalPatternDetector(board)
        pins = detector.detect_pins()

        # White to move, should detect both pinned pieces
        assert len(pins) == 2

        # Find the pawn pin and knight pin (order may vary)
        pawn_pin = next((p for p in pins if p.pinned_piece.piece_type == chess.PAWN), None)
        knight_pin = next((p for p in pins if p.pinned_piece.piece_type == chess.KNIGHT), None)

        # Check f2 pawn pinned to king
        assert pawn_pin is not None
        assert chess.square_name(pawn_pin.pinned_square) == "f2"
        assert chess.square_name(pawn_pin.pinner_square) == "d4"
        assert pawn_pin.pinner_piece.piece_type == chess.BISHOP
        assert pawn_pin.king_square is not None
        assert chess.square_name(pawn_pin.king_square) == "g1"

        # Check Nc3 knight pinned to a1 rook
        assert knight_pin is not None
        assert chess.square_name(knight_pin.pinned_square) == "c3"
        assert chess.square_name(knight_pin.pinner_square) == "d4"
        assert knight_pin.pinner_piece.piece_type == chess.BISHOP
        assert knight_pin.valuable_square is not None
        assert chess.square_name(knight_pin.valuable_square) == "a1"

    def test_relative_pin_to_bishop_on_file(self):
        """Ensure pins to bishops along a file are flagged."""
        # Rook on c8 pins pawn on c2 to bishop on c1
        board = chess.Board("2r4k/8/8/8/8/8/2P5/2B2K2 w - - 0 1")
        detector = TacticalPatternDetector(board)
        pins = detector.detect_pins()

        assert len(pins) == 1
        pin = pins[0]
        assert chess.square_name(pin.pinned_square) == "c2"
        assert pin.pinner_piece.piece_type == chess.ROOK
        assert pin.valuable_square is not None
        assert chess.square_name(pin.valuable_square) == "c1"

    def test_relative_pin_to_knight_on_rank(self):
        """Ensure pins to knights along a rank are flagged."""
        # Rook on a1 pins bishop on d1 to knight on h1
        board = chess.Board("4k3/8/8/8/8/8/6K1/r2B3N w - - 0 1")
        detector = TacticalPatternDetector(board)
        pins = detector.detect_pins()

        assert len(pins) == 1
        pin = pins[0]
        assert chess.square_name(pin.pinned_square) == "d1"
        assert pin.pinner_piece.piece_type == chess.ROOK
        assert pin.valuable_square is not None
        assert chess.square_name(pin.valuable_square) == "h1"

    def test_pin_with_multiple_attackers_on_ray(self):
        """Test correct pinner detection when multiple enemy sliders are on the pin ray.

        Regression test: with multiple enemy sliders on the same ray, we must select
        the closest one to the pinned piece (not the first in global iteration order).

        Position: King e8, Bishop e7 (white), Rook e5 (black), Rook e1 (black)
        The bishop is pinned by the rook on e5 (not e1).
        """
        board = chess.Board("4K3/4B3/8/4r3/8/8/8/4r3 w - - 0 1")
        detector = TacticalPatternDetector(board)
        pins = detector.detect_pins()

        # Should detect one pin
        assert len(pins) == 1
        pin = pins[0]

        # Verify it's the bishop on e7
        assert chess.square_name(pin.pinned_square) == "e7"
        assert pin.pinned_piece.piece_type == chess.BISHOP

        # Verify the pinner is the closer rook on e5, not e1
        assert chess.square_name(pin.pinner_square) == "e5"
        assert pin.pinner_piece.piece_type == chess.ROOK

        # It's pinned to the king
        assert pin.king_square is not None
        assert chess.square_name(pin.king_square) == "e8"


class TestForkDetection:
    """Tests for fork detection."""

    def test_knight_fork(self):
        """Test detection of knight fork."""
        # Knight on e5 forking rooks on d7 and f7
        board = chess.Board("8/3r1r2/8/4N3/8/8/8/4K3 w - - 0 1")
        detector = TacticalPatternDetector(board)
        forks = detector.detect_forks()

        # Should detect knight forking two rooks (value 10)
        assert len(forks) >= 1
        # Find the fork with the knight
        knight_forks = [f for f in forks if f.forking_piece.piece_type == chess.KNIGHT]
        assert len(knight_forks) >= 1

    def test_no_forks_in_starting_position(self):
        """Test that no significant forks are detected in starting position."""
        board = chess.Board()
        detector = TacticalPatternDetector(board)
        forks = detector.detect_forks()

        # Starting position shouldn't have valuable forks
        assert len(forks) == 0

    def test_fork_description(self):
        """Test that fork description is human-readable."""
        board = chess.Board("r3k2r/pppppppp/8/4N3/8/8/PPPPPPPP/R3K2R w KQkq - 0 1")
        detector = TacticalPatternDetector(board)
        forks = detector.detect_forks()

        if forks:
            description = forks[0].describe()
            assert "attacks" in description.lower()

    def test_queen_fork(self):
        """Test detection of queen fork."""
        # Queen on d5 forking rooks on a5 and h5
        board = chess.Board("8/8/8/r2Q3r/8/8/8/4K3 w - - 0 1")
        detector = TacticalPatternDetector(board)
        forks = detector.detect_forks()

        # Queen should be able to fork two rooks
        queen_forks = [f for f in forks if f.forking_piece.piece_type == chess.QUEEN]
        assert len(queen_forks) >= 1


class TestTacticalContext:
    """Tests for overall tactical context generation."""

    def test_tactical_context_with_pins(self):
        """Test tactical context includes pin information."""
        board = chess.Board("8/8/8/8/8/4k3/3n4/2B1K3 b - - 0 1")
        detector = TacticalPatternDetector(board)
        context = detector.get_tactical_context()

        assert "PINS DETECTED:" in context
        assert len(context) > 0

    def test_tactical_context_empty_when_no_patterns(self):
        """Test tactical context is empty when no patterns detected."""
        board = chess.Board()
        detector = TacticalPatternDetector(board)
        context = detector.get_tactical_context()

        assert context == ""

    def test_tactical_context_with_multiple_patterns(self):
        """Test tactical context with both pins and forks."""
        # This is a complex position that might have both
        board = chess.Board("r1bqk2r/pppp1ppp/2n2n2/1Bb1p3/4P3/3P1N2/PPP2PPP/RNBQK2R b KQkq - 0 5")
        detector = TacticalPatternDetector(board)
        context = detector.get_tactical_context()

        # Just verify we can generate context without errors
        assert isinstance(context, str)
