#!/usr/bin/env python3
"""Demo script to show tactical pattern detection in action."""

import chess

from chess_sandbox.commentary.tactical_patterns import TacticalPatternDetector


def demo_pin_detection():
    """Demonstrate pin detection on a position with a pin."""
    print("=" * 70)
    print("DEMO: Pin Detection")
    print("=" * 70)
    print()

    # Position with knight on c6 pinned to king on e8 by bishop on b5
    fen = "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
    board = chess.Board(fen)

    print(f"Position FEN: {fen}")
    print()
    print(board)
    print()

    detector = TacticalPatternDetector(board)
    tactical_context = detector.get_tactical_context()

    print("Tactical Context Detected:")
    print(tactical_context if tactical_context else "No patterns detected")
    print()


def demo_fork_detection():
    """Demonstrate fork detection on a position with a fork."""
    print("=" * 70)
    print("DEMO: Fork Detection")
    print("=" * 70)
    print()

    # Knight on e5 forking king on e7 and rook on h8
    fen = "r3k2r/pppppppp/8/4N3/8/8/PPPPPPPP/R3K2R w KQkq - 0 1"
    board = chess.Board(fen)

    print(f"Position FEN: {fen}")
    print()
    print(board)
    print()

    detector = TacticalPatternDetector(board)
    tactical_context = detector.get_tactical_context()

    print("Tactical Context Detected:")
    print(tactical_context if tactical_context else "No patterns detected")
    print()


def demo_combined():
    """Demonstrate detection on a complex position."""
    print("=" * 70)
    print("DEMO: Complex Position")
    print("=" * 70)
    print()

    # More complex position that might have both pins and forks
    fen = "r1bqk2r/pppp1ppp/2n2n2/1Bb1p3/4P3/3P1N2/PPP2PPP/RNBQK2R b KQkq - 0 5"
    board = chess.Board(fen)

    print(f"Position FEN: {fen}")
    print()
    print(board)
    print()

    detector = TacticalPatternDetector(board)
    tactical_context = detector.get_tactical_context()

    print("Tactical Context Detected:")
    print(tactical_context if tactical_context else "No patterns detected")
    print()


if __name__ == "__main__":
    demo_pin_detection()
    demo_fork_detection()
    demo_combined()
