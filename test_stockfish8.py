#!/usr/bin/env python3
"""Quick manual test of Stockfish 8 concept extraction.

Note: This is a manual test script, not a pytest test.
Run directly with: python3 test_stockfish8.py
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))


def main() -> None:
    """Run manual test of Stockfish 8 concept extraction."""
    from chess_sandbox.concept_extraction.stockfish_concepts import (
        Stockfish8ConceptExtractor,
        Stockfish8Config,
    )

    # Initialize with default config
    config = Stockfish8Config()
    extractor = Stockfish8ConceptExtractor(config)

    # Test position: Italian Game
    fen = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"

    print("Testing Stockfish 8 concept extraction")
    print(f"FEN: {fen}")
    print()

    concepts = extractor.get_concepts(fen)

    print("Concept scores:")
    for name, score in concepts.to_dict().items():
        print(f"  {name:15s}: {score:+6.2f}")


if __name__ == "__main__":
    main()
