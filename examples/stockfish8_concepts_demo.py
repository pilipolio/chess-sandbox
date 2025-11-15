#!/usr/bin/env python3
"""Demo script for Stockfish 8 concept extraction.

This script demonstrates how to extract chess concepts from positions using
Stockfish 8's detailed evaluation breakdown.
"""

from chess_sandbox.concept_extraction.stockfish_concepts import (
    Stockfish8ConceptExtractor,
    Stockfish8Config,
)


def main() -> None:
    """Run concept extraction demo on various positions."""
    # Initialize with default config (uses settings or default path)
    config = Stockfish8Config()
    extractor = Stockfish8ConceptExtractor(config)

    # Test positions
    positions = {
        "Starting position": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "Italian Game": "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
        "Queen's Gambit Declined": "rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 0 4",
        "King safety test (castled)": "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 4 7",
        "Passed pawn position": "8/1p3k2/8/1P6/8/8/5K2/8 w - - 0 1",
    }

    print("=" * 80)
    print("Stockfish 8 Concept Extraction Demo")
    print("=" * 80)
    print()

    for name, fen in positions.items():
        print(f"\n{name}")
        print("-" * 80)
        print(f"FEN: {fen}")
        print()

        concepts = extractor.get_concepts(fen)

        # Display concept scores
        print("Concept scores (total advantage):")
        concept_dict = concepts.to_dict()
        for concept_name, score in concept_dict.items():
            if concept_name != "total_eval":
                print(f"  {concept_name:15s}: {score:+6.2f}")

        print(f"\nTotal evaluation: {concepts.total_eval:+6.2f}")
        print()

        # Highlight interesting concepts
        interesting = []
        if concepts.mobility and abs(concepts.mobility.total_advantage) > 0.5:
            interesting.append(
                f"  - Significant mobility {'advantage' if concepts.mobility.total_advantage > 0 else 'disadvantage'}"
            )
        if concepts.king_safety and abs(concepts.king_safety.total_advantage) > 0.5:
            interesting.append(
                f"  - King safety {'advantage' if concepts.king_safety.total_advantage > 0 else 'disadvantage'}"
            )
        if concepts.passed_pawns and abs(concepts.passed_pawns.total_advantage) > 0.3:
            interesting.append("  - Passed pawn advantage")

        if interesting:
            print("Notable concepts:")
            for item in interesting:
                print(item)

    print("\n" + "=" * 80)
    print("Demo complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
