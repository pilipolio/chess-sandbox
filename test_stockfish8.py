#!/usr/bin/env python3
"""Quick test of Stockfish 8 concept extraction."""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

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
