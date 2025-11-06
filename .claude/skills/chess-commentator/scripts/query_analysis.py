#!/usr/bin/env python3
"""Query production Modal endpoints for chess position analysis and concept extraction."""

import argparse
import json
import urllib.parse
import urllib.request


def query_position_analysis(fen: str, depth: int = 20, num_lines: int = 5) -> dict:
    """Query the chess position analysis endpoint."""
    base_url = "https://pilipolio--chess-analysis-analyze.modal.run"
    params = {"fen": fen, "depth": depth, "num_lines": num_lines}
    url = f"{base_url}?{urllib.parse.urlencode(params)}"

    with urllib.request.urlopen(url) as response:
        return json.loads(response.read().decode())


def query_concept_extraction(fen: str, threshold: float = 0.1) -> dict:
    """Query the concept extraction endpoint."""
    base_url = "https://pilipolio--chess-concept-extraction-extract-concepts.modal.run"
    params = {"fen": fen, "threshold": threshold}
    url = f"{base_url}?{urllib.parse.urlencode(params)}"

    with urllib.request.urlopen(url) as response:
        return json.loads(response.read().decode())


def main():
    parser = argparse.ArgumentParser(description="Query production Modal endpoints for chess analysis")
    parser.add_argument("fen", help="Chess position in FEN notation")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Minimum confidence threshold for concepts (default: 0.1)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("CHESS POSITION ANALYSIS")
    print("=" * 70)
    print()

    try:
        analysis = query_position_analysis(args.fen)
        print(json.dumps(analysis, indent=2))
    except Exception as e:
        print(f"Error querying position analysis: {e}")

    print()
    print("=" * 70)
    print("CONCEPT EXTRACTION")
    print("=" * 70)
    print()

    try:
        concepts = query_concept_extraction(args.fen, args.threshold)
        print(json.dumps(concepts, indent=2))
    except Exception as e:
        print(f"Error querying concept extraction: {e}")


if __name__ == "__main__":
    main()
