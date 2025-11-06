#!/usr/bin/env python3
"""Query production Modal endpoints for chess position analysis and concept extraction."""

import argparse
import json
import urllib.parse
import urllib.request

import chess
import chess.svg


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


def generate_svg(fen: str, analysis_result: dict, output_path: str) -> None:
    """Generate SVG visualization with arrows for top moves from analysis.

    Args:
        fen: Chess position in FEN notation
        analysis_result: Analysis result from query_position_analysis
        output_path: Path to save the SVG file
    """
    board = chess.Board(fen)

    # Rainbow spectrum colors for lines 1-5
    colors = [
        "#0000ffcc",  # Blue
        "#9933ffcc",  # Purple
        "#ff8800cc",  # Orange
        "#00cc00cc",  # Green
        "#ff0000cc",  # Red
    ]

    arrows = []

    # Extract arrows from analysis lines
    if "lines" in analysis_result:
        for idx, line in enumerate(analysis_result["lines"][:5]):  # Top 5 lines
            if "moves" in line and line["moves"]:
                # Parse the first move in UCI notation (e.g., "e2e4")
                first_move_uci = line["moves"][0]
                try:
                    move = chess.Move.from_uci(first_move_uci)
                    color = colors[idx % len(colors)]
                    arrows.append(chess.svg.Arrow(move.from_square, move.to_square, color=color))
                except (ValueError, IndexError):
                    # Skip invalid moves
                    continue

    # Generate SVG
    svg_content = chess.svg.board(board, arrows=arrows, size=400)

    # Write to file
    with open(output_path, "w") as f:
        f.write(svg_content)


def main():
    parser = argparse.ArgumentParser(description="Query production Modal endpoints for chess analysis")
    parser.add_argument("fen", help="Chess position in FEN notation")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Minimum confidence threshold for concepts (default: 0.1)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save SVG visualization with analysis arrows",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("CHESS POSITION ANALYSIS")
    print("=" * 70)
    print()

    analysis = None
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

    # Generate SVG if output path is provided
    if args.output and analysis:
        print()
        print("=" * 70)
        print("SVG VISUALIZATION")
        print("=" * 70)
        print()
        try:
            generate_svg(args.fen, analysis, args.output)
            print(f"SVG saved to: {args.output}")
        except Exception as e:
            print(f"Error generating SVG: {e}")


if __name__ == "__main__":
    main()
