#!/usr/bin/env python3
"""Query production Modal endpoints for chess position analysis and concept extraction."""

import argparse
import json
import logging
import sys
import urllib.parse
import urllib.request

import chess
import chess.svg

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


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

    svg_content = chess.svg.board(board, arrows=[], size=400)
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

    logger.info("=" * 70)
    logger.info("CHESS POSITION ANALYSIS")
    logger.info("=" * 70)
    logger.info("")

    analysis = None
    try:
        analysis = query_position_analysis(args.fen)
        logger.info(json.dumps(analysis, indent=2))
    except Exception as e:
        logger.info(f"Error querying position analysis: {e}")

    logger.info("")
    logger.info("=" * 70)
    logger.info("CONCEPT EXTRACTION")
    logger.info("=" * 70)
    logger.info("")

    try:
        concepts = query_concept_extraction(args.fen, args.threshold)
        logger.info(json.dumps(concepts, indent=2))
    except Exception as e:
        logger.info(f"Error querying concept extraction: {e}")

    if args.output and analysis:
        logger.info("")
        logger.info("=" * 70)
        logger.info("SVG VISUALIZATION")
        logger.info("=" * 70)
        logger.info("")
        try:
            generate_svg(args.fen, analysis, args.output)
            logger.info(f"SVG saved to: {args.output}")
        except Exception as e:
            logger.info(f"Error generating SVG: {e}")


if __name__ == "__main__":
    main()
