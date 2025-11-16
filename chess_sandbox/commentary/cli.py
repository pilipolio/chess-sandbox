"""
CLI interface for chess position commentary.

This module provides the Click command-line interface for analyzing chess positions
and moves. It composes the context building and LLM modules.
"""

import click

from ..concept_extraction.model.inference import ConceptExtractor
from ..engine.analyse import EngineConfig
from ..lichess import get_analysis_url
from .context import PositionContextBuilder
from .llm import ChessPositionExplanationWithInput, explain_move


def print_explanation(explanation: ChessPositionExplanationWithInput):
    """Print a chess position explanation in a formatted way.

    Args:
        explanation: Position explanation to print
    """
    print("=" * 70)
    print("LLM Input")
    print("=" * 70)
    print()
    print(explanation.full_input)
    print()

    print("=" * 70)
    print("Commentary")
    print("=" * 70)
    print()

    print(explanation.model_dump_json(indent=2, exclude={"fen", "full_input"}))
    print("=" * 70)


@click.command()
@click.argument("fen")
@click.argument("move")
@click.option("--depth", default=20, help="Stockfish analysis depth (default: 20)")
@click.option("--num-lines", default=5, help="Number of engine lines to analyze (default: 5)")
@click.option("--model", default="gpt-5-mini", help="LLM model to use (default: gpt-5-mini)")
@click.option(
    "--reasoning-effort",
    type=click.Choice(["low", "medium", "high"], case_sensitive=False),
    help="Reasoning effort for models that support it",
)
def main(fen: str, move: str, depth: int, num_lines: int, model: str, reasoning_effort: str | None):
    """Analyze a chess move by comparing pre-move and post-move positions.

    FEN: Position in FEN notation (quote if it contains spaces)
    MOVE: Move to analyze in SAN notation (e.g., 'Nf3', 'exd5')
    """
    print(f"\nLichess analysis: {get_analysis_url(fen)}")

    print("=" * 70)
    print("MOVE ANALYSIS")
    print("=" * 70)
    print(f"\nPosition: {fen}")
    print(f"Move: {move}")
    print(f"Engine: Stockfish (depth={depth}, lines={num_lines})")
    print(f"LLM: {model}" + (f" (reasoning={reasoning_effort})" if reasoning_effort else ""))
    print()

    # Create context builder
    engine_config = EngineConfig.stockfish(num_lines=num_lines, depth=depth)
    concept_extractor = ConceptExtractor.from_hf(
        probe_repo_id="pilipolio/chess-positions-extractor",
        revision="production",
    )
    context_builder = PositionContextBuilder(engine_config, concept_extractor)

    # Build move context
    move_ctx = context_builder.build_move_context(fen, move)

    # Analyze with LLM
    move_explanation = explain_move(
        move_ctx=move_ctx,
        model=model,
        reasoning_effort=reasoning_effort,
    )

    # Print results
    print(move_explanation.model_dump_json(indent=2))
    print("=" * 70)


if __name__ == "__main__":
    main()
