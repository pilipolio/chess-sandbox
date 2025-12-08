"""Pydantic AI agent for chess puzzle analysis.

Provides a chess teacher agent equipped with engine analysis tools
to generate thinking process for puzzles in PGN notation.

Example:
    export OPENROUTER_API_KEY=sk-...
    uv run python -c "
    from chess_sandbox.puzzles_trainer.puzzle_agent import analyze_puzzle, PuzzleInfo
    result = analyze_puzzle(PuzzleInfo(
        fen='5rk1/1p3ppp/pq1Q1b2/8/8/1P3N2/P4PPP/3R2K1 b - - 2 27',
        last_move='Qd6',
        theme='Mating threat',
        expected_line=['Rd8', 'Qxd8+', 'Bxd8']
    ))
    print(result.solution_pgn)
    "
"""

import os
from dataclasses import dataclass

import chess
import chess.engine
import click
import logfire
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from chess_sandbox.engine.analyse import (
    CandidateMove,
    EngineConfig,
    PrincipalVariation,
    analyze_moves,
    analyze_variations,
)

logfire.configure()
logfire.instrument_pydantic_ai()


class PuzzleInfo(BaseModel):
    """Context information for a chess puzzle."""

    fen: str = Field(description="Starting position in FEN notation")
    last_move: str | None = Field(default=None, description="The opponent's last move that set up the puzzle")
    theme: str = Field(description="Puzzle theme (e.g., 'Mating threat', 'fork', 'pin')")
    expected_line: list[str] = Field(description="Expected solution moves in SAN notation")


@dataclass
class PuzzleAnalysisDeps:
    """Dependencies for the puzzle analysis agent."""

    engine: chess.engine.SimpleEngine
    engine_config: EngineConfig
    board: chess.Board
    puzzle_info: PuzzleInfo


class PuzzleAnalysisOutput(BaseModel):
    """Structured output for puzzle analysis."""

    position_assessment: str = Field(description="1-2 sentence summary of key position features")
    solution_pgn: str = Field(description="Solution in PGN format with {comments} explaining each move")
    key_theme: str = Field(description="The main tactical/strategic theme demonstrated")


CHESS_TEACHER_SYSTEM_PROMPT = """\
You are a chess instructor explaining your thought process when puzzle solution.

PUZZLE:
- FEN: {fen}
- {side_to_move} to move (move {move_number})
- Theme: {theme}
- Solution: {expected_line} ({solution_length} moves total)

Board:
{ascii_board}

YOUR TASK:
Write the solution showing WHY each move is best by comparing to alternatives.

STRICT RULES:
1. Output ONLY the {solution_length} moves of the solution. STOP after the last move.
2. For natural alternatives, use the evaluate_moves_tool and show in parentheses why they fail.
3. Keep refutations SHORT (1-2 moves showing the problem).

USE TOOLS:
- Call evaluate_moves_tool with candidate moves to compare alternatives

OUTPUT FORMAT:
{move_prefix} [move] {{why it's best}}
  ([alt]? {{why it fails}} [refutation])
[opponent reply] {{forced because...}} 
[next solution move] {{explanation}}
...END after {solution_length} moves

EXAMPLE (3-move solution, White to move):
1. Re8+ {{forces king to corner}} (1. Qxb7? Qd1#) Rxe8 {{only legal}} 2. Qxe8# {{mate}}
"""


def create_openrouter_model(model_name: str = "openai/gpt-5-mini") -> OpenAIModel:
    """Create OpenRouter model for the agent."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is required")

    return OpenAIModel(
        model_name,
        provider=OpenAIProvider(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        ),
    )


puzzle_agent: Agent[PuzzleAnalysisDeps, PuzzleAnalysisOutput] = Agent(
    model=create_openrouter_model(),
    deps_type=PuzzleAnalysisDeps,
    output_type=PuzzleAnalysisOutput,
)


# @puzzle_agent.tool
def analyze_variations_tool(
    ctx: RunContext[PuzzleAnalysisDeps],
    num_lines: int = 3,
) -> dict[str, object]:
    """Get top engine lines from the current position.

    Args:
        num_lines: Number of principal variations to return (default: 3)

    Returns:
        Dictionary with side_to_move and list of variations.
        Score is from White's perspective (positive = good for White).
    """
    deps = ctx.deps
    side = "White" if deps.board.turn == chess.WHITE else "Black"
    variations: list[PrincipalVariation] = analyze_variations(
        board=deps.board,
        engine=deps.engine,
        num_lines=num_lines,
        limit=deps.engine_config.limit,
    )
    return {
        "side_to_move": side,
        "move_number": deps.board.fullmove_number,
        "variations": [
            {
                "score_white_perspective": pv.score,
                "line": pv.san_moves,
                "first_move": pv.san_moves[0] if pv.san_moves else None,
            }
            for pv in variations
        ],
    }


@puzzle_agent.tool
def evaluate_moves_tool(
    ctx: RunContext[PuzzleAnalysisDeps],
    moves_san: list[str],
) -> list[dict[str, object]]:
    """Evaluate specific candidate moves from the current position.

    Args:
        moves_san: List of moves in SAN notation to evaluate (e.g., ["Nf3", "e4", "Rd8"])

    Returns:
        List of moves with their evaluation scores (positive = good for white).
    """
    deps = ctx.deps
    parsed_moves: list[chess.Move] = []
    for san in moves_san:
        try:
            move = deps.board.parse_san(san)
            parsed_moves.append(move)
        except ValueError:
            continue

    if not parsed_moves:
        return []

    candidates: list[CandidateMove] = analyze_moves(
        board=deps.board,
        engine=deps.engine,
        moves=parsed_moves,
        limit=deps.engine_config.limit,
    )
    return [{"move": cm.san_move, "score": cm.score} for cm in candidates]


def analyze_puzzle(
    puzzle_info: PuzzleInfo,
    model: str = "openai/gpt-5-mini",
    depth: int = 20,
    num_lines: int = 5,
) -> PuzzleAnalysisOutput:
    """Analyze a chess puzzle using the pydantic-ai agent.

    Args:
        puzzle_info: Puzzle context (FEN, theme, expected solution)
        model: OpenRouter model to use
        depth: Engine analysis depth
        num_lines: Number of engine lines to consider

    Returns:
        Structured puzzle analysis with PGN commentary
    """
    engine_config = EngineConfig.stockfish(num_lines=num_lines, depth=depth)
    engine = engine_config.instantiate()
    board = chess.Board(puzzle_info.fen)

    deps = PuzzleAnalysisDeps(
        engine=engine,
        engine_config=engine_config,
        board=board,
        puzzle_info=puzzle_info,
    )

    agent = Agent(
        model=create_openrouter_model(model),
        deps_type=PuzzleAnalysisDeps,
        output_type=PuzzleAnalysisOutput,
    )

    agent._function_tools = puzzle_agent._function_tools  # pyright: ignore[reportPrivateUsage]

    try:
        side_to_move = "White" if board.turn == chess.WHITE else "Black"
        move_prefix = f"{board.fullmove_number}..." if board.turn == chess.BLACK else f"{board.fullmove_number}."
        user_prompt = CHESS_TEACHER_SYSTEM_PROMPT.format(
            fen=puzzle_info.fen,
            side_to_move=side_to_move,
            move_number=board.fullmove_number,
            move_prefix=move_prefix,
            last_move=puzzle_info.last_move or "N/A",
            theme=puzzle_info.theme,
            expected_line=" ".join(puzzle_info.expected_line),
            solution_length=len(puzzle_info.expected_line),
            ascii_board=str(board),
        )

        result = agent.run_sync(
            user_prompt=user_prompt,
            deps=deps,
        )
        return result.output  # type: ignore[reportUnknownReturn]
    finally:
        engine.quit()


# Sample puzzle for testing
SAMPLE_PUZZLE = PuzzleInfo(
    fen="5rk1/1p3ppp/pq1Q1b2/8/8/1P3N2/P4PPP/3R2K1 b - - 2 27",
    last_move="Qd6",
    theme="Mating threat",
    expected_line=["Rd8", "Qxd8+", "Bxd8"],
)


@click.command("puzzle-agent")
@click.option(
    "--fen",
    type=str,
    default=SAMPLE_PUZZLE.fen,
    help="Position in FEN notation",
)
@click.option(
    "--last-move",
    type=str,
    default=SAMPLE_PUZZLE.last_move,
    help="Opponent's last move",
)
@click.option(
    "--theme",
    type=str,
    default=SAMPLE_PUZZLE.theme,
    help="Puzzle theme",
)
@click.option(
    "--expected-line",
    type=str,
    default=" ".join(SAMPLE_PUZZLE.expected_line),
    help="Expected solution moves (space-separated SAN)",
)
@click.option(
    "--model",
    type=str,
    default="openai/gpt-5-mini",
    help="OpenRouter model to use",
)
@click.option(
    "--depth",
    type=int,
    default=20,
    help="Engine analysis depth",
)
def main(
    fen: str,
    last_move: str | None,
    theme: str,
    expected_line: str,
    model: str,
    depth: int,
) -> None:
    """Analyze a chess puzzle using the pydantic-ai agent.

    Uses engine analysis tools to generate a chess teacher's thought process
    in PGN notation with comments.

    Example:
        uv run puzzle-agent --fen "5rk1/1p3ppp/pq1Q1b2/8/8/1P3N2/P4PPP/3R2K1 b - - 2 27"
    """
    load_dotenv()

    puzzle_info = PuzzleInfo(
        fen=fen,
        last_move=last_move,
        theme=theme,
        expected_line=expected_line.split(),
    )

    click.echo(f"Analyzing puzzle: {fen}")
    click.echo(f"Theme: {theme}")
    click.echo(f"Expected line: {expected_line}")
    click.echo(f"Model: {model}")
    click.echo("-" * 60)

    board = chess.Board(fen)
    click.echo(f"\n{board}\n")

    result = analyze_puzzle(puzzle_info, model=model, depth=depth)

    click.echo("=" * 60)
    click.echo("POSITION ASSESSMENT:")
    click.echo(result.position_assessment)
    click.echo()
    click.echo("SOLUTION PGN:")
    click.echo(result.solution_pgn)
    click.echo()
    click.echo("KEY THEME:")
    click.echo(result.key_theme)
    click.echo("=" * 60)


if __name__ == "__main__":
    main()
