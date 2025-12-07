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


CHESS_TEACHER_SYSTEM_PROMPT = """You are an experienced chess instructor analyzing a puzzle position.

Your task is to explain the puzzle solution as if teaching a student.

PUZZLE CONTEXT:
- Position (FEN): {fen}
- Side to move: {side_to_move}
- Move number: {move_number}
- Last move played by opponent: {last_move}
- Puzzle theme: {theme}
- Expected solution: {expected_line}

Board:
{ascii_board}

ANALYSIS TOOLS:
You have access to engine analysis tools. Use them to verify your analysis:
1. analyze_variations_tool: Get top engine lines from the current position
2. evaluate_moves_tool: Compare specific candidate moves

IMPORTANT: Start by calling analyze_variations_tool to see the engine's top lines.

OUTPUT FORMAT:
Generate your analysis in PGN notation with comments in curly brackets:
- Use SAN notation for moves (e.g., Nf3, exd5, O-O)
- Add brief comments in {{curly brackets}} after moves
- Focus on the "why" - explain threats, tactics, and strategic ideas
- Use proper move numbers (currently move {move_number})

Example output for Black to move in a deflection puzzle:
27... Rd8 {{deflecting the queen from defense}} 28. Qxd8+ {{forced capture}} Bxd8

TEACHING APPROACH:
1. First use analyze_variations_tool to see the best lines
2. Explain why the puzzle move is best (what it threatens/accomplishes)
3. Show why the opponent's responses are forced
4. Connect the solution to the puzzle theme
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


@puzzle_agent.system_prompt
def build_system_prompt(ctx: RunContext[PuzzleAnalysisDeps]) -> str:
    """Build dynamic system prompt with puzzle context."""
    puzzle = ctx.deps.puzzle_info
    board = ctx.deps.board
    side_to_move = "White" if board.turn == chess.WHITE else "Black"
    return CHESS_TEACHER_SYSTEM_PROMPT.format(
        fen=puzzle.fen,
        side_to_move=side_to_move,
        move_number=board.fullmove_number,
        last_move=puzzle.last_move or "N/A",
        theme=puzzle.theme,
        expected_line=" ".join(puzzle.expected_line),
        ascii_board=str(board),
    )


@puzzle_agent.tool
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

    # Copy tool and system prompt registrations
    agent._system_prompts = puzzle_agent._system_prompts  # pyright: ignore[reportPrivateUsage]
    agent._function_tools = puzzle_agent._function_tools  # pyright: ignore[reportPrivateUsage]

    try:
        result = agent.run_sync(
            "First call analyze_variations_tool to see the best moves, then explain the solution.",
            deps=deps,
        )
        return result.output
    finally:
        engine.quit()
