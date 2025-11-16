"""
LLM workflow module for chess position analysis.

This module manages prompts, response parsing, and OpenAI client configuration.
It consumes PositionState and MoveContext data classes from the context module.
"""

from textwrap import dedent

import chess
from openai import OpenAI
from openai.types.shared.reasoning_effort import ReasoningEffort
from openai.types.shared_params.reasoning import Reasoning
from pydantic import BaseModel, Field

from ..engine.analyse import EngineConfig, analyze_variations
from ..engine.position_analysis import PositionAnalysis
from .context import MoveContext
from .tactical_patterns import TacticalPatternDetector


class ChessPositionExplanation(BaseModel):
    """LLM structured output for position analysis."""

    assessment: int = Field(description="-1 (Black advantage), 0 (Equal), +1 (White advantage)")
    best_move: str = Field(description="Best move in algebraic notation")
    themes: list[str] = Field(description="Key strategic or tactical themes (keywords)")
    variations: str = Field(description="Key lines and variations in PGN format with comments")


class ChessPositionExplanationWithInput(ChessPositionExplanation):
    """Position analysis with input metadata."""

    fen: str = Field(description="The input position in FEN format")
    full_input: str = Field(description="The full input position ")


class PositionDiffThoughts(BaseModel):
    """Differences between pre-move and post-move positions."""

    concepts_gained: list[str] = Field(description="New concepts appearing after the move")
    concepts_lost: list[str] = Field(description="Concepts no longer present after the move")
    evaluation_change_category: str = Field(
        description="Human-readable evaluation change (significant improvement/minor improvement/roughly equal/etc.)"
    )
    tactical_changes: str = Field(description="Description of changes in tactical patterns")


class MoveExplanation(BaseModel):
    """Complete analysis of a chess move comparing pre and post-move positions."""

    move_san: str = Field(description="The move played in SAN notation")
    differences: PositionDiffThoughts = Field(description="What changed between positions")
    move_comment: str = Field(
        description="Concise 1-2 sentence explanation of why this move was played and how it changed the position"
    )


POSITION_PROMPT = dedent("""
    You are a chess grandmaster analyzing a position. Below is a chess position:

    {analysis_text}

    {tactical_context}

    Please provide a structured analysis with:
    1. **assessment**: Overall position evaluation as -1 (Black is better), 0 (Equal), or +1 (White is better)
    2. **best_move**: The best move in standard algebraic notation (e.g., "Nf3", "exd5")
    3. **themes**: List of key strategic or tactical themes
       (keywords like "fork", "pin", "weak king", "passed pawn", etc.)
    4. **variations**: Key lines and variations in standard PGN format with comments and annotations
       Example format: 1. e4 {{sharp opening}} 1... c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 {{Sicilian main line}}
       (4. Qxd4?! {{premature queen development}} 4... Nc6)

    Keep your analysis concise but insightful.
""").strip()

MOVE_PROMPT = dedent("""
    You are a chess grandmaster analyzing a move. Compare the position before and after the move,
    and provides a concise explanation of the move.

    MOVE PLAYED: {move_san}

    PRE-MOVE POSITION:
    - Position evaluation: {pre_eval_category}
    - Concepts: {pre_concepts}
    - Tactical patterns: {pre_tactics}
    - Best moves: {pre_best_moves}

    POST-MOVE POSITION:
    - Position evaluation: {post_eval_category}
    - Concepts: {post_concepts}
    - Tactical patterns: {post_tactics}
    - Best moves: {post_best_moves}

    EVALUATION CHANGE: {eval_change_category}

    IMPORTANT: Only report MEANINGFUL changes. Ignore minor fluctuations and don't mention when things
    (evaluation, concepts, tactical patterns) stayed the same.

    1. **concepts_gained**: List ONLY strategically significant concepts that appeared
       (empty list if no meaningful concepts were gained)

    2. **concepts_lost**: List ONLY strategically significant concepts that disappeared
       (empty list if no meaningful concepts were lost)

    3. **evaluation_change_category**: The provided evaluation change category
       ({eval_change_category})

    4. **tactical_changes**: Describe ONLY significant tactical shifts:
       - New tactical threats or opportunities created
       - Important tactical vulnerabilities removed
       - Leave empty string "" if no significant tactical changes

    5. **move_comment**: One concise sentence explaining the move's purpose:
       - If the move creates/addresses tactical threats, mention that
       - If the move improves strategic position (concepts), mention that
       - Reference the evaluation change if significant (improvement/worsening)
       - Be specific about what changed, not what stayed the same

    Be selective. Only highlight changes that actually matter to understanding the move.
""").strip()


def summarize_position(
    board: chess.Board,
    engine_config: EngineConfig,
    client: OpenAI,
    model: str = "gpt-4o-2024-08-06",
    reasoning_effort: ReasoningEffort | None = None,
) -> ChessPositionExplanationWithInput:
    """Analyze a chess position using Stockfish and LLM.

    Args:
        board: Chess position to analyze
        engine_config: Configuration for Stockfish engine
        client: OpenAI client
        model: LLM model to use
        reasoning_effort: Optional reasoning effort for models that support it

    Returns:
        ChessPositionExplanationWithInput with complete analysis
    """
    engine = engine_config.instantiate()

    try:
        analysis_results = analyze_variations(board, engine, engine_config.num_lines, engine_config.limit)
        position_analysis = PositionAnalysis(
            fen=board.fen(), next_move=None, principal_variations=analysis_results, human_moves=None
        )
        analysis_text = position_analysis.format_as_text()

        # Detect tactical patterns using python-chess API
        detector = TacticalPatternDetector(board)
        tactical_context = detector.get_tactical_context()

        # Add tactical context section if patterns detected
        if tactical_context:
            tactical_context = f"\n{tactical_context}\n"
        else:
            tactical_context = ""

        prompt = POSITION_PROMPT.format(analysis_text=analysis_text, tactical_context=tactical_context)

        response = client.responses.parse(
            model=model,
            input=prompt,
            text_format=ChessPositionExplanation,
            reasoning=Reasoning(effort=reasoning_effort),
        )

        # When using reasoning models, the first item is a ReasoningItem, followed by the message
        message = next((item for item in response.output if item.type == "message"), None)  # type: ignore[attr-defined]

        if not message:
            raise Exception("No message found in response output")

        text = message.content[0]  # type: ignore[attr-defined]
        assert text.type == "output_text", "Unexpected content type"  # type: ignore[attr-defined]

        if not text.parsed:  # type: ignore[attr-defined]
            raise Exception("Could not parse LLM response into CommentaryOutput")

        parsed_data: ChessPositionExplanation = text.parsed  # type: ignore[attr-defined]
        return ChessPositionExplanationWithInput(fen=board.fen(), full_input=analysis_text, **parsed_data.model_dump())
    finally:
        engine.quit()


def explain_move(
    move_ctx: MoveContext,
    client: OpenAI,
    model: str = "gpt-4o-2024-08-06",
    reasoning_effort: ReasoningEffort | None = None,
) -> MoveExplanation:
    """Analyze a chess move using pre/post position context and LLM.

    Args:
        move_ctx: MoveContext with pre/post position states
        client: OpenAI client
        model: LLM model to use
        reasoning_effort: Optional reasoning effort for models that support it

    Returns:
        MoveExplanation with complete move analysis
    """
    # Format pre/post states for prompt
    pre_dict = move_ctx.pre_move_state.to_prompt_dict()
    post_dict = move_ctx.post_move_state.to_prompt_dict()

    # Create prompt with all context
    prompt = MOVE_PROMPT.format(
        move_san=move_ctx.move_san,
        pre_eval_category=pre_dict["eval_category"],
        pre_concepts=pre_dict["concepts"],
        pre_tactics=pre_dict["tactics"],
        pre_best_moves=pre_dict["best_moves"],
        post_eval_category=post_dict["eval_category"],
        post_concepts=post_dict["concepts"],
        post_tactics=post_dict["tactics"],
        post_best_moves=post_dict["best_moves"],
        eval_change_category=move_ctx.evaluation_change_category,
    )

    # Call LLM with structured output
    response = client.responses.parse(
        model=model,
        input=prompt,
        text_format=MoveExplanation,
        reasoning=Reasoning(effort=reasoning_effort),
    )

    # Extract parsed response
    message = next((item for item in response.output if item.type == "message"), None)  # type: ignore[attr-defined]

    if not message:
        raise Exception("No message found in response output")

    text = message.content[0]  # type: ignore[attr-defined]
    assert text.type == "output_text", "Unexpected content type"  # type: ignore[attr-defined]

    if not text.parsed:  # type: ignore[attr-defined]
        raise Exception("Could not parse LLM response into PositionDiff")

    return text.parsed
