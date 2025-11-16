"""
LLM workflow module for chess position analysis using Mirascope v2.

This module manages prompts, response parsing, and LLM client configuration.
It consumes PositionState and MoveContext data classes from the context module.
"""

from textwrap import dedent

from mirascope import llm
from pydantic import BaseModel, Field

from .context import MoveContext, PositionContext


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
    context: PositionContext,
    model: str = "gpt-4o-2024-08-06",
    reasoning_effort: str | None = None,
) -> ChessPositionExplanationWithInput:
    """Analyze a chess position using pre-built context and LLM.

    This function consumes a PositionContext built by PositionContextBuilder,
    ensuring a single context-building path. It mirrors the pattern used in
    explain_move which accepts MoveContext.

    Args:
        context: Pre-built position context with analysis text and tactical patterns
        model: LLM model to use
        reasoning_effort: Optional reasoning effort for models that support it (e.g., "low", "medium", "high")

    Returns:
        ChessPositionExplanationWithInput with complete analysis
    """
    tactical_context = f"\n{context.tactical_context}\n" if context.tactical_context else ""
    prompt = POSITION_PROMPT.format(analysis_text=context.analysis_text, tactical_context=tactical_context)

    kwargs = {"provider": "openai", "model_id": model, "format": ChessPositionExplanation}
    if reasoning_effort:
        kwargs["reasoning"] = {"effort": reasoning_effort}

    decorator = llm.call(**kwargs)

    @decorator
    def _analyze_position() -> str:
        return prompt

    result = _analyze_position()

    return ChessPositionExplanationWithInput(fen=context.fen, full_input=context.analysis_text, **result.model_dump())


def explain_move(
    move_ctx: MoveContext,
    model: str = "gpt-4o-2024-08-06",
    reasoning_effort: str | None = None,
) -> MoveExplanation:
    """Analyze a chess move using pre/post position context and LLM.

    Args:
        move_ctx: MoveContext with pre/post position states
        model: LLM model to use
        reasoning_effort: Optional reasoning effort for models that support it (e.g., "low", "medium", "high")

    Returns:
        MoveExplanation with complete move analysis
    """
    pre_dict = move_ctx.pre_move_state.to_prompt_dict()
    post_dict = move_ctx.post_move_state.to_prompt_dict()

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

    kwargs = {"provider": "openai", "model_id": model, "format": MoveExplanation}
    if reasoning_effort:
        kwargs["reasoning"] = {"effort": reasoning_effort}

    decorator = llm.call(**kwargs)

    @decorator
    def _analyze_move() -> str:
        return prompt

    return _analyze_move()
