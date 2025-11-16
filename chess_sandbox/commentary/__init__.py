"""Chess commentary generation using LLMs and Stockfish engine analysis."""

from .cli import print_explanation
from .context import MoveContext, PositionContextBuilder, PositionState
from .evaluation import EvalConfig, EvaluationResult, ThemeJudge, ThemeJudgement
from .llm import (
    ChessPositionExplanation,
    ChessPositionExplanationWithInput,
    MoveExplanation,
    PositionDiffThoughts,
    explain_move,
    summarize_position,
)

__all__ = [
    # CLI
    "print_explanation",
    # Context
    "MoveContext",
    "PositionContextBuilder",
    "PositionState",
    # LLM
    "ChessPositionExplanation",
    "ChessPositionExplanationWithInput",
    "MoveExplanation",
    "PositionDiffThoughts",
    "explain_move",
    "summarize_position",
    # Evaluation
    "EvalConfig",
    "EvaluationResult",
    "ThemeJudge",
    "ThemeJudgement",
]
