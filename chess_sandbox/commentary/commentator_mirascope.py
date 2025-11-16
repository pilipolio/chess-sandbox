#!/usr/bin/env python3
"""
Chess Position Commentator using Mirascope and Lilypad

This module demonstrates how to use Mirascope's @llm.call decorator and
Lilypad's @lilypad.trace decorator for versioned, traceable LLM calls.
"""

import os
from dataclasses import dataclass
from textwrap import dedent
from typing import Any

import chess
import chess.engine
import lilypad
from mirascope.core.openai import openai_call
from pydantic import BaseModel, Field

from ..engine.analyse import EngineConfig, analyze_variations
from ..engine.position_analysis import PositionAnalysis
from .tactical_patterns import TacticalPatternDetector


class ChessPositionExplanation(BaseModel):
    assessment: int = Field(description="-1 (Black advantage), 0 (Equal), +1 (White advantage)")
    best_move: str = Field(description="Best move in algebraic notation")
    themes: list[str] = Field(description="Key strategic or tactical themes (keywords)")
    variations: str = Field(description="Key lines and variations in PGN format with comments")


class ChessPositionExplanationWithInput(ChessPositionExplanation):
    fen: str = Field(description="The input position in FEN format")
    full_input: str = Field(description="The full input position")
    analysis_config: dict[str, Any] = Field(description="Configuration used for analysis")


@dataclass
class MirascopeCommentatorConfig:
    """Configuration for the Mirascope-based commentator."""

    engine_depth: int = 20
    engine_num_lines: int = 5
    llm_model: str = "gpt-4o-mini"
    include_tactical_patterns: bool = True
    lilypad_project_id: str | None = None
    lilypad_api_key: str | None = None

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> "MirascopeCommentatorConfig":
        return cls(
            engine_depth=params.get("engine", {}).get("depth", 20),
            engine_num_lines=params.get("engine", {}).get("num_lines", 5),
            llm_model=params.get("llm", {}).get("model", "gpt-4o-mini"),
            include_tactical_patterns=params.get("include_tactical_patterns", True),
            lilypad_project_id=params.get("lilypad", {}).get("project_id"),
            lilypad_api_key=params.get("lilypad", {}).get("api_key"),
        )


ANALYSIS_PROMPT = dedent("""
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


def analyze_position_with_llm(analysis_text: str, tactical_context: str, model: str) -> ChessPositionExplanation:
    """
    Analyze a chess position using an LLM.

    This function uses Mirascope's @openai_call decorator (added dynamically) and
    Lilypad's @lilypad.trace decorator for automatic versioning/tracing.
    """

    @openai_call(model=model, response_model=ChessPositionExplanation)  # type: ignore[misc]
    @lilypad.trace(versioning="automatic")  # type: ignore[misc]
    def _inner_call() -> str:
        return ANALYSIS_PROMPT.format(analysis_text=analysis_text, tactical_context=tactical_context)

    return _inner_call()


class MirascopeCommentator:
    """Chess position commentator using Mirascope and Lilypad."""

    def __init__(self, config: MirascopeCommentatorConfig):
        self.config = config

        # Configure Lilypad if credentials are provided
        if config.lilypad_project_id and config.lilypad_api_key:
            lilypad.configure(
                project_id=config.lilypad_project_id,
                api_key=config.lilypad_api_key,
                auto_llm=True,  # Automatically trace all LLM calls
            )

    @classmethod
    def create(cls, params: dict[str, Any]) -> "MirascopeCommentator":
        """Create a commentator from a parameter dictionary."""
        config = MirascopeCommentatorConfig.from_dict(params)
        return cls(config)

    def analyze(self, board: chess.Board) -> ChessPositionExplanationWithInput:
        """
        Analyze a chess position and return a structured explanation.

        Args:
            board: The chess board to analyze

        Returns:
            ChessPositionExplanationWithInput containing the analysis
        """
        # Prepare analysis text
        if self.config.engine_num_lines > 0:
            # Use Stockfish engine for analysis
            config = EngineConfig.stockfish(num_lines=self.config.engine_num_lines, depth=self.config.engine_depth)
            engine = config.instantiate()
            try:
                limit = chess.engine.Limit(depth=self.config.engine_depth)
                analysis_results = analyze_variations(board, engine, self.config.engine_num_lines, limit)
                position_analysis = PositionAnalysis(
                    fen=board.fen(), next_move=None, principal_variations=analysis_results, human_moves=None
                )
                analysis_text = position_analysis.format_as_text()
            finally:
                engine.quit()
        else:
            # Skip engine analysis - just provide the FEN
            analysis_text = f"Position (FEN): {board.fen()}"

        # Detect tactical patterns
        tactical_context = ""
        if self.config.include_tactical_patterns:
            detector = TacticalPatternDetector(board)
            tactical_context = detector.get_tactical_context()
            if tactical_context:
                tactical_context = f"\n{tactical_context}\n"

        # Call the LLM via Mirascope (with Lilypad tracing)
        explanation = analyze_position_with_llm(
            analysis_text=analysis_text, tactical_context=tactical_context, model=self.config.llm_model
        )

        return ChessPositionExplanationWithInput(
            fen=board.fen(),
            full_input=analysis_text,
            analysis_config={
                "engine_depth": self.config.engine_depth,
                "engine_num_lines": self.config.engine_num_lines,
                "llm_model": self.config.llm_model,
                "include_tactical_patterns": self.config.include_tactical_patterns,
            },
            **explanation.model_dump(),
        )


def main():
    """Example usage of the Mirascope commentator."""
    # Example configuration
    params = {
        "engine": {"depth": 20, "num_lines": 5},
        "llm": {"model": "gpt-4o-mini"},
        "include_tactical_patterns": True,
        "lilypad": {
            "project_id": os.environ.get("LILYPAD_PROJECT_ID"),
            "api_key": os.environ.get("LILYPAD_API_KEY"),
        },
    }

    # Example position
    fen = "8/8/2K5/p1p5/P1P5/1k6/8/8 w - - 0 58"
    board = chess.Board(fen)

    commentator = MirascopeCommentator.create(params)
    explanation = commentator.analyze(board)

    print("=" * 70)
    print("CHESS POSITION ANALYSIS")
    print("=" * 70)
    print(f"\nFEN: {explanation.fen}")
    print(f"Assessment: {explanation.assessment}")
    print(f"Best Move: {explanation.best_move}")
    print(f"Themes: {', '.join(explanation.themes)}")
    print(f"\nVariations:\n{explanation.variations}")
    print("=" * 70)


if __name__ == "__main__":
    main()
