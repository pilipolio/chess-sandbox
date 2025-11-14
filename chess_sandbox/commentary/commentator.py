#!/usr/bin/env python3
"""
Chess Engine Analysis with LLM Explanations - PoC

Demonstrates how combining Stockfish chess engine analysis with LLM
can provide natural language explanations for chess positions.
"""

import os
from dataclasses import dataclass
from textwrap import dedent
from typing import Any

import chess
import chess.engine
from openai import OpenAI
from openai.types.shared.reasoning_effort import ReasoningEffort
from openai.types.shared_params.reasoning import Reasoning
from pydantic import BaseModel, Field

from ..engine.analyse import EngineConfig, analyze_variations
from ..engine.position_analysis import PositionAnalysis
from ..lichess import get_analysis_url
from .tactical_patterns import TacticalPatternDetector


class ChessPositionExplanation(BaseModel):
    assessment: int = Field(description="-1 (Black advantage), 0 (Equal), +1 (White advantage)")
    best_move: str = Field(description="Best move in algebraic notation")
    themes: list[str] = Field(description="Key strategic or tactical themes (keywords)")
    variations: str = Field(description="Key lines and variations in PGN format with comments")


class ChessPositionExplanationWithInput(ChessPositionExplanation):
    fen: str = Field(description="The input position in FEN format")
    full_input: str = Field(description="The full input position ")


@dataclass
class Commentator:
    """Chess position commentator combining Stockfish analysis with LLM explanations."""

    PROMPT = dedent("""
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

    engine_num_lines: int
    engine_depth: int
    llm_model: str
    llm_reasoning_effort: ReasoningEffort | None
    client: OpenAI

    @classmethod
    def create(cls, params: dict[str, Any]) -> "Commentator":
        return cls(
            engine_depth=params.get("engine", {}).get("depth", 20),
            engine_num_lines=params.get("engine", {}).get("num_lines", 5),
            llm_model=params.get("llm", {}).get("model", "gpt-4o-2024-08-06"),
            llm_reasoning_effort=params.get("llm", {}).get("reasoning_effort"),
            client=OpenAI(api_key=os.environ.get("OPENAI_API_KEY")),
        )

    def analyze(self, board: chess.Board) -> ChessPositionExplanationWithInput:
        config = EngineConfig.stockfish(num_lines=self.engine_num_lines, depth=self.engine_depth)
        engine = config.instantiate()

        try:
            limit = chess.engine.Limit(depth=self.engine_depth)
            analysis_results = analyze_variations(board, engine, self.engine_num_lines, limit)
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

            prompt = self.PROMPT.format(analysis_text=analysis_text, tactical_context=tactical_context)

            response = self.client.responses.parse(
                model=self.llm_model,
                input=prompt,
                text_format=ChessPositionExplanation,
                reasoning=Reasoning(effort=self.llm_reasoning_effort),
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
            return ChessPositionExplanationWithInput(
                fen=board.fen(), full_input=analysis_text, **parsed_data.model_dump()
            )
        finally:
            engine.quit()


def print_explanation(explanation: ChessPositionExplanationWithInput):
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


def main():
    params = {
        "engine": {"depth": 20, "num_lines": 5},
        "llm": {"model": "gpt-5-mini", "reasoning_effort": "low"},
        # "llm": {"model": "gpt-4o-mini"},
    }

    fen = "8/8/2K5/p1p5/P1P5/1k6/8/8 w - - 0 58"
    print(get_analysis_url(fen))
    board = chess.Board(fen)

    commentator = Commentator.create(params)

    explanation = commentator.analyze(board)
    print_explanation(explanation)


if __name__ == "__main__":
    main()
