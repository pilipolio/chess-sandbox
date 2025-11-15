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
import click
from openai import OpenAI
from openai.types.shared.reasoning_effort import ReasoningEffort
from openai.types.shared_params.reasoning import Reasoning
from pydantic import BaseModel, Field

from ..concept_extraction.model.inference import ConceptExtractor
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


class PositionState(BaseModel):
    """Complete state of a chess position including evaluation, concepts, and tactical patterns."""

    evaluation: float = Field(description="Stockfish evaluation in pawns (positive = White advantage)")
    concepts: dict[str, float] = Field(description="Chess concepts detected with confidence scores (0.0-1.0)")
    tactical_patterns: dict[str, list[str]] = Field(
        description="Tactical patterns detected (pins, forks) with descriptions"
    )
    best_moves: list[str] = Field(description="Top engine moves in SAN notation")


class PositionDiff(BaseModel):
    """Differences between pre-move and post-move positions."""

    concepts_gained: list[str] = Field(description="New concepts appearing after the move")
    concepts_lost: list[str] = Field(description="Concepts no longer present after the move")
    evaluation_change: float = Field(description="Change in evaluation in pawns (positive = improved for side to move)")
    tactical_changes: str = Field(description="Description of changes in tactical patterns")
    move_comment: str = Field(
        description="Concise 1-2 sentence explanation of why this move was played and how it changed the position"
    )


class MoveExplanation(BaseModel):
    """Complete analysis of a chess move comparing pre and post-move positions."""

    move_san: str = Field(description="The move played in SAN notation")
    pre_move_state: PositionState = Field(description="Position state before the move")
    post_move_state: PositionState = Field(description="Position state after the move")
    differences: PositionDiff = Field(description="What changed between positions")
    move_comment: str = Field(
        description="Concise 1-2 sentence explanation of why this move was played and how it changed the position"
    )


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

    MOVE_ANALYSIS_PROMPT = dedent("""
        You are a chess grandmaster analyzing a move. Compare the position before and after the move,
        and identify ONLY the SIGNIFICANT changes that matter.

        MOVE PLAYED: {move_san}

        PRE-MOVE POSITION:
        - Evaluation: {pre_eval:+.1f}
        - Concepts: {pre_concepts}
        - Tactical patterns: {pre_tactics}
        - Best moves: {pre_best_moves}

        POST-MOVE POSITION:
        - Evaluation: {post_eval:+.1f}
        - Concepts: {post_concepts}
        - Tactical patterns: {post_tactics}
        - Best moves: {post_best_moves}

        IMPORTANT: Only report MEANINGFUL changes. Ignore minor fluctuations.

        1. **concepts_gained**: List ONLY strategically significant concepts that appeared
           (empty list if no meaningful concepts were gained)

        2. **concepts_lost**: List ONLY strategically significant concepts that disappeared
           (empty list if no meaningful concepts were lost)

        3. **evaluation_change**: Numeric change in evaluation (post - pre)

        4. **tactical_changes**: Describe ONLY significant tactical shifts:
           - New tactical threats or opportunities created
           - Important tactical vulnerabilities removed
           - Leave empty string "" if no significant tactical changes

        5. **move_comment**: One concise sentence explaining the move's purpose:
           - If the move creates/addresses tactical threats, mention that
           - If the move improves strategic position (concepts), mention that
           - If evaluation changed significantly, mention whether it's an improvement/mistake
           - Be specific about what changed, not what stayed the same

        Be selective. Only highlight changes that actually matter to understanding the move.
    """).strip()

    engine_num_lines: int
    engine_depth: int
    llm_model: str
    llm_reasoning_effort: ReasoningEffort | None
    client: OpenAI
    concept_extractor: ConceptExtractor

    @classmethod
    def create(cls, params: dict[str, Any]) -> "Commentator":
        return cls(
            engine_depth=params.get("engine", {}).get("depth", 20),
            engine_num_lines=params.get("engine", {}).get("num_lines", 5),
            llm_model=params.get("llm", {}).get("model", "gpt-4o-2024-08-06"),
            llm_reasoning_effort=params.get("llm", {}).get("reasoning_effort"),
            client=OpenAI(api_key=os.environ.get("OPENAI_API_KEY")),
            concept_extractor=ConceptExtractor.from_hf(
                probe_repo_id=params.get("concept_extractor", {}).get(
                    "probe_repo_id", "pilipolio/chess-positions-extractor"
                ),
                revision="production",
            ),
        )

    def _extract_position_context(
        self, board: chess.Board, engine: chess.engine.SimpleEngine, limit: chess.engine.Limit
    ) -> PositionState:
        """Extract all context (concepts, tactics, evaluation) for a position.

        Args:
            board: The chess position to analyze
            engine: Pre-instantiated Stockfish engine
            limit: Engine analysis limit

        Returns:
            PositionState with complete position context
        """
        # Get engine analysis
        analysis_results = analyze_variations(board, engine, self.engine_num_lines, limit)

        # Extract evaluation from best line (from perspective of side to move)
        if analysis_results and analysis_results[0].score is not None:
            best_eval = analysis_results[0].score
            # Negate if black to move (engine returns from white's perspective)
            if board.turn == chess.BLACK:
                best_eval = -best_eval
        else:
            best_eval = 0.0

        # Round evaluation to 1 decimal
        best_eval = round(best_eval, 1)

        # Get top moves
        best_moves = [pv.san_moves[0] for pv in analysis_results if pv.san_moves][:5]

        # Extract concepts with confidence scores
        concept_predictions_raw = self.concept_extractor.extract_concepts_with_confidence(board.fen())
        # Since we pass a single FEN string, we get a list of tuples, not a list of lists
        assert isinstance(concept_predictions_raw, list) and (
            not concept_predictions_raw or isinstance(concept_predictions_raw[0], tuple)
        ), "Expected list[tuple[str, float]] for single FEN"
        concept_predictions: list[tuple[str, float]] = concept_predictions_raw  # type: ignore[assignment]
        # Filter concepts below 0.1 threshold and round to 1 decimal
        concepts_dict: dict[str, float] = {
            name: round(float(score), 1) for name, score in concept_predictions if score >= 0.1
        }

        # Detect tactical patterns
        detector = TacticalPatternDetector(board)
        pins = detector.detect_pins()
        forks = detector.detect_forks()

        tactical_patterns: dict[str, list[str]] = {}
        if pins:
            tactical_patterns["pins"] = [pin.describe() for pin in pins]
        if forks:
            tactical_patterns["forks"] = [fork.describe() for fork in forks]

        return PositionState(
            evaluation=best_eval,
            concepts=concepts_dict,
            tactical_patterns=tactical_patterns,
            best_moves=best_moves,
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

    def analyze_move(self, board: chess.Board, move_san: str) -> MoveExplanation:
        """Analyze a chess move by comparing pre-move and post-move positions.

        Args:
            board: The chess position before the move
            move_san: The move to analyze in SAN notation (e.g., 'Nf3', 'exd5')

        Returns:
            MoveExplanation with complete analysis of position changes
        """
        config = EngineConfig.stockfish(num_lines=self.engine_num_lines, depth=self.engine_depth)
        engine = config.instantiate()

        try:
            limit = chess.engine.Limit(depth=self.engine_depth)

            # Extract pre-move context
            pre_move_state = self._extract_position_context(board, engine, limit)

            # Make the move on a copy
            post_move_board = board.copy()
            post_move_board.push_san(move_san)

            # Extract post-move context (note: post_move_board.turn is now the OTHER side)
            # We need to negate the evaluation to get it from the original side's perspective
            post_move_state_raw = self._extract_position_context(post_move_board, engine, limit)

            # Adjust post-move evaluation to be from original side's perspective
            # After the move, it's the opponent's turn, so negate to compare apples-to-apples
            post_move_state = PositionState(
                evaluation=-post_move_state_raw.evaluation,
                concepts=post_move_state_raw.concepts,
                tactical_patterns=post_move_state_raw.tactical_patterns,
                best_moves=post_move_state_raw.best_moves,
            )

            # Format concepts for prompt
            def format_concepts(concepts: dict[str, float]) -> str:
                if not concepts:
                    return "None"
                return ", ".join([f"{name} ({score:.1f})" for name, score in sorted(concepts.items())])

            # Format tactical patterns for prompt
            def format_tactics(patterns: dict[str, list[str]]) -> str:
                if not patterns:
                    return "None"
                parts: list[str] = []
                for pattern_type, descriptions in patterns.items():
                    parts.append(f"{pattern_type.upper()}: {'; '.join(descriptions)}")
                return " | ".join(parts)

            # Create prompt with all context
            prompt = self.MOVE_ANALYSIS_PROMPT.format(
                move_san=move_san,
                pre_eval=pre_move_state.evaluation,
                pre_concepts=format_concepts(pre_move_state.concepts),
                pre_tactics=format_tactics(pre_move_state.tactical_patterns),
                pre_best_moves=", ".join(pre_move_state.best_moves),
                post_eval=post_move_state.evaluation,
                post_concepts=format_concepts(post_move_state.concepts),
                post_tactics=format_tactics(post_move_state.tactical_patterns),
                post_best_moves=", ".join(post_move_state.best_moves),
            )

            # Call LLM with structured output
            response = self.client.responses.parse(
                model=self.llm_model,
                input=prompt,
                text_format=PositionDiff,
                reasoning=Reasoning(effort=self.llm_reasoning_effort),
            )

            # Extract parsed response
            message = next((item for item in response.output if item.type == "message"), None)  # type: ignore[attr-defined]

            if not message:
                raise Exception("No message found in response output")

            text = message.content[0]  # type: ignore[attr-defined]
            assert text.type == "output_text", "Unexpected content type"  # type: ignore[attr-defined]

            if not text.parsed:  # type: ignore[attr-defined]
                raise Exception("Could not parse LLM response into PositionDiff")

            # Get the parsed differences
            parsed_diff: PositionDiff = text.parsed  # type: ignore[attr-defined]

            # Return complete move explanation
            return MoveExplanation(
                move_san=move_san,
                pre_move_state=pre_move_state,
                post_move_state=post_move_state,
                differences=parsed_diff,
                move_comment=parsed_diff.move_comment,
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
    params = {
        "engine": {"depth": depth, "num_lines": num_lines},
        "llm": {"model": model, "reasoning_effort": reasoning_effort},
    }

    print(f"\nLichess analysis: {get_analysis_url(fen)}")
    board = chess.Board(fen)

    commentator = Commentator.create(params)

    print("=" * 70)
    print("MOVE ANALYSIS")
    print("=" * 70)
    print(f"\nPosition: {fen}")
    print(f"Move: {move}")
    print(f"Engine: Stockfish (depth={depth}, lines={num_lines})")
    print(f"LLM: {model}" + (f" (reasoning={reasoning_effort})" if reasoning_effort else ""))
    print()

    move_explanation = commentator.analyze_move(board, move)

    print(move_explanation.model_dump_json(indent=2))
    print("=" * 70)


if __name__ == "__main__":
    main()
