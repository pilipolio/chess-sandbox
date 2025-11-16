"""
Pure chess position context building module.

This module handles extracting context from chess positions using Stockfish,
concept extraction, and tactical pattern detection. It's dependency-light
(no OpenAI or Click) to make it easy to test and reuse.
"""

import chess
import chess.engine
from pydantic import BaseModel, Field

from ..concept_extraction.model.inference import ConceptExtractor
from ..engine.analyse import EngineConfig, analyze_variations
from ..engine.position_analysis import PositionAnalysis
from .tactical_patterns import TacticalPatternDetector


class PositionState(BaseModel):
    """Complete state of a chess position including evaluation, concepts, and tactical patterns."""

    evaluation: float = Field(description="Stockfish evaluation in pawns (positive = White advantage)")
    evaluation_category: str = Field(
        description="Human-readable evaluation category (winning/advantage/slight advantage/equal/etc.)"
    )
    concepts: dict[str, float] = Field(description="Chess concepts detected with confidence scores (0.0-1.0)")
    tactical_patterns: dict[str, list[str]] = Field(
        description="Tactical patterns detected (pins, forks) with descriptions"
    )
    best_moves: list[str] = Field(description="Top engine moves in SAN notation")

    def to_prompt_dict(self) -> dict[str, str]:
        """Format position state as dictionary suitable for LLM prompts.

        Returns:
            Dictionary with formatted strings for evaluation, concepts, tactics, and moves
        """

        def format_concepts(concepts: dict[str, float]) -> str:
            if not concepts:
                return "None"
            return ", ".join([f"{name} ({score:.1f})" for name, score in sorted(concepts.items())])

        def format_tactics(patterns: dict[str, list[str]]) -> str:
            if not patterns:
                return "None"
            parts: list[str] = []
            for pattern_type, descriptions in patterns.items():
                parts.append(f"{pattern_type.upper()}: {'; '.join(descriptions)}")
            return " | ".join(parts)

        return {
            "eval_category": self.evaluation_category,
            "concepts": format_concepts(self.concepts),
            "tactics": format_tactics(self.tactical_patterns),
            "best_moves": ", ".join(self.best_moves),
        }


class MoveContext(BaseModel):
    """Context for a chess move, including pre/post states and delta metadata."""

    move_san: str = Field(description="The move in SAN notation")
    pre_move_state: PositionState = Field(description="Position state before the move")
    post_move_state: PositionState = Field(description="Position state after the move")
    evaluation_change: float = Field(description="Change in evaluation (post - pre)")
    evaluation_change_category: str = Field(
        description="Human-readable evaluation change category (significant improvement/minor improvement/etc.)"
    )


class PositionContext(BaseModel):
    """Complete context for position analysis including formatted text for LLM prompts."""

    fen: str = Field(description="Position in FEN notation")
    position_state: PositionState = Field(description="Raw position state (evaluation, concepts, tactics)")
    analysis_text: str = Field(description="Formatted engine analysis text from PositionAnalysis")
    tactical_context: str = Field(description="Formatted tactical patterns text")


class PositionContextBuilder:
    """Builds chess position context using engine analysis, concept extraction, and tactical detection.

    This class owns the Stockfish lifecycle, concept extraction, and tactical detection.
    It produces immutable PositionState and MoveContext data structures.
    """

    def __init__(self, engine_config: EngineConfig, concept_extractor: ConceptExtractor):
        """Initialize the context builder.

        Args:
            engine_config: Configuration for the Stockfish engine
            concept_extractor: Concept extraction model
        """
        self.engine_config = engine_config
        self.concept_extractor = concept_extractor

    @staticmethod
    def _categorize_evaluation(evaluation: float) -> str:
        """Categorize a chess position evaluation into human-readable categories.

        Args:
            evaluation: Evaluation in pawns (positive = advantage, negative = disadvantage)

        Returns:
            Category string like "winning", "advantage", "equal", etc.
        """
        if evaluation > 3.0:
            return "winning"
        elif evaluation >= 1.0:
            return "advantage"
        elif evaluation >= 0.2:
            return "slight advantage"
        elif evaluation > -0.2:
            return "equal"
        elif evaluation > -1.0:
            return "slight disadvantage"
        elif evaluation > -3.0:
            return "disadvantage"
        else:
            return "losing"

    @staticmethod
    def _categorize_evaluation_change(change: float) -> str:
        """Categorize change in evaluation into human-readable categories.

        Args:
            change: Change in evaluation (positive = improvement, negative = worsening)

        Returns:
            Category string like "significant improvement", "roughly equal", etc.
        """
        if change >= 1.0:
            return "significant improvement"
        elif change >= 0.2:
            return "minor improvement"
        elif change > -0.2:
            return "roughly equal"
        elif change > -1.0:
            return "minor worsening"
        else:
            return "significant worsening"

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
        analysis_results = analyze_variations(board, engine, self.engine_config.num_lines, limit)

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
            evaluation_category=self._categorize_evaluation(best_eval),
            concepts=concepts_dict,
            tactical_patterns=tactical_patterns,
            best_moves=best_moves,
        )

    def build_position(self, fen: str) -> PositionState:
        """Build context for a chess position.

        Args:
            fen: Position in FEN notation

        Returns:
            PositionState with complete position context
        """
        board = chess.Board(fen)
        engine = self.engine_config.instantiate()

        try:
            return self._extract_position_context(board, engine, self.engine_config.limit)
        finally:
            engine.quit()

    def build_position_context(self, fen: str) -> PositionContext:
        """Build complete context for position analysis including formatted text for LLM prompts.

        This is the single source of truth for context building. It creates both the
        structured PositionState and the formatted text needed for LLM prompts.

        Args:
            fen: Position in FEN notation

        Returns:
            PositionContext with PositionState, analysis text, and tactical context
        """
        board = chess.Board(fen)
        engine = self.engine_config.instantiate()

        try:
            # Get engine analysis (single call)
            analysis_results = analyze_variations(board, engine, self.engine_config.num_lines, self.engine_config.limit)

            # Build PositionState (reuse existing logic)
            position_state = self._extract_position_context(board, engine, self.engine_config.limit)

            # Format analysis text for LLM prompt
            position_analysis = PositionAnalysis(
                fen=fen, next_move=None, principal_variations=analysis_results, human_moves=None
            )
            analysis_text = position_analysis.format_as_text()

            # Get tactical context text
            detector = TacticalPatternDetector(board)
            tactical_context = detector.get_tactical_context()

            return PositionContext(
                fen=fen,
                position_state=position_state,
                analysis_text=analysis_text,
                tactical_context=tactical_context,
            )

        finally:
            engine.quit()

    def build_move_context(self, fen: str, move_san: str) -> MoveContext:
        """Build context for a chess move by comparing pre-move and post-move positions.

        Args:
            fen: Position in FEN notation before the move
            move_san: Move to analyze in SAN notation (e.g., 'Nf3', 'exd5')

        Returns:
            MoveContext with pre/post states and delta metadata
        """
        board = chess.Board(fen)
        engine = self.engine_config.instantiate()

        try:
            # Extract pre-move context
            pre_move_state = self._extract_position_context(board, engine, self.engine_config.limit)

            # Make the move on a copy
            post_move_board = board.copy()
            post_move_board.push_san(move_san)

            # Extract post-move context (note: post_move_board.turn is now the OTHER side)
            # We need to negate the evaluation to get it from the original side's perspective
            post_move_state_raw = self._extract_position_context(post_move_board, engine, self.engine_config.limit)

            # Adjust post-move evaluation to be from original side's perspective
            # After the move, it's the opponent's turn, so negate to compare apples-to-apples
            post_move_evaluation = -post_move_state_raw.evaluation
            post_move_state = PositionState(
                evaluation=post_move_evaluation,
                evaluation_category=self._categorize_evaluation(post_move_evaluation),
                concepts=post_move_state_raw.concepts,
                tactical_patterns=post_move_state_raw.tactical_patterns,
                best_moves=post_move_state_raw.best_moves,
            )

            # Compute evaluation change and categorize it
            evaluation_change = post_move_state.evaluation - pre_move_state.evaluation
            evaluation_change_category = self._categorize_evaluation_change(evaluation_change)

            return MoveContext(
                move_san=move_san,
                pre_move_state=pre_move_state,
                post_move_state=post_move_state,
                evaluation_change=evaluation_change,
                evaluation_change_category=evaluation_change_category,
            )

        finally:
            engine.quit()
