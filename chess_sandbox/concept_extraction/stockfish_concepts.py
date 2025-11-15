"""Stockfish 8 concept extraction using the eval command.

This module provides functionality to extract chess concepts from positions
using Stockfish 8's eval command, which returns detailed evaluation breakdowns
by term (Material, Mobility, King safety, etc.).

Note: The 'eval' command is a non-standard UCI extension specific to Stockfish.
This implementation uses subprocess communication since python-chess doesn't
support custom UCI commands through its engine API.
"""

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess

from chess_sandbox.config import settings


@dataclass
class ConceptScores:
    """Evaluation scores for a single concept term."""

    white_mg: float
    white_eg: float
    black_mg: float
    black_eg: float
    total_mg: float
    total_eg: float

    @property
    def total_advantage(self) -> float:
        """Average advantage across middlegame and endgame phases."""
        return (self.total_mg + self.total_eg) / 2


@dataclass
class PositionConcepts:
    """All concept scores for a chess position."""

    material: ConceptScores | None = None
    imbalance: ConceptScores | None = None
    pawns: ConceptScores | None = None
    knights: ConceptScores | None = None
    bishops: ConceptScores | None = None
    rooks: ConceptScores | None = None
    queens: ConceptScores | None = None
    mobility: ConceptScores | None = None
    king_safety: ConceptScores | None = None
    threats: ConceptScores | None = None
    passed_pawns: ConceptScores | None = None
    space: ConceptScores | None = None
    total_eval: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with simplified concept names and total advantages."""
        result: dict[str, Any] = {}
        for field in [
            "material",
            "imbalance",
            "pawns",
            "knights",
            "bishops",
            "rooks",
            "queens",
            "mobility",
            "king_safety",
            "threats",
            "passed_pawns",
            "space",
        ]:
            score = getattr(self, field)
            if score is not None:
                result[field] = score.total_advantage
        if self.total_eval is not None:
            result["total_eval"] = self.total_eval
        return result


class Stockfish8Config:
    """Configuration for Stockfish 8 concept extraction.

    Follows the EngineConfig pattern from chess_sandbox.engine.analyse.
    """

    def __init__(self, stockfish_8_path: str | Path | None = None):
        """Initialize configuration.

        Args:
            stockfish_8_path: Path to Stockfish 8 binary.
                If None, uses STOCKFISH_8_PATH from settings,
                falling back to data/engines/stockfish-8/src/stockfish.
        """
        if stockfish_8_path is None:
            # Try settings first, then fallback to default location
            if hasattr(settings, "STOCKFISH_8_PATH"):
                self.stockfish_path = Path(settings.STOCKFISH_8_PATH)
            else:
                # Default path relative to project root
                default_path = Path(__file__).parent.parent.parent / "data" / "engines" / "stockfish-8" / "src" / "stockfish"
                self.stockfish_path = default_path
        else:
            self.stockfish_path = Path(stockfish_8_path)

    def validate(self) -> None:
        """Validate that Stockfish 8 binary exists.

        Raises:
            FileNotFoundError: If binary not found at configured path
        """
        if not self.stockfish_path.exists():
            raise FileNotFoundError(
                f"Stockfish 8 binary not found at {self.stockfish_path}. "
                f"Please compile Stockfish 8 or update STOCKFISH_8_PATH in settings."
            )


class Stockfish8ConceptExtractor:
    """Extract chess concepts using Stockfish 8 eval command.

    Uses subprocess to communicate directly with Stockfish since the eval
    command is a non-standard UCI extension not supported by python-chess.

    Example:
        >>> from pathlib import Path
        >>> config = Stockfish8Config()
        >>> extractor = Stockfish8ConceptExtractor(config)
        >>> fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        >>> concepts = extractor.get_concepts(fen)
        >>> concepts.mobility.total_advantage
        0.0
    """

    # Mapping from eval output terms to attribute names
    TERM_MAP = {
        "Material": "material",
        "Imbalance": "imbalance",
        "Pawns": "pawns",
        "Knights": "knights",
        "Bishop": "bishops",
        "Rooks": "rooks",
        "Queens": "queens",
        "Mobility": "mobility",
        "King safety": "king_safety",
        "Threats": "threats",
        "Passed pawns": "passed_pawns",
        "Space": "space",
    }

    def __init__(self, config: Stockfish8Config | None = None):
        """Initialize extractor.

        Args:
            config: Configuration object. If None, uses default config.

        Raises:
            FileNotFoundError: If Stockfish 8 binary not found
        """
        if config is None:
            config = Stockfish8Config()

        config.validate()
        self.config = config
        self.stockfish_path = config.stockfish_path

    def _parse_score(self, value: str) -> float | None:
        """Parse a score value, handling --- for zero/invalid."""
        if value == "---" or value == "":
            return None
        try:
            return float(value)
        except ValueError:
            return None

    def _parse_eval_line(self, line: str) -> tuple[str, ConceptScores] | None:
        """Parse a single evaluation term line.

        Expected format:
        '       Material |   ---   --- |   ---   --- |  0.00  0.00 '

        Returns:
            Tuple of (concept_name, ConceptScores) or None if line can't be parsed
        """
        # Split by pipe character
        parts = [p.strip() for p in line.split("|")]
        if len(parts) != 4:
            return None

        term = parts[0].strip()
        if term not in self.TERM_MAP and term != "Total":
            return None

        # Parse white scores (MG, EG)
        white_scores = parts[1].split()
        if len(white_scores) != 2:
            return None
        white_mg = self._parse_score(white_scores[0])
        white_eg = self._parse_score(white_scores[1])

        # Parse black scores (MG, EG)
        black_scores = parts[2].split()
        if len(black_scores) != 2:
            return None
        black_mg = self._parse_score(black_scores[0])
        black_eg = self._parse_score(black_scores[1])

        # Parse total scores (MG, EG)
        total_scores = parts[3].split()
        if len(total_scores) != 2:
            return None
        total_mg = self._parse_score(total_scores[0])
        total_eg = self._parse_score(total_scores[1])

        if any(
            score is None
            for score in [white_mg, white_eg, black_mg, black_eg, total_mg, total_eg]
        ):
            # Handle lines with --- values (which map to None)
            # For consistency, if any component is None, we might still want to track
            # But for now, let's use 0.0 as default for missing values
            white_mg = white_mg or 0.0
            white_eg = white_eg or 0.0
            black_mg = black_mg or 0.0
            black_eg = black_eg or 0.0
            total_mg = total_mg or 0.0
            total_eg = total_eg or 0.0

        scores = ConceptScores(
            white_mg=white_mg,
            white_eg=white_eg,
            black_mg=black_mg,
            black_eg=black_eg,
            total_mg=total_mg,
            total_eg=total_eg,
        )

        return (term, scores)

    def _parse_eval_output(self, output: str) -> PositionConcepts:
        """Parse the full eval command output.

        Args:
            output: Raw output from Stockfish eval command

        Returns:
            PositionConcepts object with all parsed scores
        """
        concepts = PositionConcepts()

        lines = output.split("\n")
        for line in lines:
            # Parse evaluation term lines
            parsed = self._parse_eval_line(line)
            if parsed is not None:
                term, scores = parsed
                if term in self.TERM_MAP:
                    attr_name = self.TERM_MAP[term]
                    setattr(concepts, attr_name, scores)

            # Parse total evaluation line at the end
            # "Total Evaluation: 0.08 (white side)"
            if line.strip().startswith("Total Evaluation:"):
                match = re.search(r"Total Evaluation:\s*([-+]?\d+\.\d+)", line)
                if match:
                    concepts.total_eval = float(match.group(1))

        return concepts

    def get_concepts(self, fen: str) -> PositionConcepts:
        """Extract concept scores for a chess position.

        Args:
            fen: Position in FEN notation

        Returns:
            PositionConcepts object with all concept scores

        Raises:
            ValueError: If FEN is invalid
            RuntimeError: If Stockfish process fails
        """
        # Validate FEN
        try:
            chess.Board(fen)
        except ValueError as e:
            raise ValueError(f"Invalid FEN: {e}") from e

        # Prepare UCI commands
        commands = f"uci\nposition fen {fen}\neval\nquit\n"

        # Run Stockfish
        try:
            result = subprocess.run(
                [str(self.stockfish_path)],
                input=commands,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError("Stockfish process timed out") from e
        except Exception as e:
            raise RuntimeError(f"Failed to run Stockfish: {e}") from e

        if result.returncode != 0:
            raise RuntimeError(
                f"Stockfish exited with code {result.returncode}: {result.stderr}"
            )

        # Parse output
        return self._parse_eval_output(result.stdout)

    def get_concepts_batch(self, fens: list[str]) -> list[PositionConcepts]:
        """Extract concepts for multiple positions.

        Args:
            fens: List of FEN strings

        Returns:
            List of PositionConcepts, one per FEN
        """
        return [self.get_concepts(fen) for fen in fens]
