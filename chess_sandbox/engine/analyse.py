#!/usr/bin/env python3
"""
Wraps UCI-compatible chess engines analyse method to return principal variations and candidate moves.
"""

import subprocess

import chess
import chess.engine
from chess.engine import InfoDict
from pydantic import BaseModel

from chess_sandbox.config import settings


class PrincipalVariation(BaseModel):
    score: float | None
    san_moves: list[str]


class CandidateMove(BaseModel):
    san_move: str
    score: float | None


class EngineConfig(BaseModel):
    engine_path: str
    weights_path: str | None = None
    num_lines: int
    limit: chess.engine.Limit

    @classmethod
    def stockfish(cls, num_lines: int = 5, depth: int = 20) -> "EngineConfig":
        return cls(
            num_lines=num_lines,
            engine_path=settings.STOCKFISH_PATH,
            limit=chess.engine.Limit(depth=depth),
        )

    @classmethod
    def maia(cls, num_lines: int = 5, nodes: int = 1) -> "EngineConfig":
        """
        Configure the engine for Maia analysis
        Args:
            num_lines: number of lines to return
            nodes: number of nodes to use for analysis, by default 1 node is used
            to disable searching and only rely on NN as per documentation:
            https://github.com/CSSLab/maia-chess/tree/master?tab=readme-ov-file#how-to-run-maia
        """
        return cls(
            engine_path=settings.LC0_PATH,
            weights_path=settings.MAIA_WEIGHTS_PATH,
            num_lines=num_lines,
            limit=chess.engine.Limit(nodes=nodes),
        )

    def instantiate(self) -> chess.engine.SimpleEngine:
        """Instantiate and configure the chess engine.

        Returns:
            Configured SimpleEngine instance ready for analysis
        """
        engine = chess.engine.SimpleEngine.popen_uci(self.engine_path, stderr=subprocess.DEVNULL)

        if self.weights_path is not None:
            engine.configure({"WeightsFile": self.weights_path, "UCI_ShowWDL": True})

        return engine


def validate_fen(fen: str) -> None:
    """Validate FEN notation.

    Args:
        fen: Position in FEN notation

    Raises:
        ValueError: If FEN notation is invalid
    """
    try:
        chess.Board(fen)
    except ValueError as e:
        raise ValueError(f"Invalid FEN notation: {e}") from e


def info_to_pv(info: InfoDict, board: chess.Board) -> PrincipalVariation | None:
    """Convert engine InfoDict to PrincipalVariation with SAN moves.

    >>> import chess.engine
    >>> board = chess.Board()
    >>> # Simulate engine info for e2e4
    >>> info = {'pv': [chess.Move.from_uci('e2e4')], 'score': chess.engine.PovScore(chess.engine.Cp(17), chess.WHITE)}
    >>> info_to_pv(info, board)
    PrincipalVariation(score=0.17, san_moves=['e4'])
    """
    score = info.get("score")

    pv = info.get("pv", [])

    temp_board = board.copy()
    san_moves: list[str] = []
    for move in pv:
        san_moves.append(temp_board.san(move))
        temp_board.push(move)

    score_value = score.white().score() if score else None

    return PrincipalVariation(score=score_value / 100 if score_value is not None else None, san_moves=san_moves)


def analyze_moves(
    board: chess.Board,
    engine: chess.engine.SimpleEngine,
    moves: list[chess.Move],
    limit: chess.engine.Limit,
) -> list[CandidateMove]:
    """Analyze specific candidate moves by evaluating positions after each move.

    For each move, makes the move and analyzes the resulting position with multipv=1,
    then returns the evaluation from the original side's perspective.

    Args:
        board: Current position
        engine: Pre-instantiated engine instance
        moves: List of candidate moves to evaluate
        limit: Analysis time/depth limit

    Returns:
        List of CandidateMove objects with move and score
    """
    if len(moves) == 0:
        return []

    candidate_moves: list[CandidateMove] = []

    for move in moves:
        board_after = board.copy()
        board_after.push(move)

        result = engine.analyse(board_after, limit=limit)
        score = result.get("score")

        score_value = None
        if score:
            score_value = score.white().score()
            if score_value is not None:
                score_value = score_value / 100.0

        candidate_moves.append(CandidateMove(san_move=board.san(move), score=score_value))

    return candidate_moves


def analyze_variations(
    board: chess.Board,
    engine: chess.engine.SimpleEngine,
    num_lines: int,
    limit: chess.engine.Limit,
) -> list[PrincipalVariation]:
    """Analyze position with engine, returning top principal variations.

    Args:
        board: Chess board position to analyze
        engine: Pre-instantiated engine instance
        num_lines: Number of principal variations to return
        limit: Analysis time/depth limit

    Returns:
        List of PrincipalVariation objects with scores and move sequences
    """
    if num_lines == 0:
        return []

    analysis_results = engine.analyse(board, limit, multipv=num_lines)

    principal_variations: list[PrincipalVariation] = []
    for info in analysis_results:
        pv = info_to_pv(info, board)
        if pv is not None:
            principal_variations.append(pv)

    return principal_variations
