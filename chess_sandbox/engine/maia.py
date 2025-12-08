#!/usr/bin/env python3
"""
Extract human move predictions (policy probabilities) from Maia neural network via lc0.

Maia models predict human-like moves based on training on human games at different
rating levels. This module provides an interface to extract raw policy probabilities
for all legal moves in a position.
"""

import re
import subprocess

import chess
import chess.engine
from pydantic import BaseModel

from chess_sandbox.config import settings


class HumanMove(BaseModel):
    """A candidate move with its policy probability from Maia."""

    san_move: str
    policy: float  # percentage 0-100


class MaiaConfig(BaseModel):
    """Configuration for Maia policy extraction."""

    engine_path: str
    weights_path: str
    nodes: int = 1  # minimal nodes to get policy values

    @classmethod
    def default(cls, nodes: int = 1) -> "MaiaConfig":
        """Create default Maia configuration using settings."""
        return cls(
            engine_path=settings.LC0_PATH,
            weights_path=settings.MAIA_WEIGHTS_PATH,
            nodes=nodes,
        )

    def instantiate(self) -> chess.engine.SimpleEngine:
        """Instantiate and configure lc0 engine for policy extraction."""
        engine = chess.engine.SimpleEngine.popen_uci(self.engine_path, stderr=subprocess.DEVNULL)
        engine.configure(
            {
                "WeightsFile": self.weights_path,
                "VerboseMoveStats": True,
            }
        )
        return engine


def analyze_human_moves(
    board: chess.Board,
    engine: chess.engine.SimpleEngine,
    nodes: int = 1,
) -> list[HumanMove]:
    """
    Extract policy probabilities for all legal moves from Maia.

    Uses lc0's VerboseMoveStats option to capture raw policy values
    from the neural network during position analysis.

    Args:
        board: Chess board position to analyze
        engine: Pre-instantiated lc0 engine with VerboseMoveStats enabled
        nodes: Number of nodes for analysis (default: 1 for minimal search)

    Returns:
        List of HumanMove objects sorted by policy probability (descending)

    Example:
        >>> config = MaiaConfig.default()  # doctest: +SKIP
        >>> engine = config.instantiate()  # doctest: +SKIP
        >>> board = chess.Board("3r2k1/1p3ppp/pq1Q1b2/8/8/1P3N2/P4PPP/3R2K1 w - - 3 28")  # doctest: +SKIP
        >>> moves = analyze_human_moves(board, engine)  # doctest: +SKIP
        >>> moves[0].san_move  # doctest: +SKIP
        'Qxb6'
        >>> moves[0].policy > 50  # doctest: +SKIP
        True
    """
    policy_data: dict[str, float] = {}

    with engine.analysis(board, chess.engine.Limit(nodes=nodes)) as analysis:
        for info in analysis:
            if "string" in info:
                string = info["string"]
                # Parse verbose stats: "d6b6  (1244) N:  2 (+ 1) (P: 62.87%)"
                # Note: Index field can have trailing spaces e.g. "(34  )"
                match = re.match(r"(\w+)\s+\(\s*\d+\s*\)\s+N:\s*\d+.*?\(P:\s*([\d.]+)%\)", string)
                if match and match.group(1) != "node":
                    uci = match.group(1)
                    policy = float(match.group(2))
                    # Keep last seen value (visited nodes show updated values)
                    policy_data[uci] = policy

    # Convert UCI moves to SAN and create HumanMove objects
    human_moves: list[HumanMove] = []
    for uci, policy in policy_data.items():
        try:
            move = chess.Move.from_uci(uci)
            san = board.san(move)
            human_moves.append(HumanMove(san_move=san, policy=policy))
        except ValueError:
            continue

    # Sort by policy descending
    human_moves.sort(key=lambda m: m.policy, reverse=True)
    return human_moves


def get_human_moves(
    fen: str,
    config: MaiaConfig | None = None,
) -> list[HumanMove]:
    """
    High-level function to get human move predictions for a FEN position.

    Args:
        fen: Position in FEN notation
        config: Optional MaiaConfig (uses defaults if not provided)

    Returns:
        List of HumanMove objects sorted by policy probability (descending)

    Example:
        >>> moves = get_human_moves(  # doctest: +SKIP
        ...     "3r2k1/1p3ppp/pq1Q1b2/8/8/1P3N2/P4PPP/3R2K1 w - - 3 28"
        ... )
        >>> moves[0].san_move  # doctest: +SKIP
        'Qxb6'
    """
    if config is None:
        config = MaiaConfig.default()

    board = chess.Board(fen)
    engine = config.instantiate()

    try:
        return analyze_human_moves(board, engine, nodes=config.nodes)
    finally:
        engine.quit()
