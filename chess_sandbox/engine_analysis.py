#!/usr/bin/env python3
"""
Deterministic Chess Engine Analysis Module

Provides Stockfish-based position analysis and text formatting utilities.
"""

import subprocess

import chess
import chess.engine
import click
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

    score_value = score.relative.score() if score else None

    return PrincipalVariation(score=score_value / 100 if score_value is not None else None, san_moves=san_moves)


def analyze_moves(board: chess.Board, config: EngineConfig, moves: list[chess.Move]) -> list[CandidateMove]:
    """Analyze specific candidate moves by evaluating positions after each move.

    For each move, makes the move and analyzes the resulting position with multipv=1,
    then returns the evaluation from the original side's perspective.

    Args:
        board: Current position
        config: Engine configuration
        moves: List of candidate moves to evaluate

    Returns:
        List of CandidateMove objects with move and score
    """
    if len(moves) == 0:
        return []

    engine = chess.engine.SimpleEngine.popen_uci(config.engine_path, stderr=subprocess.DEVNULL)
    if config.weights_path is not None:
        engine.configure({"WeightsFile": config.weights_path, "UCI_ShowWDL": True})

    candidate_moves: list[CandidateMove] = []

    for move in moves:
        board_after = board.copy()
        board_after.push(move)

        result = engine.analyse(board_after, limit=config.limit)
        score = result.get("score")

        score_value = None
        if score:
            score_value = score.white().score()
            if score_value is not None:
                score_value = score_value / 100.0

        candidate_moves.append(CandidateMove(san_move=board.san(move), score=score_value))

    engine.quit()
    return candidate_moves


def analyze_variations(board: chess.Board, config: EngineConfig) -> list[PrincipalVariation]:
    """Analyze position with configured engine, returning top principal variations.

    Args:
        board: Chess board position to analyze
        config: Engine configuration

    Returns:
        List of PrincipalVariation objects with scores and move sequences
    """
    if config.num_lines == 0:
        return []

    engine = chess.engine.SimpleEngine.popen_uci(config.engine_path, stderr=subprocess.DEVNULL)

    if config.weights_path is not None:
        engine.configure({"WeightsFile": config.weights_path, "UCI_ShowWDL": True})

    analysis_results = engine.analyse(board, config.limit, multipv=config.num_lines)

    principal_variations: list[PrincipalVariation] = []
    for info in analysis_results:
        pv = info_to_pv(info, board)
        if pv is not None:
            principal_variations.append(pv)

    engine.quit()
    return principal_variations


def format_as_text(
    board: chess.Board,
    principal_variations: list[PrincipalVariation],
    max_display_moves: int = 8,
    human_moves: list[CandidateMove] | None = None,
) -> str:
    lines: list[str] = []
    lines.append("POSITION:")
    lines.append(str(board))
    lines.append(f"\nFEN: {board.fen()}")
    lines.append(f"Turn: {'White' if board.turn == chess.WHITE else 'Black'} to move\n")

    if len(principal_variations) > 0:
        header = "PRINCIPAL VARIATIONS:\n"
        lines.append(header)

        for i, pv in enumerate[PrincipalVariation](principal_variations, 1):
            eval_str = f"{pv.score:+.2f}"
            displayed_moves = pv.san_moves[:max_display_moves]
            move_sequence = " ".join(displayed_moves)

            lines.append(f"Line {i}: Eval {eval_str}")
            lines.append(f"  Moves: {move_sequence}")

            if len(pv.san_moves) > max_display_moves:
                lines.append(f"  ... and {len(pv.san_moves) - max_display_moves} more moves")
            lines.append("")

    if human_moves is not None and len(human_moves) > 0:
        lines.append("HUMAN-LIKE MOVES (Maia):\n")

        for i, candidate in enumerate(human_moves, 1):
            eval_str = f"{candidate.score:+.2f}" if candidate.score is not None else "N/A"
            lines.append(f"Rank {i}: {candidate.san_move} (Eval {eval_str})")

        lines.append("")

    return "\n".join(lines)


@click.command()
@click.argument("fen")
@click.option("--next-move", required=False)
@click.option("--depth", default=20, help="Analysis depth for Stockfish (default: 20)")
@click.option("--num-lines", default=5, help="Number of lines to analyze (default: 5)")
@click.option("--with-maia", is_flag=True, help="Also analyze with Lc0/Maia for human-like evaluation")
@click.option("--maia-nodes", default=1, help="Number of nodes for Lc0/Maia analysis (default: 1)")
def main(fen: str, next_move: str, depth: int, num_lines: int, with_maia: bool, maia_nodes: int):
    """Analyze a chess position given in FEN notation."""
    board = chess.Board(fen)

    if next_move:
        board.push_san(next_move)

    stockfish_variations = analyze_variations(board, config=EngineConfig.stockfish(num_lines=num_lines, depth=depth))

    human_moves = None
    if with_maia:
        maia_config = EngineConfig.maia(num_lines=num_lines, nodes=maia_nodes)
        maia_variations = analyze_variations(board, config=maia_config)
        candidate_moves = [board.parse_san(pv.san_moves[0]) for pv in maia_variations if pv.san_moves]

        # Maia variations from multipv analysis have all the same score, so we need to analyze each move individually
        # TODO: understand if that's possible to return policy network probabilities instead of pseudo eval
        human_moves = analyze_moves(board, maia_config, candidate_moves)

    print(format_as_text(board, stockfish_variations, human_moves=human_moves))


if __name__ == "__main__":
    main()
