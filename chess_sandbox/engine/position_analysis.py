#!/usr/bin/env python3
import chess
import click
from pydantic import BaseModel, computed_field

from chess_sandbox.engine.analyse import (
    CandidateMove,
    EngineConfig,
    PrincipalVariation,
    analyze_moves,
    analyze_variations,
)


class PositionAnalysis(BaseModel):
    fen: str
    next_move: str | None
    principal_variations: list[PrincipalVariation]
    human_moves: list[CandidateMove] | None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def formatted_text(self) -> str:
        return self.format_as_text()

    def format_as_text(self, max_display_moves: int = 8) -> str:
        board = chess.Board(self.fen)
        if self.next_move:
            board.push_san(self.next_move)

        lines: list[str] = []
        lines.append("POSITION:")
        lines.append(str(board))
        lines.append(f"\nFEN: {board.fen()}")
        lines.append(f"Turn: {'White' if board.turn == chess.WHITE else 'Black'} to move\n")

        if len(self.principal_variations) > 0:
            header = "PRINCIPAL VARIATIONS:\n"
            lines.append(header)

            for i, pv in enumerate(self.principal_variations, 1):
                eval_str = f"{pv.score:+.2f}"
                displayed_moves = pv.san_moves[:max_display_moves]
                move_sequence = " ".join(displayed_moves)

                lines.append(f"Line {i}: Eval {eval_str}")
                lines.append(f"  Moves: {move_sequence}")

                if len(pv.san_moves) > max_display_moves:
                    lines.append(f"  ... and {len(pv.san_moves) - max_display_moves} more moves")
                lines.append("")

        if self.human_moves is not None and len(self.human_moves) > 0:
            lines.append("HUMAN-LIKE MOVES (Maia):\n")

            for i, candidate in enumerate(self.human_moves, 1):
                eval_str = f"{candidate.score:+.2f}" if candidate.score is not None else "N/A"
                lines.append(f"Rank {i}: {candidate.san_move} (Eval {eval_str})")

            lines.append("")

        return "\n".join(lines)


def analyze_position(
    fen: str,
    next_move: str | None = None,
    depth: int = 20,
    num_lines: int = 5,
    with_maia: bool = False,
    maia_nodes: int = 1,
) -> PositionAnalysis:
    """Analyze a chess position given in FEN notation.

    Args:
        fen: Position in FEN notation
        next_move: Optional next move in SAN notation to analyze after making it
        depth: Analysis depth for Stockfish (default: 20)
        num_lines: Number of lines to analyze (default: 5)
        with_maia: Also analyze with Lc0/Maia for human-like evaluation
        maia_nodes: Number of nodes for Lc0/Maia analysis (default: 1)

    Returns:
        PositionAnalysis object with variations, moves, and formatted text
    """
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

    return PositionAnalysis(
        fen=fen,
        next_move=next_move,
        principal_variations=stockfish_variations,
        human_moves=human_moves,
    )


@click.command()
@click.argument("fen")
@click.option("--next-move", required=False)
@click.option("--depth", default=20, help="Analysis depth for Stockfish (default: 20)")
@click.option("--num-lines", default=5, help="Number of lines to analyze (default: 5)")
@click.option("--with-maia", is_flag=True, help="Also analyze with Lc0/Maia for human-like evaluation")
@click.option("--maia-nodes", default=1, help="Number of nodes for Lc0/Maia analysis (default: 1)")
def cli(fen: str, next_move: str | None, depth: int, num_lines: int, with_maia: bool, maia_nodes: int):
    """Analyze a chess position given in FEN notation."""
    result = analyze_position(fen, next_move, depth, num_lines, with_maia, maia_nodes)
    print(result.formatted_text)


if __name__ == "__main__":
    cli()
