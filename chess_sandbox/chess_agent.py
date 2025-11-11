"""
Pydantic AI chess agent with tools for interactive move exploration and evaluation.

This is a lightweight proof-of-concept demonstrating pydantic_ai agents and tools
for chess analysis, going beyond static commentary to enable interactive exploration.
"""

from dataclasses import dataclass

import chess
from pydantic_ai import Agent, RunContext

from chess_sandbox.engine import EngineConfig, analyze_moves, analyze_variations


@dataclass
class ChessAgentDeps:
    """Dependencies for the chess agent - tracks board state and engine."""

    board: chess.Board
    engine: chess.engine.SimpleEngine


chess_agent = Agent(
    "openai:gpt-4o",
    deps_type=ChessAgentDeps,
    system_prompt="""You are a chess playing and analysis assistant. You have access to tools to:
- View the current board position
- Get all legal moves available
- Analyze the position with Stockfish engine
- Evaluate specific candidate moves
- Make moves on the board
- Check game status (checkmate, stalemate, etc.)

Help the user explore chess positions, evaluate moves, and understand the position.
Use the tools to provide concrete analysis backed by engine evaluations.
""",
)


@chess_agent.tool
def get_board_fen(ctx: RunContext[ChessAgentDeps]) -> str:
    """Get the current board position in FEN notation."""
    return ctx.deps.board.fen()


@chess_agent.tool
def get_board_state(ctx: RunContext[ChessAgentDeps]) -> dict[str, str | bool]:
    """Get detailed information about the current board state."""
    board = ctx.deps.board
    return {
        "fen": board.fen(),
        "turn": "White" if board.turn == chess.WHITE else "Black",
        "is_check": board.is_check(),
        "is_checkmate": board.is_checkmate(),
        "is_stalemate": board.is_stalemate(),
        "is_game_over": board.is_game_over(),
        "fullmove_number": str(board.fullmove_number),
        "halfmove_clock": str(board.halfmove_clock),
    }


@chess_agent.tool
def get_legal_moves(ctx: RunContext[ChessAgentDeps]) -> list[str]:
    """Get all legal moves in the current position in SAN (Standard Algebraic Notation)."""
    board = ctx.deps.board
    return [board.san(move) for move in board.legal_moves]


@chess_agent.tool
def analyze_position(ctx: RunContext[ChessAgentDeps], num_lines: int = 3, depth: int = 18) -> str:
    """
    Analyze the current position using Stockfish engine.

    Args:
        num_lines: Number of principal variations to analyze (default 3)
        depth: Search depth in plies (default 18)

    Returns:
        Formatted analysis with scores and variations
    """
    board = ctx.deps.board
    engine = ctx.deps.engine
    limit = chess.engine.Limit(depth=depth)

    variations = analyze_variations(board, engine, num_lines, limit)

    result_lines = [f"Analysis of position (depth {depth}):"]
    for i, pv in enumerate(variations, 1):
        score_str = f"{pv.score:+.2f}" if pv.score is not None else "N/A"
        moves_str = " ".join(pv.san_moves[:6])  # Show first 6 moves
        result_lines.append(f"{i}. {score_str}: {moves_str}")

    return "\n".join(result_lines)


@chess_agent.tool
def evaluate_candidate_moves(
    ctx: RunContext[ChessAgentDeps], moves: list[str], depth: int = 18
) -> str:
    """
    Evaluate specific candidate moves using the engine.

    Args:
        moves: List of moves in SAN notation to evaluate (e.g. ["Nf3", "e4", "d4"])
        depth: Search depth in plies (default 18)

    Returns:
        Formatted evaluation showing score for each candidate move
    """
    board = ctx.deps.board
    engine = ctx.deps.engine
    limit = chess.engine.Limit(depth=depth)

    # Convert SAN to UCI moves
    uci_moves = []
    for san_move in moves:
        try:
            move = board.parse_san(san_move)
            uci_moves.append(move)
        except ValueError:
            return f"Invalid move: {san_move}"

    candidates = analyze_moves(board, engine, uci_moves, limit)

    result_lines = ["Candidate move evaluations:"]
    for candidate in candidates:
        score_str = f"{candidate.score:+.2f}" if candidate.score is not None else "N/A"
        result_lines.append(f"  {candidate.san_move}: {score_str}")

    return "\n".join(result_lines)


@chess_agent.tool
def make_move(ctx: RunContext[ChessAgentDeps], move: str) -> str:
    """
    Make a move on the board.

    Args:
        move: Move in SAN notation (e.g. "Nf3", "e4", "O-O")

    Returns:
        Confirmation message with new position FEN
    """
    board = ctx.deps.board

    try:
        chess_move = board.parse_san(move)
        board.push(chess_move)
        return f"Made move {move}. New position: {board.fen()}"
    except ValueError:
        return f"Invalid move: {move}. Legal moves are: {', '.join(get_legal_moves(ctx))}"


@chess_agent.tool
def undo_last_move(ctx: RunContext[ChessAgentDeps]) -> str:
    """Undo the last move made on the board."""
    board = ctx.deps.board

    if len(board.move_stack) == 0:
        return "No moves to undo - board is at starting position"

    last_move = board.pop()
    return f"Undid move {board.san(last_move)}. Position: {board.fen()}"


@chess_agent.tool
def reset_board(ctx: RunContext[ChessAgentDeps], fen: str | None = None) -> str:
    """
    Reset the board to starting position or a specific FEN.

    Args:
        fen: Optional FEN string. If not provided, resets to starting position.

    Returns:
        Confirmation message
    """
    board = ctx.deps.board

    if fen:
        try:
            board.set_fen(fen)
            return f"Board set to FEN: {fen}"
        except ValueError:
            return f"Invalid FEN: {fen}"
    else:
        board.reset()
        return "Board reset to starting position"


def create_chess_agent(fen: str | None = None, engine_depth: int = 20) -> tuple[Agent, ChessAgentDeps]:
    """
    Create a chess agent with initialized board and engine.

    Args:
        fen: Starting position (default: standard starting position)
        engine_depth: Default engine depth for analysis

    Returns:
        Tuple of (agent, dependencies)
    """
    board = chess.Board(fen) if fen else chess.Board()
    config = EngineConfig.stockfish(depth=engine_depth, num_lines=3)
    engine = config.instantiate()

    deps = ChessAgentDeps(board=board, engine=engine)
    return chess_agent, deps
