import modal

from chess_sandbox.engine.position_analysis import PositionAnalysis, analyze_position

image = (
    modal.Image.debian_slim()
    .apt_install("stockfish")
    .env({"STOCKFISH_PATH": "/usr/games/stockfish"})
    .uv_sync(uv_project_dir="./", frozen=True)
    .uv_pip_install("fastapi[standard]")
    .add_local_python_source("chess_sandbox")
)

app = modal.App(name="chess-analysis", image=image)


@app.function()  # type: ignore
@modal.fastapi_endpoint(method="GET")  # type: ignore
def analyze(fen: str, depth: int = 20, num_lines: int = 5) -> PositionAnalysis:
    """
    Analyze a chess position using Stockfish.

    Args:
        fen: Position in FEN notation (required)
        depth: Stockfish analysis depth (default=20)
        num_lines: Number of principal variations (default=5)

    Returns:
        PositionAnalysis object with structured data including:
        - fen: Input position
        - next_move: Next move if provided
        - principal_variations: List of variations with scores and SAN moves
        - human_moves: Maia human-like moves (if enabled)

    Raises:
        ValueError: Invalid FEN notation
        RuntimeError: Engine analysis error
    """
    return analyze_position(fen=fen, depth=depth, num_lines=num_lines)
