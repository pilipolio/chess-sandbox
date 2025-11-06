"""Modal endpoints for chess position analysis."""

import chess.engine
import modal

from chess_sandbox.engine.analyse import EngineConfig
from chess_sandbox.engine.position_analysis import PositionAnalysis, analyze_position

image = (
    modal.Image.debian_slim()
    .apt_install("stockfish")
    .uv_sync(uv_project_dir="./", frozen=True)
    .uv_pip_install("fastapi[standard]")
    .add_local_python_source("chess_sandbox")
    .add_local_file(".env.modal", "/root/.env")
)

app = modal.App(name="chess-analysis", image=image)


_stockfish_engine: chess.engine.SimpleEngine | None = None


def get_stockfish_engine() -> chess.engine.SimpleEngine:
    global _stockfish_engine
    if _stockfish_engine is None:
        print("Initializing Stockfish engine...")
        config = EngineConfig.stockfish()
        _stockfish_engine = config.instantiate()
        print("Stockfish engine initialized successfully")
    return _stockfish_engine


@app.function()  # type: ignore
@modal.fastapi_endpoint(method="GET")  # type: ignore
def analyze(fen: str, depth: int = 20, num_lines: int = 5) -> PositionAnalysis:
    engine = get_stockfish_engine()
    return analyze_position(
        fen=fen,
        stockfish_engine=engine,
        depth=depth,
        num_lines=num_lines,
    )
