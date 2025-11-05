"""Modal endpoints for chess position analysis."""

import modal

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


@app.function()  # type: ignore
@modal.fastapi_endpoint(method="GET")  # type: ignore
def analyze(fen: str, depth: int = 20, num_lines: int = 5) -> PositionAnalysis:
    return analyze_position(fen=fen, depth=depth, num_lines=num_lines)
