"""Modal app for chess puzzle SFT training with GPU.

Usage:
    modal run chess_sandbox/puzzles_trainer/modal_pipeline.py
    modal run chess_sandbox/puzzles_trainer/modal_pipeline.py::train --model qwen3-4b --use-4bit

Setup: Requires HuggingFace token configured as Modal secret 'huggingface-read-write-secret'
"""

import subprocess

import modal

from chess_sandbox.git import get_commit_sha

image = (
    modal.Image.debian_slim()
    .env({"UV_INDEX_URL": "https://download.pytorch.org/whl/cu124"})
    .uv_sync(uv_project_dir="./", frozen=True)
    .add_local_python_source("chess_sandbox")
)

app = modal.App(name="chess-puzzle-sft", image=image)


@app.function(  # type: ignore
    gpu="A10G",
    timeout=7200,  # 2 hours
    secrets=[modal.Secret.from_name("huggingface-read-write-secret")],  # type: ignore
    env={"GIT_COMMIT": get_commit_sha()},
)
def train(*arglist: str):
    """Train chess puzzle SFT model via Modal subprocess invocation.

    Args:
        arglist: CLI arguments passed to puzzles_trainer CLI
            --model: Model to fine-tune (qwen3-0.6b, qwen3-4b)
            --use-4bit: Use 4-bit quantization
            --max-steps: Max training steps
            --output-dir: Output directory

    Raises:
        RuntimeError: If training fails (non-zero exit code)
    """
    cmd = ["uv", "run", "python", "-m", "chess_sandbox.puzzles_trainer.cli", *arglist]
    subprocess.run(cmd, check=True)
