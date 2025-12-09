"""Modal app for chess puzzle SFT training with GPU.

Usage:
    modal run chess_sandbox/puzzles_trainer/modal_pipeline.py
    modal run chess_sandbox/puzzles_trainer/modal_pipeline.py::train -- --model Qwen/Qwen3-4B --use-4bit

Setup: Requires HuggingFace token configured as Modal secret 'huggingface-read-write-secret'
"""

import subprocess

import modal

image = (
    modal.Image.from_registry("huggingface/trl:0.25.1")  # type: ignore[reportUnknownMemberType]
    .pip_install("python-chess", "wandb==0.23.0", "click", "datasets")
    .add_local_python_source("chess_sandbox")
)

app = modal.App(name="chess-puzzle-sft", image=image)


@app.function()  # type: ignore
def pip_freeze():
    """Run pip freeze to show installed packages."""
    result = subprocess.run(["pip", "freeze"], capture_output=True, text=True, check=True)
    print(result.stdout)


@app.function(  # type: ignore
    gpu="A10G",
    timeout=14400,  # 4 hours
    secrets=[modal.Secret.from_name("huggingface-read-write-secret"), modal.Secret.from_name("wandb")],  # type: ignore
)
def train(*arglist: str):
    """Train chess puzzle SFT model via Modal subprocess invocation.

    Args:
        arglist: CLI arguments passed to puzzles_trainer CLI
            --model: HuggingFace model ID (e.g., Qwen/Qwen3-0.6B)
            --use-4bit: Use 4-bit quantization
            --max-steps: Max training steps
            --output-dir: Output directory

    Raises:
        RuntimeError: If training fails (non-zero exit code)
    """
    cmd = ["python", "-m", "chess_sandbox.puzzles_trainer.trainer", "train", *arglist]
    subprocess.run(cmd, check=True)


@app.function(  # type: ignore
    gpu="A10G",
    timeout=14400,  # 4 hours
    secrets=[modal.Secret.from_name("huggingface-read-write-secret"), modal.Secret.from_name("wandb")],  # type: ignore
)
def train_reasoning(*arglist: str):
    """Train chess reasoning SFT model via Modal subprocess invocation.

    Args:
        arglist: CLI arguments passed to reasoning_trainer CLI
            --model-id: HuggingFace model ID (e.g., Qwen/Qwen3-0.6B)
            --use-4bit: Use 4-bit quantization
            --max-steps: Max training steps
            --wandb-project: W&B project name

    Raises:
        RuntimeError: If training fails (non-zero exit code)
    """
    cmd = ["python", "-m", "chess_sandbox.puzzles_trainer.reasoning_trainer", *arglist]
    subprocess.run(cmd, check=True)
