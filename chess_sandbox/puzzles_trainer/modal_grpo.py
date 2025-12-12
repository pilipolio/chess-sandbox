"""Modal app for GRPO training with GPU.

Usage:
    modal run chess_sandbox/puzzles_trainer/modal_grpo.py::train_grpo
    modal run chess_sandbox/puzzles_trainer/modal_grpo.py::train_grpo -- --use-vllm --max-steps 100

Setup: Requires HuggingFace token configured as Modal secret 'huggingface-read-write-secret'
"""

import subprocess

import modal

# Image with TRL 0.25.1+ and vLLM support for GRPO
image = (
    modal.Image.from_registry("huggingface/trl:0.25.1")  # type: ignore[reportUnknownMemberType]
    .pip_install(
        "python-chess",
        "wandb==0.23.0",
        "click",
        "datasets",
        "bitsandbytes==0.48.2",
        "vllm==0.10.2",
    )
    .add_local_python_source("chess_sandbox")
)

app = modal.App(name="chess-grpo-training", image=image)


@app.function()  # type: ignore
def pip_freeze():
    """Show installed packages for debugging."""
    result = subprocess.run(["pip", "freeze"], capture_output=True, text=True, check=True)
    print(result.stdout)


@app.function(  # type: ignore
    gpu="A10G",
    timeout=28800,  # 8 hours (GRPO takes longer than SFT)
    secrets=[
        modal.Secret.from_name("huggingface-read-write-secret"),  # type: ignore
        modal.Secret.from_name("wandb"),  # type: ignore
    ],
)
def train_grpo(*arglist: str):
    """Train chess reasoning GRPO model via Modal.

    Args:
        arglist: CLI arguments passed to grpo_trainer
            --model-id: Model to fine-tune (default: Qwen/Qwen3-0.6B)
            --use-4bit: Use 4-bit quantization
            --use-vllm: Enable vLLM acceleration
            --num-generations: Completions per prompt (default: 8)
            --max-steps: Max training steps
            --wandb-project: W&B project name

    Example:
        modal run modal_grpo.py::train_grpo -- --use-vllm --max-steps 100 --wandb-project chess-grpo-test

    Raises:
        RuntimeError: If training fails
    """
    cmd = ["python", "-m", "chess_sandbox.puzzles_trainer.grpo_trainer", *arglist]
    subprocess.run(cmd, check=True)
