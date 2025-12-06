"""Modal vLLM inference endpoint for chess puzzle LoRA model.

Deploy:
    modal deploy chess_sandbox/puzzles_trainer/modal_vllm_inference.py

Serve locally (for testing):
    modal serve chess_sandbox/puzzles_trainer/modal_vllm_inference.py

Usage:
    curl https://your-workspace--chess-puzzle-vllm-serve.modal.run/v1/completions \
        -H "Content-Type: application/json" \
        -d '{"model": "chess-puzzle", "prompt": "...", "max_tokens": 256}'
"""

import subprocess

import modal
from huggingface_hub import snapshot_download  # pyright: ignore[reportUnknownVariableType]

BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
LORA_ADAPTER = "pilipolio/chess-puzzle-sft-qwen3-4b"
VLLM_PORT = 8000

image = modal.Image.debian_slim(python_version="3.12").pip_install("vllm>=0.6.0", "huggingface-hub")  # type: ignore[reportUnknownMemberType]

app = modal.App("chess-puzzle-vllm", image=image)


@app.function(  # type: ignore[reportUnknownMemberType]
    gpu="A10G",
    scaledown_window=300,
    timeout=600,
    secrets=[modal.Secret.from_name("huggingface-read-write-secret")],  # type: ignore[reportUnknownMemberType]
)
@modal.web_server(port=VLLM_PORT, startup_timeout=300)  # type: ignore[reportUnknownMemberType]
def serve():
    """Serve vLLM with LoRA adapter via OpenAI-compatible API."""
    lora_path = snapshot_download(repo_id=LORA_ADAPTER)

    cmd = [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        BASE_MODEL,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--max-model-len",
        "2048",
        "--enable-lora",
        "--lora-modules",
        f"chess-puzzle={lora_path}",
        "--max-lora-rank",
        "32",
    ]

    subprocess.Popen(cmd)
