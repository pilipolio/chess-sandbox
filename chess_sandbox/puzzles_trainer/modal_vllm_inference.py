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

BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
LORA_ADAPTER = "pilipolio/chess-puzzle-sft-qwen3-4b"
VLLM_PORT = 8000

MINUTES = 60

# Use NVIDIA CUDA base image as recommended by Modal docs
vllm_image = (
    modal.Image.from_registry(  # type: ignore[reportUnknownMemberType]
        "nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .uv_pip_install("vllm==0.11.2", "huggingface-hub==0.36.0")  # type: ignore[reportUnknownMemberType]
)

# Persistent volumes for model caching (avoids re-downloading on cold starts)
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)  # type: ignore[reportUnknownMemberType]
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)  # type: ignore[reportUnknownMemberType]

app = modal.App("chess-puzzle-vllm", image=vllm_image)


@app.function(  # type: ignore[reportUnknownMemberType]
    gpu="A10G",
    scaledown_window=5 * MINUTES,
    timeout=10 * MINUTES,
    secrets=[modal.Secret.from_name("huggingface-read-write-secret")],  # type: ignore[reportUnknownMemberType]
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
@modal.web_server(port=VLLM_PORT, startup_timeout=5 * MINUTES)  # type: ignore[reportUnknownMemberType]
def serve():
    """Serve vLLM with LoRA adapter via OpenAI-compatible API."""
    from huggingface_hub import snapshot_download  # pyright: ignore[reportUnknownVariableType]

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
        "--enforce-eager",  # Faster cold starts (disables CUDA graph capture)
    ]

    subprocess.Popen(cmd)
