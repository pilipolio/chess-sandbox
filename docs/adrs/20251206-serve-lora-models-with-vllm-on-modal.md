# Serve LoRA Models with vLLM on Modal

## Context and Problem Statement

After SFT training on Modal (see experiment log in chess-llm-finetuning.md), we need to serve the resulting LoRA adapter (`pilipolio/chess-puzzle-sft-qwen3-4b`) for inference. The model is stored on HuggingFace Hub as a LoRA adapter (not merged), requiring both base model loading and adapter application.

## Considered Options

* Modal + vLLM with OpenAI-compatible API
* Modal + transformers/PEFT direct inference
* HuggingFace Inference Endpoints
* Merge LoRA and deploy to OpenRouter

## Decision Outcome

Chosen option: "Modal + vLLM with OpenAI-compatible API", because it provides the best balance of performance, cost, and integration simplicity.

### Key advantages:

1. **vLLM LoRA support**: Serves adapters directly without merging via `--enable-lora` flag
2. **OpenAI-compatible API**: Drop-in replacement for existing `llm_evaluation.py` - just change base URL
3. **Consistent with training pipeline**: Both training and inference use Modal with HF Hub models
4. **Cost-efficient**: Serverless with 5-minute idle timeout, pay only when used
5. **Performance**: vLLM's PagedAttention provides efficient memory usage and batching

### Implementation:

Following [Modal's vLLM inference best practices](https://modal.com/docs/examples/vllm_inference):

```python
# modal_vllm_inference.py

# NVIDIA CUDA base image (Modal provides CUDA drivers)
vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.11.2", "huggingface-hub==0.36.0")
)

# Persistent volumes for model caching
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

@app.function(
    gpu="A10G",
    volumes={"/root/.cache/huggingface": hf_cache_vol, "/root/.cache/vllm": vllm_cache_vol},
)
@modal.web_server(port=8000, startup_timeout=300)
def serve():
    lora_path = snapshot_download(repo_id="pilipolio/chess-puzzle-sft-qwen3-4b")
    subprocess.Popen([
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", "Qwen/Qwen3-4B-Instruct-2507",
        "--enable-lora", "--lora-modules", f"chess-puzzle={lora_path}",
        "--enforce-eager",  # Faster cold starts
    ])
```

**Key patterns from Modal docs:**
- **NVIDIA CUDA base image**: Modal provides CUDA drivers, so use official NVIDIA image
- **Persistent volumes**: Cache HuggingFace and vLLM artifacts across container restarts
- **Version pinning**: Pin vLLM version to avoid breaking changes
- **`--enforce-eager`**: Disable CUDA graph capture for faster startup (trade-off: slightly lower throughput)

### Consequences

* Good, because vLLM handles LoRA adapters natively without merging overhead
* Good, because existing evaluation code works with minimal changes (add `--base-url`)
* Good, because Modal's GPU serverless matches our training infrastructure
* Good, because TRL's vLLM integration (for future GRPO) uses same patterns
* Good, because persistent volumes reduce cold start time after first run
* Bad, because first cold start still takes ~2-3 minutes (initial model download)
* Bad, because A10G may be oversized for 4B model (but needed for vLLM overhead)

## Related Decisions

* [20251029-use-modal-for-serverless-endpoints.md](20251029-use-modal-for-serverless-endpoints.md) - Modal as platform choice
* [20251101-use-huggingface-hub-for-versioning.md](20251101-use-huggingface-hub-for-versioning.md) - HF Hub for model storage

## Notes

TRL's vLLM integration (`vllm_integration.py`) is **training-only** - designed for online generation during GRPO/DPO training, not inference serving. For inference, use vLLM's OpenAI-compatible server directly.
