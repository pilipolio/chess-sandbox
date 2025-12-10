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
    .uv_pip_install("vllm==0.11.2", "huggingface-hub==0.36.0", "outlines==1.2.9")
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

## Structured Outputs

vLLM supports structured outputs via the [Outlines library](https://docs.vllm.ai/en/v0.8.2/features/structured_outputs.html). Add `outlines==1.2.9` to the image dependencies.

Use `response_format` with Pydantic models or `guided_json` via `extra_body`.

## Reasoning + Structured Outputs

vLLM supports combining reasoning mode (`<think>` tags) with structured outputs. The structured output engine skips format enforcement when reasoning tokens are detected.

### Server Configuration

```python
cmd = [
    "python", "-m", "vllm.entrypoints.openai.api_server",
    "--model", BASE_MODEL,
    "--enable-lora",
    "--lora-modules", f"chess-reasoning={lora_path}",
    "--max-model-len", "4096",  # Needed for longer reasoning traces
    "--reasoning-parser", "qwen3",  # Enable <think> tag parsing
    "--enforce-eager",
]
```

### Response Fields

- `message.reasoning` - Thinking content from `<think>` tags
- `message.reasoning_content` - Same as reasoning (alias)
- `message.content` - Final output text (after `</think>`)
- `message.parsed` - Pydantic model instance (if using `response_format`)

### Evaluation Integration

The `reasoning_evaluation.py` script includes `PlainReasoningModel` that captures reasoning:

```python
response = client.chat.completions.create(...)
reasoning = getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None)
```

See [vllm-reasoning-and-structured-outputs.md](../vllm-reasoning-and-structured-outputs.md) for details.

**References:**
- [vLLM Reasoning Outputs](https://docs.vllm.ai/en/latest/features/reasoning_outputs.html)
- [vLLM Structured Outputs](https://docs.vllm.ai/en/v0.8.2/features/structured_outputs.html)

## Notes

TRL's vLLM integration (`vllm_integration.py`) is **training-only** - designed for online generation during GRPO/DPO training, not inference serving. For inference, use vLLM's OpenAI-compatible server directly.
