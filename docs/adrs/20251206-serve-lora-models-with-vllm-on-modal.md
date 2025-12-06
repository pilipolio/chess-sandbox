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

```python
# modal_vllm_inference.py
@app.function(gpu="A10G", scaledown_window=300)
@modal.web_server(port=8000, startup_timeout=180)
def serve():
    lora_path = snapshot_download(repo_id="pilipolio/chess-puzzle-sft-qwen3-4b")
    subprocess.Popen([
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", "Qwen/Qwen3-4B-Instruct-2507",
        "--enable-lora",
        "--lora-modules", f"chess-puzzle={lora_path}",
    ])
```

### Consequences

* Good, because vLLM handles LoRA adapters natively without merging overhead
* Good, because existing evaluation code works with minimal changes (add `--base-url`)
* Good, because Modal's GPU serverless matches our training infrastructure
* Good, because TRL's vLLM integration (for future GRPO) uses same patterns
* Bad, because cold start takes ~2-3 minutes (model download + vLLM init)
* Bad, because A10G may be oversized for 4B model (but needed for vLLM overhead)

## Related Decisions

* [20251029-use-modal-for-serverless-endpoints.md](20251029-use-modal-for-serverless-endpoints.md) - Modal as platform choice
* [20251101-use-huggingface-hub-for-versioning.md](20251101-use-huggingface-hub-for-versioning.md) - HF Hub for model storage

## Notes

TRL's vLLM integration (`vllm_integration.py`) is **training-only** - designed for online generation during GRPO/DPO training, not inference serving. For inference, use vLLM's OpenAI-compatible server directly.
