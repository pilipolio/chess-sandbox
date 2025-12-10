# vLLM Reasoning + Structured Outputs

vLLM supports combining reasoning/thinking mode with structured outputs for models like Qwen3.

## How It Works

The structured output engine **skips format enforcement when reasoning tokens are detected**. This allows:

1. Model generates `<think>...</think>` reasoning freely
2. Structured output constraints apply only to content after `</think>`
3. Both reasoning trace and structured data are returned in separate response fields

## Server Configuration

Enable reasoning parser when starting vLLM:

```bash
--reasoning-parser qwen3
--max-model-len 4096  # Increase for longer reasoning traces
```

## Response Fields

| Field | Description |
|-------|-------------|
| `message.reasoning` | Thinking content from `<think>` tags |
| `message.reasoning_content` | Same as reasoning (alias) |
| `message.content` | Final output text (after `</think>`) |
| `message.parsed` | Pydantic model instance (if using `response_format`) |

## Python Client Usage

```python
from openai import OpenAI

client = OpenAI(base_url="https://your-vllm-endpoint/v1", api_key="dummy")
response = client.chat.completions.create(
    model="chess-reasoning",
    messages=[{"role": "user", "content": "..."}],
    max_tokens=2048,
)

# Access reasoning trace
reasoning = getattr(response.choices[0].message, "reasoning", None)
# Or use the alias
reasoning = getattr(response.choices[0].message, "reasoning_content", None)
# Final answer
content = response.choices[0].message.content
```

## Supported Models

| Model | Parser Flag |
|-------|-------------|
| Qwen3 series | `--reasoning-parser qwen3` |
| DeepSeek R1 | `--reasoning-parser deepseek_r1` |
| DeepSeek V3 | `--reasoning-parser deepseek_v3` |
| GLM-4.5 | `--reasoning-parser glm45` |

## Implementation

See `chess_sandbox/puzzles_trainer/`:
- `modal_reasoning_vllm.py` - vLLM server with reasoning parser
- `reasoning_evaluation.py` - Evaluation with `PlainReasoningModel` that captures reasoning

## References

- [vLLM Reasoning Outputs](https://docs.vllm.ai/en/latest/features/reasoning_outputs.html)
- [vLLM Structured Outputs](https://docs.vllm.ai/en/v0.8.2/features/structured_outputs.html)
- [Qwen3 Thinking Mode](https://qwenlm.github.io/blog/qwen3/)
