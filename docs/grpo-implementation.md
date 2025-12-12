# GRPO Pipeline Implementation

## Status: MVP Complete

Implementation of GRPO (Group Relative Policy Optimization) training pipeline for chess reasoning models.

---

## Files Created

```
chess_sandbox/puzzles_trainer/
├── grpo_rewards.py      # Reward functions (4-component: legality, correctness, format, piece accuracy)
├── grpo_trainer.py      # GRPOTrainer wrapper + Click CLI
├── modal_grpo.py        # Modal deployment (A10G, 8hr timeout)

pyproject.toml           # Entry point: grpo-trainer
```

---

## Usage

### CLI Options
```bash
uv run grpo-trainer --help

Options:
  --model-id TEXT              Model to fine-tune (default: Qwen/Qwen3-0.6B)
  --dataset-id TEXT            HuggingFace dataset (default: pilipolio/chess-reasoning-traces)
  --output-model-id TEXT       Hub model ID for output
  --use-4bit                   Use 4-bit quantization (CUDA only)
  --use-vllm                   Use vLLM for accelerated generation
  --vllm-gpu-memory FLOAT      vLLM GPU memory utilization (default: 0.4)
  --num-generations INTEGER    Completions per prompt (default: 8)
  --max-completion-length INT  Max tokens to generate (default: 512)
  --max-steps INTEGER          Max training steps (for testing)
  --wandb-project TEXT         W&B project name
  --wandb-run-name TEXT        W&B run name
```

### Modal Deployment
```bash
# Quick test (10 steps)
modal run chess_sandbox/puzzles_trainer/modal_grpo.py::train_grpo -- \
    --model-id Qwen/Qwen3-0.6B --max-steps 10 --use-4bit --wandb-project chess-grpo-test

# With vLLM acceleration
modal run chess_sandbox/puzzles_trainer/modal_grpo.py::train_grpo -- \
    --model-id Qwen/Qwen3-0.6B --use-vllm --max-steps 50 --wandb-project chess-grpo-test

# Full training (detached)
modal run --detach chess_sandbox/puzzles_trainer/modal_grpo.py::train_grpo -- \
    --model-id pilipolio/chess-reasoning-sft-qwen3-4b --use-vllm --wandb-project chess-grpo
```

---

## Reward Function

From `grpo_rewards.py`:

| Component | Weight | Signal |
|-----------|--------|--------|
| Legality | 40% | First move must be legal (-1.0 if illegal) |
| Correctness | 40% | First move matches puzzle solution |
| Format | 15% | 5 reasoning sections present in `<think>` |
| Piece accuracy | 5% | Board awareness from Step 2 parsing |

Reuses existing verification functions from `reasoning_verifier.py`.

---

## GRPOConfig Defaults

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `num_generations` | 8 | Sample diversity vs compute |
| `beta` | 0.0 | KL disabled per recent best practice |
| `learning_rate` | 1e-6 | Lower than SFT for RL stability |
| `per_device_train_batch_size` | 1 | Low because 8 completions per prompt |
| `gradient_accumulation_steps` | 8 | Effective batch = 64 completions |
| `max_completion_length` | 512 | Room for thinking + solution |
| `vllm_mode` | colocate | Share GPU memory on A10G |
| `vllm_gpu_memory_utilization` | 0.4 | Leave 60% for training |

---

## Next Steps

1. **Test on Modal** - Run with vLLM on A10G GPU
2. **Monitor training** - Check reward curves in W&B
3. **Evaluate** - Compare GRPO vs SFT on held-out puzzles using `reasoning_evaluation.py`
4. **Scale up** - Move to SFT checkpoint once validated on base model

---

## Potential Issues & Mitigations

| Issue | Mitigation |
|-------|------------|
| OOM with vLLM | Reduce `vllm_gpu_memory_utilization` to 0.3 |
| High reward variance | Increase `num_generations` to 16 |
| Training instability | Add KL penalty (`beta=0.04`) |
| Reward hacking | Add length penalty in reward function |
