"""Debug utilities for GRPO training pipeline.

Standalone script to sample model generations and diagnose issues like:
- Truncation (missing </think> tags)
- EOS token behavior
- Reward function parsing failures
"""

from typing import Any

import click
import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from chess_sandbox.puzzles_trainer.grpo_rewards import compute_single_reward
from chess_sandbox.puzzles_trainer.reasoning_verifier import extract_solution_section


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sample_generations(
    model_id: str,
    dataset_id: str,
    num_samples: int = 5,
    max_completion_length: int = 1024,
    use_4bit: bool = False,
) -> list[dict[str, Any]]:
    """Sample generations and compute diagnostic info."""
    device = get_device()
    print(f"Using device: {device}")
    print(f"Loading model: {model_id}")

    model_kwargs: dict[str, Any] = {"torch_dtype": torch.float16}
    if use_4bit and device == "cuda":
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        print("Using 4-bit quantization")

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"Tokenizer EOS: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")

    print(f"Loading dataset: {dataset_id}")
    dataset: Dataset = load_dataset(dataset_id, split="train")  # pyright: ignore[reportAssignmentType]
    if "question" in dataset.column_names:
        dataset = dataset.rename_column("question", "prompt")

    results = []
    for i in range(min(num_samples, len(dataset))):
        example = dataset[i]
        prompt = example["prompt"]

        chat_messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_completion_length,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_len:]
        completion_clean = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        has_eos = tokenizer.eos_token_id in generated_tokens.tolist()
        has_think_close = "</think>" in completion_clean
        solution = extract_solution_section(completion_clean)
        reward = compute_single_reward(completion_clean, example["fen"], example["first_move"])

        results.append(
            {
                "index": i,
                "fen": example["fen"],
                "first_move": example["first_move"],
                "completion_length": len(generated_tokens),
                "has_eos": has_eos,
                "has_think_close": has_think_close,
                "solution_extracted": solution is not None,
                "solution": solution,
                "reward": reward,
                "completion": completion_clean,
            }
        )

        print(f"  Sample {i}: {len(generated_tokens)} tokens, EOS={has_eos}, reward={reward:.2f}")

    return results


@click.command("grpo-debug")
@click.option("--model-id", type=str, required=True, help="Model to test")
@click.option(
    "--dataset-id",
    type=str,
    default="pilipolio/chess-reasoning-traces",
    help="Dataset to sample from",
)
@click.option("--num-samples", type=int, default=5, help="Number of samples to generate")
@click.option(
    "--max-completion-length",
    type=int,
    default=1024,
    help="Max tokens to generate (default: 1024)",
)
@click.option("--use-4bit", is_flag=True, help="Use 4-bit quantization (CUDA only)")
@click.option("--verbose", "-v", is_flag=True, help="Show full completions")
def main(
    model_id: str,
    dataset_id: str,
    num_samples: int,
    max_completion_length: int,
    use_4bit: bool,
    verbose: bool,
) -> None:
    """Debug GRPO generations to diagnose truncation/reward issues."""
    print(f"Testing model: {model_id}")
    print(f"Max completion length: {max_completion_length}")
    print("-" * 60)

    results = sample_generations(
        model_id=model_id,
        dataset_id=dataset_id,
        num_samples=num_samples,
        max_completion_length=max_completion_length,
        use_4bit=use_4bit,
    )

    eos_count = sum(1 for r in results if r["has_eos"])
    think_close_count = sum(1 for r in results if r["has_think_close"])
    solution_count = sum(1 for r in results if r["solution_extracted"])
    avg_reward = sum(r["reward"] for r in results) / len(results)
    avg_length = sum(r["completion_length"] for r in results) / len(results)

    print("\n" + "=" * 60)
    print(f"Summary ({len(results)} samples):")
    print(f"  EOS emitted:        {eos_count}/{len(results)}")
    print(f"  Has </think>:       {think_close_count}/{len(results)}")
    print(f"  Solution extracted: {solution_count}/{len(results)}")
    print(f"  Average reward:     {avg_reward:.2f}")
    print(f"  Average length:     {avg_length:.0f} tokens")
    print("=" * 60)

    if verbose:
        for r in results:
            print(f"\n[Sample {r['index']}]")
            print(f"  FEN: {r['fen']}")
            print(f"  Expected: {r['first_move']}")
            print(f"  Length: {r['completion_length']} tokens")
            print(f"  EOS: {r['has_eos']}, </think>: {r['has_think_close']}")
            print(f"  Reward: {r['reward']:.2f}")
            if r["solution"]:
                print(f"  Solution: {r['solution']}")
            print(f"  Completion:\n{r['completion']}")
            print("-" * 40)
    else:
        for r in results:
            status = "OK" if r["reward"] > -0.5 else "FAIL"
            tail = r["completion"][-100:].replace("\n", " ")
            print(f"\n[{status}] Sample {r['index']}: reward={r['reward']:.2f}")
            print(f"  ...{tail}")


if __name__ == "__main__":
    main()
