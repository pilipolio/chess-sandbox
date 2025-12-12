"""Test EOS token behavior across models.

Verifies that models properly terminate generations with EOS tokens
rather than always hitting max_length.
"""

import click
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def check_tokenizer_config(model_id: str) -> dict[str, object]:
    """Check tokenizer special token configuration."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    return {
        "model_id": model_id,
        "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token": tokenizer.pad_token,
        "pad_token_id": tokenizer.pad_token_id,
        "pad_equals_eos": tokenizer.pad_token_id == tokenizer.eos_token_id,
    }


def test_natural_termination(
    model_id: str,
    test_prompts: list[str],
    max_new_tokens: int = 256,
) -> list[dict[str, object]]:
    """Test if model naturally terminates or always hits max length."""
    device = get_device()
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = []
    for prompt in test_prompts:
        chat_messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        generated = outputs[0][input_len:]

        has_eos = tokenizer.eos_token_id in generated.tolist()
        hit_max = len(generated) >= max_new_tokens

        results.append(
            {
                "prompt": prompt[:50] + "..." if len(prompt) > 50 else prompt,
                "generated_tokens": len(generated),
                "max_tokens": max_new_tokens,
                "has_eos": has_eos,
                "hit_max_length": hit_max,
                "natural_termination": has_eos and not hit_max,
            }
        )

    return results


@click.command("test-eos")
@click.option(
    "--base-model",
    type=str,
    default="Qwen/Qwen3-0.6B",
    help="Base model to test",
)
@click.option("--sft-model", type=str, default=None, help="SFT checkpoint to compare")
@click.option("--max-new-tokens", type=int, default=256, help="Max tokens to generate")
def main(base_model: str, sft_model: str | None, max_new_tokens: int) -> None:
    """Test EOS token behavior for base and SFT models."""
    test_prompts = [
        "What is 2+2?",
        "Say hello.",
        "Explain chess briefly.",
    ]

    models_to_test = [base_model]
    if sft_model:
        models_to_test.append(sft_model)

    for model_id in models_to_test:
        print(f"\n{'='*60}")
        print(f"Testing: {model_id}")
        print("=" * 60)

        config = check_tokenizer_config(model_id)
        print("\nTokenizer Config:")
        print(f"  EOS token: {config['eos_token']} (id: {config['eos_token_id']})")
        print(f"  PAD token: {config['pad_token']} (id: {config['pad_token_id']})")
        print(f"  PAD == EOS: {config['pad_equals_eos']}")

        print(f"\nGeneration Tests (max_new_tokens={max_new_tokens}):")
        results = test_natural_termination(model_id, test_prompts, max_new_tokens)

        natural_term_count = sum(1 for r in results if r["natural_termination"])
        print(f"  Natural termination: {natural_term_count}/{len(results)}")

        for r in results:
            if r["natural_termination"]:
                status = "OK"
            elif r["hit_max_length"]:
                status = "TRUNCATED"
            else:
                status = "NO_EOS"
            print(f"    [{status}] {r['generated_tokens']}/{r['max_tokens']} tokens - {r['prompt']}")


if __name__ == "__main__":
    main()
