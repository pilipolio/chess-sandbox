"""Modal deployment for GRPO debug script."""

import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch",
    "transformers",
    "datasets",
    "accelerate",
    "click",
    "chess",
    "peft",
)

app = modal.App("chess-grpo-debug")

# Map adapter IDs to their base models
ADAPTER_BASE_MODELS = {
    "pilipolio/chess-reasoning-sft-qwen3-4b": "Qwen/Qwen3-4B",
    "pilipolio/chess-reasoning-sft-qwen3-0.6b": "Qwen/Qwen3-0.6B",
}


@app.function(
    image=image,
    gpu="A10G",
    timeout=600,
)
def run_debug(model_id: str, num_samples: int = 3, max_completion_length: int = 1024):
    """Run GRPO debug on Modal GPU."""
    import re

    import torch
    from datasets import load_dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Testing model: {model_id}")
    print(f"Max completion length: {max_completion_length}")
    print("-" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Check if this is a LoRA adapter that needs base model
    if model_id in ADAPTER_BASE_MODELS:
        base_model_id = ADAPTER_BASE_MODELS[model_id]
        print(f"Loading base model: {base_model_id}")
        base_model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=torch.float16, device_map="auto")
        print(f"Loading adapter: {model_id}")
        model = PeftModel.from_pretrained(base_model, model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_id)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"Tokenizer EOS: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")

    dataset = load_dataset("pilipolio/chess-reasoning-traces", split="train")
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
        completion = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        has_eos = tokenizer.eos_token_id in generated_tokens.tolist()
        has_think_close = "</think>" in completion

        # Simple reward check
        solution_match = re.search(r"</think>\s*\n?\s*(.+?)$", completion, re.DOTALL)
        solution = solution_match.group(1).strip() if solution_match else None

        # Extract first move from solution
        first_move = None
        if solution:
            move_match = re.search(
                r"([KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?|O-O(?:-O)?)",
                solution,
            )
            if move_match:
                first_move = move_match.group(1)

        # Check if first move is correct
        expected = example["first_move"]
        correct = first_move and first_move.lower().rstrip("+#") == expected.lower().rstrip("+#")

        result = {
            "index": i,
            "fen": example["fen"],
            "expected": expected,
            "extracted": first_move,
            "correct": correct,
            "length": len(generated_tokens),
            "has_eos": has_eos,
            "has_think_close": has_think_close,
            "completion": completion,
        }
        results.append(result)

        print(f"\n[Sample {i}]")
        print(f"  FEN: {example['fen'][:50]}...")
        print(f"  Expected: {expected}, Extracted: {first_move}, Correct: {correct}")
        print(f"  Length: {len(generated_tokens)} tokens, EOS: {has_eos}, </think>: {has_think_close}")

        # Show completion tail
        tail = completion[-300:] if len(completion) > 300 else completion
        print(f"  Tail: ...{tail}")

    # Summary
    eos_count = sum(1 for r in results if r["has_eos"])
    think_count = sum(1 for r in results if r["has_think_close"])
    correct_count = sum(1 for r in results if r["correct"])

    print("\n" + "=" * 60)
    print(f"Summary ({len(results)} samples):")
    print(f"  EOS emitted:     {eos_count}/{len(results)}")
    print(f"  Has </think>:    {think_count}/{len(results)}")
    print(f"  Correct move:    {correct_count}/{len(results)}")
    print("=" * 60)

    return results


@app.local_entrypoint()
def main(
    model_id: str = "pilipolio/chess-reasoning-sft-qwen3-4b",
    num_samples: int = 5,
    max_completion_length: int = 1024,
):
    """Run GRPO debug from CLI."""
    results = run_debug.remote(model_id, num_samples, max_completion_length)
    print("\nResults returned from Modal:")
    for r in results:
        status = "OK" if r["correct"] else "FAIL"
        print(f"  [{status}] Sample {r['index']}: expected={r['expected']}, got={r['extracted']}")
