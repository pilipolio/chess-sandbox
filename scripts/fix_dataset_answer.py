"""Fix answer field in chess-reasoning-traces dataset.

Replaces the full PGN with {COMMENT} placeholders with just the first move.
Uses existing solution_moves_sans field which already has the correct moves.
"""

from datasets import load_dataset

DATASET_ID = "pilipolio/chess-reasoning-traces"


def fix_answer(example: dict) -> dict:
    """Replace answer's post-</think> content with just first move."""
    answer = example["answer"]
    solution_sans = example.get("solution_moves_sans", [])
    first_move = solution_sans[0] if solution_sans else ""

    # Split at </think> and replace everything after with just first move
    if "</think>" in answer:
        think_part = answer.split("</think>")[0]
        example["answer"] = f"{think_part}</think>\n{first_move}"

    return example


def main():
    print(f"Loading dataset: {DATASET_ID}")
    ds = load_dataset(DATASET_ID)

    print("Applying fix...")
    ds_fixed = ds.map(fix_answer)

    # Preview before push
    print("\n=== Sample fixed answer (last 200 chars) ===")
    print(ds_fixed["train"][0]["answer"][-200:])

    print("\n=== Checking for {COMMENT} in fixed answers ===")
    comment_count = sum(1 for ex in ds_fixed["train"] if "{COMMENT}" in ex["answer"])
    print(f"Examples with {{COMMENT}}: {comment_count}/{len(ds_fixed['train'])}")

    # Uncomment to push
    # ds_fixed.push_to_hub(DATASET_ID)
    # print(f"\nPushed to {DATASET_ID}")


if __name__ == "__main__":
    main()
