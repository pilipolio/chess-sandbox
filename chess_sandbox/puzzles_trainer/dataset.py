"""Chess puzzle dataset loading and formatting."""

from datasets import load_dataset

DATASET_ID = "pilipolio/lichess-puzzles-solutions"


def format_puzzle(example: dict) -> dict:
    """Format puzzle as chat messages for SFT."""
    return {
        "messages": [
            {"role": "user", "content": example["question"]},
            {"role": "assistant", "content": example["answer"]},
        ]
    }


def load_puzzle_dataset():
    """Load and format the chess puzzles dataset."""
    print(f"Loading dataset: {DATASET_ID}")
    dataset = load_dataset(DATASET_ID)

    train_dataset = dataset["train"].map(format_puzzle)
    test_dataset = dataset["test"].map(format_puzzle)

    print(f"Train examples: {len(train_dataset)}")
    print(f"Test examples: {len(test_dataset)}")

    return train_dataset, test_dataset
