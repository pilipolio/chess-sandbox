"""
Dataset loading utilities for concept probe training and evaluation.

Provides functions to load and filter labeled chess positions from both
HuggingFace Hub and local JSONL files.
"""

import json
import random
from pathlib import Path

from huggingface_hub import hf_hub_download

from ..labelling.labeller import LabelledPosition


def load_and_filter_positions(
    data_path: Path, random_state: int = 42
) -> tuple[list[LabelledPosition], list[list[str]]]:
    """
    Load and balance labeled positions from JSONL file.

    Reads positions from a JSONL file and creates a balanced dataset with 50/50
    ratio of positions with and without validated concepts. Positions without
    concepts are randomly undersampled to match the count of positions with concepts.

    Args:
        data_path: Path to JSONL file with labeled positions
        random_state: Random seed for reproducible undersampling

    Returns:
        Tuple of (positions, labels):
            - positions: List of LabelledPosition objects (balanced with/without concepts)
            - labels: List of concept name lists, one per position (empty list for positions without concepts)
    """
    random.seed(random_state)

    all_positions: list[LabelledPosition] = []
    with data_path.open() as f:
        for line in f:
            all_positions.append(LabelledPosition.from_dict(json.loads(line)))

    # Separate positions with and without validated concepts
    positions_with_concepts: list[LabelledPosition] = []
    labels_with_concepts: list[list[str]] = []
    positions_without_concepts: list[LabelledPosition] = []

    for pos in all_positions:
        if pos.concepts:
            validated_concepts = [c.name for c in pos.concepts if c.validated_by is not None]
            if validated_concepts:
                positions_with_concepts.append(pos)
                labels_with_concepts.append(validated_concepts)
            else:
                positions_without_concepts.append(pos)
        else:
            positions_without_concepts.append(pos)

    print(f"Loaded {len(all_positions)} positions:")
    print(f"  - {len(positions_with_concepts)} with validated concepts")
    print(f"  - {len(positions_without_concepts)} without validated concepts")

    # Undersample positions without concepts to match positions with concepts (50/50 balance)
    target_count = len(positions_with_concepts)
    if len(positions_without_concepts) > target_count:
        sampled_without_concepts = random.sample(positions_without_concepts, target_count)
        print(f"Undersampled to {target_count} positions without concepts for 50/50 balance")
    else:
        sampled_without_concepts = positions_without_concepts
        print(
            f"Warning: Only {len(positions_without_concepts)} positions without concepts "
            f"(less than {target_count} with concepts)"
        )

    # Combine and shuffle
    positions = positions_with_concepts + sampled_without_concepts
    labels = labels_with_concepts + [[] for _ in sampled_without_concepts]

    # Shuffle together while maintaining position-label correspondence
    combined = list(zip(positions, labels))
    random.shuffle(combined)
    positions, labels = zip(*combined) if combined else ([], [])

    print(f"Final balanced dataset: {len(positions)} positions (50/50 with/without concepts)")
    return list(positions), list(labels)


def load_dataset_from_hf(
    repo_id: str, filename: str, revision: str | None = None, random_state: int = 42
) -> tuple[list[LabelledPosition], list[list[str]]]:
    """
    Load dataset from HuggingFace Hub.

    Downloads JSONL file from HF dataset repo and parses it.

    Args:
        repo_id: HuggingFace dataset repository ID
        filename: JSONL filename in the repo
        revision: Git revision (tag, branch, commit). Defaults to "main"
        random_state: Random seed for reproducible undersampling

    Returns:
        Tuple of (positions, labels):
            - positions: List of LabelledPosition objects (balanced with/without concepts)
            - labels: List of concept name lists, one per position (empty list for positions without concepts)
    """
    data_path = Path(hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", revision=revision))
    return load_and_filter_positions(data_path, random_state=random_state)
