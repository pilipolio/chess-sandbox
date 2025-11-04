"""
Dataset loading utilities for concept probe training and evaluation.

Provides functions to load and filter labeled chess positions from both
HuggingFace Hub and local JSONL files.
"""

import json
from pathlib import Path

from huggingface_hub import hf_hub_download

from ..labelling.labeller import LabelledPosition


def load_and_filter_positions(data_path: Path) -> tuple[list[LabelledPosition], list[list[str]]]:
    """
    Load and filter labeled positions from JSONL file.

    Reads positions from a JSONL file and filters to only include positions
    with validated concepts. This is shared logic used by both training and
    evaluation workflows.

    Args:
        data_path: Path to JSONL file with labeled positions

    Returns:
        Tuple of (positions, labels):
            - positions: List of LabelledPosition objects with validated concepts
            - labels: List of concept name lists, one per position
    """
    all_positions: list[LabelledPosition] = []
    with data_path.open() as f:
        for line in f:
            all_positions.append(LabelledPosition.from_dict(json.loads(line)))

    positions_with_concepts = [p for p in all_positions if p.concepts]
    print(f"Loaded {len(all_positions)} positions, kept {len(positions_with_concepts)} with concepts")

    positions: list[LabelledPosition] = []
    labels: list[list[str]] = []

    for pos in positions_with_concepts:
        validated_concepts = [c.name for c in pos.concepts if c.validated_by is not None]
        if validated_concepts:
            positions.append(pos)
            labels.append(validated_concepts)

    print(f"Kept {len(positions)} positions with at least one validated concept")
    return positions, labels


def load_dataset_from_hf(
    repo_id: str, filename: str, revision: str | None = None
) -> tuple[list[LabelledPosition], list[list[str]]]:
    """
    Load dataset from HuggingFace Hub.

    Downloads JSONL file from HF dataset repo and parses it.

    Args:
        repo_id: HuggingFace dataset repository ID
        filename: JSONL filename in the repo
        revision: Git revision (tag, branch, commit). Defaults to "main"

    Returns:
        Tuple of (positions, labels):
            - positions: List of LabelledPosition objects with validated concepts
            - labels: List of concept name lists, one per position
    """
    data_path = Path(hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", revision=revision))
    return load_and_filter_positions(data_path)
