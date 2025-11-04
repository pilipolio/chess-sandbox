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


def load_positions(data_path: Path) -> list[LabelledPosition]:
    all_positions: list[LabelledPosition] = []
    with data_path.open() as f:
        for line in f:
            all_positions.append(LabelledPosition.from_dict(json.loads(line)))
    return all_positions


def rebalance_positions(positions: list[LabelledPosition], random_state: int = 42) -> list[LabelledPosition]:
    """
    Balance labeled positions to achieve 50/50 ratio of positions with/without validated concepts.

    Creates a balanced dataset by undersampling positions without validated concepts
    to match the count of positions with validated concepts.

    Args:
        positions: List of LabelledPosition objects to balance
        random_state: Random seed for reproducible undersampling

    Returns:
        List of balanced LabelledPosition objects. Each position retains its concepts
        field, allowing downstream code to extract labels on-demand.

    Example:
        >>> from ..labelling.labeller import LabelledPosition, Concept
        >>> # Create positions with validated concepts
        >>> pos1 = LabelledPosition(
        ...     fen="fen1", move_number=1, side_to_move="white", comment="", game_id="game1",
        ...     move_san="e4", previous_fen="start1", concepts=[Concept(name="fork", validated_by="user1")]
        ... )
        >>> pos2 = LabelledPosition(
        ...     fen="fen2", move_number=2, side_to_move="black", comment="", game_id="game2",
        ...     move_san="Nf6", previous_fen="start2", concepts=[Concept(name="pin", validated_by="user2")]
        ... )
        >>> # Create positions without validated concepts
        >>> pos3 = LabelledPosition(
        ...     fen="fen3", move_number=3, side_to_move="white", comment="", game_id="game3",
        ...     move_san="d4", previous_fen="start3", concepts=None
        ... )
        >>> pos4 = LabelledPosition(
        ...     fen="fen4", move_number=4, side_to_move="black", comment="", game_id="game4",
        ...     move_san="e5", previous_fen="start4", concepts=[]
        ... )
        >>> pos5 = LabelledPosition(
        ...     fen="fen5", move_number=5, side_to_move="white", comment="", game_id="game5",
        ...     move_san="Nc3", previous_fen="start5", concepts=[Concept(name="skewer", validated_by=None)]
        ... )
        >>> # Balance with more positions without concepts than with
        >>> balanced = rebalance_positions([pos1, pos2, pos3, pos4, pos5], random_state=42)
        Loaded 5 positions:
          - 2 with validated concepts
          - 3 without validated concepts
        Undersampled to 2 positions without concepts for 50/50 balance
        Final balanced dataset: 4 positions (50/50 with/without concepts)
        >>> len(balanced)
        4
        >>> # Count positions with validated concepts
        >>> with_concepts = [p for p in balanced if p.concepts and any(c.validated_by for c in p.concepts)]
        >>> len(with_concepts)
        2
        >>> # Count positions without validated concepts
        >>> without_concepts = [p for p in balanced if not p.concepts or not any(c.validated_by for c in p.concepts)]
        >>> len(without_concepts)
        2
    """
    random.seed(random_state)

    positions_with_concepts: list[LabelledPosition] = []
    positions_without_concepts: list[LabelledPosition] = []

    for pos in positions:
        if pos.concepts:
            validated_concepts = [c for c in pos.concepts if c.validated_by is not None]
            if validated_concepts:
                positions_with_concepts.append(pos)
            else:
                positions_without_concepts.append(pos)
        else:
            positions_without_concepts.append(pos)

    print(f"Loaded {len(positions)} positions:")
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
    all_positions = positions_with_concepts + sampled_without_concepts
    random.shuffle(all_positions)

    print(f"Final balanced dataset: {len(all_positions)} positions (50/50 with/without concepts)")
    return all_positions


def load_dataset_from_hf(
    repo_id: str, filename: str, revision: str | None = None, random_state: int = 42
) -> list[LabelledPosition]:
    """
    Load and balance dataset from HuggingFace Hub.

    Downloads JSONL file from HF dataset repo, parses it, and balances it with
    50/50 ratio of positions with/without validated concepts.

    Args:
        repo_id: HuggingFace dataset repository ID
        filename: JSONL filename in the repo
        revision: Git revision (tag, branch, commit). Defaults to "main"
        random_state: Random seed for reproducible undersampling

    Returns:
        List of balanced LabelledPosition objects. Positions retain their concepts
        field for downstream label extraction.
    """
    data_path = Path(hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", revision=revision))
    positions = load_positions(data_path)
    return rebalance_positions(positions, random_state=random_state)
