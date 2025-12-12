"""Reward functions for GRPO training of chess reasoning models.

Provides verifiable rewards based on chess move legality, correctness,
and reasoning structure. Compatible with TRL GRPOTrainer.
"""

from __future__ import annotations

from chess_sandbox.puzzles_trainer.reasoning_verifier import (
    extract_pgn_moves,
    extract_piece_positions_section,
    extract_solution_section,
    normalize_move,
    parse_sections,
    validate_move_sequence,
    validate_piece_positions,
)


def compute_single_reward(
    completion: str,
    fen: str,
    expected_first_move: str,
) -> float:
    """Compute reward for a single completion.

    Reward components (from docs/chess-llm-finetuning.md):
    - Legality (40%): -1.0 for illegal first move
    - First move correctness (40%): match puzzle solution
    - Format compliance (15%): 5 reasoning sections present
    - Piece accuracy (5%): board awareness in Step 2

    Args:
        completion: Model output (should contain <think>...</think> and solution)
        fen: FEN position string
        expected_first_move: Expected first move in SAN notation

    Returns:
        Reward in range [-1.0, 1.0]

    >>> compute_single_reward("</think>e4", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "e4")
    0.7
    >>> compute_single_reward("</think>e5", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "e4")
    0.3
    >>> compute_single_reward("</think>Qxh7#", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "e4")
    -1.0
    """
    solution_section = extract_solution_section(completion)
    if not solution_section:
        return -1.0

    pgn_moves = extract_pgn_moves(solution_section)
    if not pgn_moves:
        return -1.0

    valid_moves, illegal_moves = validate_move_sequence(fen, pgn_moves)

    # Legality check (40%)
    if not valid_moves or illegal_moves:
        legality_score = -1.0
    else:
        legality_score = 1.0

    # First move correctness (40%)
    first_move_score = 0.0
    if valid_moves:
        extracted_first = normalize_move(valid_moves[0])
        expected_first = normalize_move(expected_first_move)
        first_move_score = 1.0 if extracted_first == expected_first else 0.0

    # Format compliance (15%) - check 5 sections
    sections = parse_sections(completion)
    sections_count = sum(sections.values())
    format_score = sections_count / 5.0

    # Piece accuracy (5%)
    piece_section = extract_piece_positions_section(completion)
    piece_score = 0.0
    if piece_section:
        piece_score = validate_piece_positions(fen, piece_section)

    # Weighted combination
    total = 0.40 * legality_score + 0.40 * first_move_score + 0.15 * format_score + 0.05 * piece_score

    return total


def chess_reasoning_reward(
    completions: list[str],
    prompts: list[str],
    fen: list[str],
    first_move: list[str],
    **kwargs: object,
) -> list[float]:
    """Main reward function for chess reasoning GRPO.

    Compatible with TRL GRPOTrainer reward function signature.
    Dataset columns (fen, first_move) are passed as kwargs.

    Reward components:
    - Legality (40%): -1.0 for illegal first move
    - First move correctness (40%): match puzzle solution
    - Format compliance (15%): 5 reasoning sections present
    - Piece accuracy (5%): board awareness in Step 2

    Args:
        completions: Generated outputs from model
        prompts: Input prompts (required by TRL, but not used in reward computation)
        fen: FEN positions from dataset
        first_move: Expected first moves from dataset
        **kwargs: Additional dataset columns (solution, themes, etc.)

    Returns:
        List of float rewards in [-1.0, 1.0] range
    """
    rewards: list[float] = []
    for completion, position_fen, expected_move in zip(completions, fen, first_move, strict=True):
        reward = compute_single_reward(completion, position_fen, expected_move)
        rewards.append(reward)
    return rewards


def legality_reward(
    completions: list[str],
    fen: list[str],
    **kwargs: object,
) -> list[float]:
    """Simple legality-only reward for early training stages.

    Returns 1.0 for legal first move, -1.0 otherwise.

    Args:
        completions: Generated outputs from model
        fen: FEN positions from dataset
        **kwargs: Additional dataset columns (ignored)

    Returns:
        List of float rewards (-1.0 or 1.0)
    """
    rewards: list[float] = []
    for completion, position_fen in zip(completions, fen, strict=True):
        solution_section = extract_solution_section(completion)
        if not solution_section:
            rewards.append(-1.0)
            continue

        pgn_moves = extract_pgn_moves(solution_section)
        valid_moves, illegal_moves = validate_move_sequence(position_fen, pgn_moves)

        if valid_moves and not illegal_moves:
            rewards.append(1.0)
        else:
            rewards.append(-1.0)

    return rewards


def correctness_reward(
    completions: list[str],
    fen: list[str],
    first_move: list[str],
    **kwargs: object,
) -> list[float | None]:
    """First move correctness reward.

    Returns None for illegal moves (to be combined with legality_reward
    using TRL's multiple reward function support).

    Args:
        completions: Generated outputs from model
        fen: FEN positions from dataset
        first_move: Expected first moves from dataset
        **kwargs: Additional dataset columns (ignored)

    Returns:
        List of float rewards (1.0, 0.0, or None for illegal)
    """
    rewards: list[float | None] = []
    for completion, position_fen, expected in zip(completions, fen, first_move, strict=True):
        solution_section = extract_solution_section(completion)
        if not solution_section:
            reward: float | None = None
            rewards.append(reward)
            continue

        pgn_moves = extract_pgn_moves(solution_section)
        valid_moves, _ = validate_move_sequence(position_fen, pgn_moves)

        if not valid_moves:
            reward = None
            rewards.append(reward)
            continue

        extracted = normalize_move(valid_moves[0])
        expected_norm = normalize_move(expected)
        reward = 1.0 if extracted == expected_norm else 0.0
        rewards.append(reward)

    return rewards
