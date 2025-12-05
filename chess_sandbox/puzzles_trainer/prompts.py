"""Shared prompt templates for chess puzzle training."""

PUZZLE_EXAMPLE_1 = {
    "fen": "8/1r6/8/5k1p/2B4K/1P6/8/1R6 w - - 1 44",
    "answer": "h4h5",
}

PUZZLE_EXAMPLE_2 = {
    "fen": "1r3rk1/pbp2p1p/1p4pb/3qp3/1n6/PP1PP1PP/NBPQ3K/R4RN1 w - - 2 18",
    "answer": "a2b4",
}


def build_puzzle_prompt(fen: str) -> str:
    """Build a puzzle prompt with few-shot examples."""
    return f"""Find the best move for this chess puzzle. Respond with ONLY the move in UCI notation (e.g., e2e4).

Example 1:
FEN: {PUZZLE_EXAMPLE_1["fen"]}
Answer: {PUZZLE_EXAMPLE_1["answer"]}

Example 2:
FEN: {PUZZLE_EXAMPLE_2["fen"]}
Answer: {PUZZLE_EXAMPLE_2["answer"]}

Now solve:
FEN: {fen}
Answer:"""
