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


ASCII_BOARD_EXAMPLE_FEN = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
ASCII_BOARD_EXAMPLE_OUTPUT = """r n b q k b n r
p p p p p p p p
. . . . . . . .
. . . . . . . .
. . . . P . . .
. . . . . . . .
P P P P . P P P
R N B Q K B N R"""


def build_ascii_board_prompt(fen: str) -> str:
    """Build an ASCII board prompt with example."""
    return f"""Generate an ASCII representation of this chess position.
Output ONLY the 8x8 grid. Use uppercase for White, lowercase for Black, dots for empty.

Example:
FEN: {ASCII_BOARD_EXAMPLE_FEN}
Output:
{ASCII_BOARD_EXAMPLE_OUTPUT}

FEN: {fen}
Output:"""


CONCEPT_EXAMPLE_FEN = "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4"
CONCEPT_EXAMPLE_ANSWER = "mate, mateIn1, oneMove"

AVAILABLE_THEMES = [
    "mate",
    "mateIn1",
    "oneMove",
    "middlegame",
    "endgame",
    "kingsideAttack",
    "master",
    "opening",
    "hangingPiece",
    "rookEndgame",
    "backRankMate",
    "attackingF2F7",
    "queensideAttack",
    "smotheredMate",
    "pin",
]


def build_concept_detection_prompt(fen: str) -> str:
    """Build a concept detection prompt with example."""
    themes_list = ", ".join(AVAILABLE_THEMES)
    return f"""What chess concepts or themes are present in this position?
Respond with ONLY theme names from this list, comma-separated: {themes_list}

Example:
FEN: {CONCEPT_EXAMPLE_FEN}
Answer: {CONCEPT_EXAMPLE_ANSWER}

FEN: {fen}
Answer:"""


CAPTURES_EXAMPLE_FEN = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
CAPTURES_EXAMPLE_ANSWER = "Bxf7+"


def build_legal_captures_prompt(fen: str) -> str:
    """Build a legal captures prompt with example."""
    return f"""List all legal captures in this position.
Respond with ONLY the captures in SAN notation, comma-separated. If no captures, respond "none".

Example:
FEN: {CAPTURES_EXAMPLE_FEN}
Answer: {CAPTURES_EXAMPLE_ANSWER}

FEN: {fen}
Answer:"""
