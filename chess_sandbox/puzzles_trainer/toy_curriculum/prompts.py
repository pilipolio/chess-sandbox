"""Prompt templates for toy chess curriculum exercises."""

CAPTURE_SEQUENCE_EXAMPLE = {
    "piece_list": "White: Nc3 | Black: pd5, pe4",
    "fen": "8/8/8/3p4/4p3/2N5/8/8 w - - 0 1",
    "answer": "c3e4, e4d5",
}

MOVEMENT_PATH_EXAMPLE = {
    "piece_list": "White: Na1 |",
    "start_square": "a1",
    "target_square": "c5",
    "answer": "a1b3, b3c5",
}

FEN_TO_PIECE_LIST_EXAMPLE = {
    "fen": "8/8/8/3p4/4N3/8/8/8 w - - 0 1",
    "answer": "White: Ne4 | Black: pd5",
}

PIECE_LIST_TO_FEN_EXAMPLE = {
    "piece_list": "White: Ne4 | Black: pd5",
    "answer": "8/8/8/3p4/4N3/8/8/8 w - - 0 1",
}


def build_capture_sequence_prompt(fen: str, piece_list: str) -> str:
    """Build capture sequence prompt with few-shot example.

    >>> prompt = build_capture_sequence_prompt("8/8/8/3p4/8/2N5/8/8 w - - 0 1", "White: Nc3 | Black: pd5")
    >>> "Capture all black pawns" in prompt
    True
    >>> "c3e4, e4d5" in prompt
    True
    """
    return f"""Capture all black pawns with the white piece.
Respond with ONLY the move sequence in UCI notation, comma-separated.

Example:
Position (FEN): {CAPTURE_SEQUENCE_EXAMPLE["fen"]}
Position (pieces): {CAPTURE_SEQUENCE_EXAMPLE["piece_list"]}
Answer: {CAPTURE_SEQUENCE_EXAMPLE["answer"]}

Position (FEN): {fen}
Position (pieces): {piece_list}
Answer:"""


def build_movement_path_prompt(piece_list: str, start_square: str, target_square: str) -> str:
    """Build movement path prompt with few-shot example.

    >>> prompt = build_movement_path_prompt("White: Nb1 |", "b1", "d4")
    >>> "reach" in prompt and "d4" in prompt
    True
    """
    return f"""Move the piece to reach the target square using the shortest path.
Respond with ONLY the move sequence in UCI notation, comma-separated.

Example:
Position: {MOVEMENT_PATH_EXAMPLE["piece_list"]}
Move from {MOVEMENT_PATH_EXAMPLE["start_square"]} to {MOVEMENT_PATH_EXAMPLE["target_square"]}
Answer: {MOVEMENT_PATH_EXAMPLE["answer"]}

Position: {piece_list}
Move from {start_square} to {target_square}
Answer:"""


def build_fen_to_piece_list_prompt(fen: str) -> str:
    """Build FEN to piece list conversion prompt.

    >>> prompt = build_fen_to_piece_list_prompt("8/8/8/8/4R3/8/8/8 w - - 0 1")
    >>> "White:" in prompt and "Black:" in prompt
    True
    """
    return f"""List all pieces from this FEN position.
Format: "White: pieces | Black: pieces" using piece symbol + square (e.g., Ne4, pd5).
Uppercase for White, lowercase for Black.

Example:
FEN: {FEN_TO_PIECE_LIST_EXAMPLE["fen"]}
Answer: {FEN_TO_PIECE_LIST_EXAMPLE["answer"]}

FEN: {fen}
Answer:"""


def build_piece_list_to_fen_prompt(piece_list: str) -> str:
    """Build piece list to FEN conversion prompt.

    >>> prompt = build_piece_list_to_fen_prompt("White: Re4 | Black:")
    >>> "FEN" in prompt
    True
    """
    return f"""Generate the FEN for this position.
White to move, no castling rights.

Example:
Pieces: {PIECE_LIST_TO_FEN_EXAMPLE["piece_list"]}
Answer: {PIECE_LIST_TO_FEN_EXAMPLE["answer"]}

Pieces: {piece_list}
Answer:"""


def build_legal_moves_uci_prompt(fen: str, piece_list: str) -> str:
    """Build legal moves (UCI format) prompt.

    >>> prompt = build_legal_moves_uci_prompt("8/8/8/8/4N3/8/8/8 w - - 0 1", "White: Ne4 |")
    >>> "legal moves" in prompt.lower()
    True
    """
    return f"""List all legal moves for the white piece in UCI notation, comma-separated.

Example:
Position (FEN): 8/8/8/8/4N3/8/8/8 w - - 0 1
Position (pieces): White: Ne4 | Black:
Answer: e4c3, e4c5, e4d2, e4d6, e4f2, e4f6, e4g3, e4g5

Position (FEN): {fen}
Position (pieces): {piece_list}
Answer:"""
