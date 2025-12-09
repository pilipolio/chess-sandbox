"""Verify generated reasoning traces for chess puzzles.

Validates structure, move legality, and correctness of generated reasoning traces.
Assigns a score based on section completeness, move legality, and first move correctness.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import chess
from chess import pgn


def _empty_dict() -> dict[str, bool]:
    return {}


def _empty_list() -> list[str]:
    return []


@dataclass
class VerificationResult:
    """Result of verifying a reasoning trace."""

    score: float  # 0.0 to 1.0
    sections_found: dict[str, bool] = field(default_factory=_empty_dict)
    illegal_moves: list[str] = field(default_factory=_empty_list)
    format_errors: list[str] = field(default_factory=_empty_list)
    first_move_correct: bool = False
    extracted_first_move: str | None = None
    piece_positions_accuracy: float = 0.0


SECTION_PATTERNS = {
    "fen_parsing": r"##\s*Step\s*1[:\s]*FEN\s*parsing",
    "piece_positions": r"##\s*Step\s*2[:\s]*Piece\s*Positions",
    "position_summary": r"##\s*Step\s*3[:\s]*Position\s*Summary",
    "candidate_moves": r"##\s*Step\s*4[:\s]*Candidate\s*Moves",
    "lines_exploration": r"##\s*Step\s*5[:\s]*Lines\s*Exploration",
}

PIECE_NAME_MAP = {
    "king": chess.KING,
    "k": chess.KING,
    "queen": chess.QUEEN,
    "q": chess.QUEEN,
    "rook": chess.ROOK,
    "r": chess.ROOK,
    "bishop": chess.BISHOP,
    "b": chess.BISHOP,
    "knight": chess.KNIGHT,
    "n": chess.KNIGHT,
    "pawn": chess.PAWN,
    "pawns": chess.PAWN,
    "p": chess.PAWN,
}

SQUARE_PATTERN = re.compile(r"[a-h][1-8]")


def extract_piece_positions_section(reasoning: str) -> str | None:
    """Extract the piece positions content from Step 2 section."""
    pattern = r"##\s*Step\s*2[:\s]*Piece\s*Positions\s*\n(.*?)(?=##\s*Step|</think>|$)"
    match = re.search(pattern, reasoning, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else None


def parse_flexible_piece_list(
    text: str,
) -> list[tuple[chess.PieceType, chess.Square, chess.Color]]:
    """Parse piece positions from flexible natural language format.

    Supports formats like:
    - "Qa3" (standard algebraic)
    - "Queen a3" or "Q a3" (piece name + square)
    - "pawns a2, b2, c2" (grouped pieces)
    - "White: Qa3, Kh1. Black: Kg8, Ra8" (color sections)
    """
    pieces: list[tuple[chess.PieceType, chess.Square, chess.Color]] = []
    current_color = chess.WHITE
    current_piece_type: chess.PieceType | None = None

    tokens = re.split(r"[\s,.:;]+", text)

    i = 0
    while i < len(tokens):
        token = tokens[i].strip()
        if not token:
            i += 1
            continue

        token_lower = token.lower()

        if token_lower in ("white", "white:"):
            current_color = chess.WHITE
            current_piece_type = None
            i += 1
            continue

        if token_lower in ("black", "black:"):
            current_color = chess.BLACK
            current_piece_type = None
            i += 1
            continue

        if token_lower in PIECE_NAME_MAP:
            current_piece_type = PIECE_NAME_MAP[token_lower]
            i += 1
            continue

        square_match = SQUARE_PATTERN.search(token_lower)
        if square_match:
            square_str = square_match.group(0)
            square = chess.parse_square(square_str)

            piece_type = current_piece_type
            prefix = token_lower[: square_match.start()]
            if prefix and prefix in PIECE_NAME_MAP:
                piece_type = PIECE_NAME_MAP[prefix]

            if piece_type is not None:
                pieces.append((piece_type, square, current_color))

        i += 1

    return pieces


def validate_piece_positions(fen: str, piece_text: str) -> float:
    """Validate piece positions against FEN.

    Returns accuracy as fraction of expected pieces correctly identified (0.0 to 1.0).
    Extra pieces (hallucinations) count as errors.
    """
    board = chess.Board(fen)

    expected: set[tuple[chess.PieceType, chess.Square, chess.Color]] = set()
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            expected.add((piece.piece_type, square, piece.color))

    parsed = set(parse_flexible_piece_list(piece_text))

    correct = len(expected & parsed)
    errors = len(expected - parsed) + len(parsed - expected)
    total = correct + errors

    return correct / total if total > 0 else 0.0


def parse_sections(reasoning: str) -> dict[str, bool]:
    """Check which sections are present in the reasoning trace."""
    return {name: bool(re.search(pattern, reasoning, re.IGNORECASE)) for name, pattern in SECTION_PATTERNS.items()}


def extract_solution_section(reasoning: str) -> str | None:
    """Extract the solution content from after </think> tag."""
    match = re.search(r"</think>\s*\n?\s*(.+?)$", reasoning, re.DOTALL)
    return match.group(1).strip() if match else None


def extract_pgn_moves(solution_text: str) -> list[str]:
    """Extract PGN moves from solution section.

    Uses chess.pgn.MOVETEXT_REGEX to extract SAN moves, ignoring move numbers.
    Moves will be validated against the actual position, so move numbers are not needed.

    Filters out square names (like "e7", "b1") that match the regex but aren't valid moves.

    TODO: In the future, we may want to verify move numbers match expected sequence.

    Returns list of SAN move strings in order of appearance.
    """
    moves: list[str] = []

    # First, remove comments and other non-move content to avoid regex issues
    # Comments are in { } brackets
    import re as re_module

    text_without_comments = re_module.sub(r"\{[^}]*\}", "", solution_text)

    # Use chess.pgn's MOVETEXT_REGEX to find all SAN moves
    # This handles all standard chess notation including castling, promotions, etc.
    for match in pgn.MOVETEXT_REGEX.finditer(text_without_comments):
        token = match.group(0)
        # MOVETEXT_REGEX matches moves in group 1, but also matches other tokens
        # Check if it's actually a move (not a comment, NAG, result, etc.)
        if match.group(1):  # Group 1 contains the move
            san = match.group(1)
            # Skip non-move tokens like NAGs, results, parentheses, etc.
            if not any(
                token.startswith(prefix)
                for prefix in [";", "$", "(", ")", "*", "1-0", "0-1", "1/2", "--", "Z0", "0000", "@@@@"]
            ):
                # Check for check/mate symbols immediately after the move
                end_pos = match.end()
                if end_pos < len(text_without_comments) and text_without_comments[end_pos] in "+#":
                    san += text_without_comments[end_pos]
                moves.append(san)

    return moves


def validate_move_sequence(fen: str, moves: list[str]) -> tuple[list[str], list[str]]:
    """Validate a sequence of moves from a position.

    Returns (valid_moves, illegal_moves) where each is a list of SAN strings.
    """
    board = chess.Board(fen)
    valid: list[str] = []
    illegal: list[str] = []

    for san in moves:
        try:
            # Try to parse the move
            move = board.parse_san(san)
            if move in board.legal_moves:
                valid.append(san)
                board.push(move)
            else:
                illegal.append(san)
        except (chess.InvalidMoveError, chess.AmbiguousMoveError, ValueError):
            illegal.append(san)

    return valid, illegal


def normalize_move(san: str) -> str:
    """Normalize a SAN move for comparison (remove annotations)."""
    return re.sub(r"[!?]+$", "", san).strip()


def verify_reasoning_trace(
    fen: str,
    reasoning: str,
    expected_solution: list[str],
    max_illegal_moves: int = 2,
) -> VerificationResult:
    """Validate generated reasoning trace with score-based filtering.

    Args:
        fen: The puzzle position FEN
        reasoning: The generated reasoning trace (including <think> tags)
        expected_solution: List of SAN moves that solve the puzzle
        max_illegal_moves: Maximum illegal moves before legality score drops to 0

    Returns:
        VerificationResult with score and detailed validation info
    """
    result = VerificationResult(score=0.0)

    # 1. Parse sections
    result.sections_found = parse_sections(reasoning)
    sections_count = sum(result.sections_found.values())

    # 2. Extract and validate piece positions from Step 2
    piece_section = extract_piece_positions_section(reasoning)
    if piece_section:
        result.piece_positions_accuracy = validate_piece_positions(fen, piece_section)

    # 3. Extract and validate moves from Solution section
    extracted_solution_valid_moves: list[str] = []
    solution_section = extract_solution_section(reasoning)
    if solution_section:
        pgn_moves = extract_pgn_moves(solution_section)
        extracted_solution_valid_moves, result.illegal_moves = validate_move_sequence(fen, pgn_moves)
    else:
        result.format_errors.append("No Solution section found")

    # 4. Check first move correctness
    result.extracted_first_move = extracted_solution_valid_moves[0] if extracted_solution_valid_moves else None
    if result.extracted_first_move and expected_solution:
        normalized_extracted = normalize_move(result.extracted_first_move)
        normalized_expected = normalize_move(expected_solution[0])
        result.first_move_correct = normalized_extracted == normalized_expected

    # 5. Calculate score
    # Section completeness: 30%
    sections_score = 0.3 * (sections_count / 5)

    # Move legality: 40%
    illegal_count = len(result.illegal_moves)
    legality_score = 0.4 * max(0.0, 1 - illegal_count / max(max_illegal_moves, 1))

    # First move correctness: 30%
    first_move_score = 0.3 if result.first_move_correct else 0.0

    result.score = sections_score + legality_score + first_move_score

    return result
