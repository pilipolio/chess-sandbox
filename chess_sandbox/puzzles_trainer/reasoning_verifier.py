"""Verify generated reasoning traces for chess puzzles.

Validates structure, move legality, and correctness of generated reasoning traces.
Assigns a score based on section completeness, move legality, and first move correctness.
"""

import re
from dataclasses import dataclass, field

import chess


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


SECTION_PATTERNS = {
    "position_analysis": r"##\s*Position\s*Analysis",
    "tactical_assessment": r"##\s*Tactical\s*Assessment",
    "solution": r"##\s*Solution",
}

# Matches PGN moves like "19. Nxh7!" or "19...Kxh7" with optional annotations
PGN_MOVE_PATTERN = re.compile(
    r"""
    (\d+)           # Move number
    (\.{1,3})       # 1-3 dots (. for white, ... for black)
    \s*
    ([KQRBN]?       # Optional piece letter
    [a-h]?[1-8]?    # Optional disambiguation
    x?              # Optional capture
    [a-h][1-8]      # Destination square
    (?:=[QRBN])?    # Optional promotion
    [+#]?)          # Optional check/mate
    [!?]*           # Optional annotation symbols
    """,
    re.VERBOSE,
)

# Alternative pattern for castling
CASTLING_PATTERN = re.compile(r"(\d+)(\.{1,3})\s*(O-O(?:-O)?)[+#]?[!?]*")


def parse_sections(reasoning: str) -> dict[str, bool]:
    """Check which sections are present in the reasoning trace."""
    return {name: bool(re.search(pattern, reasoning, re.IGNORECASE)) for name, pattern in SECTION_PATTERNS.items()}


def extract_solution_section(reasoning: str) -> str | None:
    """Extract the Solution section content from reasoning."""
    match = re.search(r"##\s*Solution\s*\n(.*?)(?=##|\</think\>|$)", reasoning, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else None


def extract_pgn_moves(solution_text: str) -> list[tuple[int, bool, str]]:
    """Extract PGN moves from solution section.

    Returns list of (move_number, is_white, san_move) tuples.
    """
    moves: list[tuple[int, bool, str]] = []

    # Find regular moves
    for match in PGN_MOVE_PATTERN.finditer(solution_text):
        move_num = int(match.group(1))
        is_white = len(match.group(2)) == 1  # Single dot = white
        san = match.group(3)
        moves.append((move_num, is_white, san))

    # Find castling moves
    for match in CASTLING_PATTERN.finditer(solution_text):
        move_num = int(match.group(1))
        is_white = len(match.group(2)) == 1
        san = match.group(3)
        moves.append((move_num, is_white, san))

    # Sort by move number and color (white before black)
    return sorted(moves, key=lambda x: (x[0], not x[1]))


def validate_move_sequence(fen: str, moves: list[tuple[int, bool, str]]) -> tuple[list[str], list[str]]:
    """Validate a sequence of moves from a position.

    Returns (valid_moves, illegal_moves) where each is a list of SAN strings.
    """
    board = chess.Board(fen)
    valid: list[str] = []
    illegal: list[str] = []

    for _move_num, _is_white, san in moves:
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


def extract_final_move(reasoning: str) -> str | None:
    """Extract the final move after </think> tag."""
    # Look for move after </think>
    match = re.search(r"</think>\s*\n?\s*([A-Za-z0-9+#=x-]+)", reasoning)
    if match:
        return match.group(1).strip()

    # Fallback: last line that looks like a move
    lines = reasoning.strip().split("\n")
    for line in reversed(lines):
        line = line.strip()
        if re.match(r"^[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?$", line):
            return line
        if re.match(r"^O-O(?:-O)?[+#]?$", line):
            return line

    return None


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

    # 2. Extract and validate moves from Solution section
    solution_section = extract_solution_section(reasoning)
    if solution_section:
        pgn_moves = extract_pgn_moves(solution_section)
        _valid_moves, result.illegal_moves = validate_move_sequence(fen, pgn_moves)
    else:
        result.format_errors.append("No Solution section found")

    # 3. Check first move correctness
    result.extracted_first_move = extract_final_move(reasoning)
    if result.extracted_first_move and expected_solution:
        normalized_extracted = normalize_move(result.extracted_first_move)
        normalized_expected = normalize_move(expected_solution[0])
        result.first_move_correct = normalized_extracted == normalized_expected

    # 4. Calculate score
    # Section completeness: 30%
    sections_score = 0.3 * (sections_count / 3)

    # Move legality: 40%
    illegal_count = len(result.illegal_moves)
    legality_score = 0.4 * max(0.0, 1 - illegal_count / max(max_illegal_moves, 1))

    # First move correctness: 30%
    first_move_score = 0.3 if result.first_move_correct else 0.0

    result.score = sections_score + legality_score + first_move_score

    return result
