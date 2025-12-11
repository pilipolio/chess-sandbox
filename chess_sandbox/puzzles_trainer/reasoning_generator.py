"""Generate synthetic reasoning traces for chess puzzles using LLMs.

Uses OpenRouter API by default to access models like gpt-oss-20b.
Set OPENROUTER_API_KEY environment variable or use --api-key flag.

Example:
    export OPENROUTER_API_KEY=sk-...
    uv run puzzles-generate-reasoning --sample-size 20 --model openai/gpt-oss-20b:free
"""

import asyncio
import math
import os
from typing import Any

import chess
import chess.engine
import click
import logfire
from datasets import (  # pyright: ignore[reportMissingTypeStubs]
    Dataset,
    DatasetDict,
    load_dataset,  # pyright: ignore[reportMissingTypeStubs,reportUnknownVariableType]
)
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from tqdm.asyncio import tqdm_asyncio

from chess_sandbox.engine.analyse import EngineConfig
from chess_sandbox.engine.maia import HumanMove, MaiaConfig, analyze_human_moves
from chess_sandbox.puzzles_trainer.reasoning_verifier import VerificationResult, verify_reasoning_trace


class RefutedLine(BaseModel):
    """A refuted human-likely move with Stockfish's response."""

    human_move: str  # SAN notation (e.g., "Qxb6")
    human_policy: float  # Maia policy % (e.g., 62.0)
    refutation_line: list[str]  # SAN moves showing refutation (2-ply)
    score: float | None  # Evaluation in pawns (None for mate)
    mate_in: int | None  # Mate-in-N if applicable
    explanation: str  # Short description of why it fails


class ReasoningTrace(BaseModel):
    """Structured reasoning for a chess puzzle."""

    fen_parsing: str = Field(
        description="Breaking down FEN into each rank (starting from Rank8 to Rank1) to extract piece positions, "
        "e.g., 'Rank8: r6k -> a8 rook, h8 king. Rank7: pp2r2p -> a7 pawn, b7 pawn, e7 rook, h7 pawn.'"
    )
    piece_positions: str = Field(
        description="List key pieces and their squares, e.g., 'White: Qh6, Re6, Nb3. Black: Kh8, Re7, Qb2, Bg3'"
    )
    position_summary: str = Field(
        description="A single short sentence summarising material balance, king safety, piece activity, move number, "
        "side to move and the move just played. NO LINES"
    )
    candidate_moves_csv: list[str] = Field(
        description="Candidate moves using comma separated SAN notations: Qxh7, Rxe7, ..."
    )
    candidate_moves_reasoning: str = Field(
        description="Explain candidate moves based on piece positions and relevant tactical themes, "
        "unbiased by the solution and using short sentences, e.g., 'White queen on c2 can deliver "
        "a check and capture the pawn on h7 along the open b1-h7 diagonal c2-d3-e4-f5-g6-h7'"
    )
    lines_pgn_with_comments: str = Field(
        description="Copy the provided PGN exactly, ONLY adding {comments} after key moves. "
        "Do NOT modify, add, or remove any moves."
    )


load_dotenv()
logfire.configure()
logfire.instrument_openai()

DATASET_ID = "Lichess/chess-puzzles"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

# Theme quotas for balanced sampling (must sum to 1.0)
THEME_QUOTAS: dict[str, float] = {
    "fork": 0.20,
    "pin": 0.15,
    "skewer": 0.10,
    "discoveredAttack": 0.10,
    "hangingPiece": 0.10,
    "backRankMate": 0.10,
    "mateIn1": 0.10,
    "deflection": 0.05,
    "attraction": 0.05,
    "other": 0.05,  # Catch-all for puzzles not matching above themes
}


def extract_puzzle_position_and_solution(fen: str, uci_moves: str) -> tuple[str, str, list[str]] | None:
    """Extract puzzle position, opponent's last move, and solution.

    In Lichess puzzles, the first move is the opponent's setup move.
    The puzzle position is AFTER this move, and the solution starts from move 2.

    Returns:
        Tuple of (puzzle_fen, last_move_san, solution_san_list) or None if invalid.
        Returns None if ANY move in the solution sequence is illegal (strict validation).
    """
    board = chess.Board(fen)
    moves = uci_moves.strip().split()

    if len(moves) < 2:
        return None

    try:
        # First move is opponent's setup
        opponent_move = chess.Move.from_uci(moves[0])
        if opponent_move not in board.legal_moves:
            return None
        last_move_san = board.san(opponent_move)
        board.push(opponent_move)
        puzzle_fen = board.fen()

        # STRICT: Validate ENTIRE solution sequence - reject if any move is illegal
        solution_san: list[str] = []
        for uci in moves[1:]:
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                return None  # Reject puzzle entirely, don't return partial solution
            solution_san.append(board.san(move))
            board.push(move)

        if not solution_san:
            return None

        return puzzle_fen, last_move_san, solution_san

    except ValueError:
        return None


def build_piece_placement_summary(board: chess.Board) -> str:
    """Build compact piece placement summary by color."""
    piece_names = {
        chess.KING: "K",
        chess.QUEEN: "Q",
        chess.ROOK: "R",
        chess.BISHOP: "B",
        chess.KNIGHT: "N",
    }

    def pieces_for_color(color: chess.Color) -> str:
        pieces: list[str] = []
        pawn_count = 0
        for square, piece in board.piece_map().items():
            if piece.color != color:
                continue
            square_name = chess.square_name(square)
            if piece.piece_type == chess.PAWN:
                pawn_count += 1
            else:
                piece_char = piece_names.get(piece.piece_type, "?")
                pieces.append(f"{piece_char}{square_name}")
        if pawn_count > 0:
            pieces.append(f"pawns ({pawn_count})")
        return ", ".join(pieces)

    white_pieces = pieces_for_color(chess.WHITE)
    black_pieces = pieces_for_color(chess.BLACK)
    return f"White: {white_pieces}\nBlack: {black_pieces}"


def get_maia_predictions(
    board: chess.Board,
    engine: chess.engine.SimpleEngine,
    top_n: int = 3,
) -> list[HumanMove]:
    """Get top N Maia human move predictions for a position.

    Args:
        board: Chess board position to analyze
        engine: Pre-instantiated Maia engine
        top_n: Number of top predictions to return

    Returns:
        List of HumanMove objects sorted by policy probability (descending)
    """
    try:
        moves = analyze_human_moves(board, engine, nodes=1)
        return moves[:top_n]
    except Exception:
        return []


def format_maia_predictions(maia_moves: list[HumanMove]) -> str:
    """Format Maia predictions as a string for the prompt."""
    if not maia_moves:
        return "None"
    parts = [f"{m.san_move} ({m.policy:.0f}%)" for m in maia_moves]
    return ", ".join(parts)


def format_score_for_prompt(score: float | None, mate_in: int | None, is_white_to_move: bool) -> str:
    """Format engine score for prompt display using standard chess notation.

    Args:
        score: Score in centipawns from white's perspective (None for mate)
        mate_in: Mate-in-N if applicable (positive = mating, negative = getting mated)
        is_white_to_move: Whether it's white's turn

    Returns:
        Formatted string like "+1.5", "-#3", or "N/A"
    """
    if mate_in is not None:
        # Mate score: adjust sign based on side to move
        # Positive mate_in means the side to move is getting mated
        adjusted_mate = -mate_in if is_white_to_move else mate_in
        sign = "+" if adjusted_mate > 0 else "-"
        return f"{sign}#{abs(mate_in)}"

    if score is not None:
        # Regular score: adjust perspective if black to move
        adjusted_score = score / 100 if is_white_to_move else -score / 100
        return f"{adjusted_score:+.1f}"

    return "N/A"


def build_refutation_explanation(score: float | None, mate_in: int | None, is_white_to_move: bool) -> str:
    """Generate human-readable explanation for why a move fails."""
    if mate_in is not None:
        # Getting mated
        return "leads to forced mate"

    if score is None:
        return "unclear"

    # Score is in centipawns from white's perspective; adjust for side to move
    relative_score = score if is_white_to_move else -score

    if relative_score < -300:
        return "loses significant material"
    if relative_score < -100:
        return "loses material"
    if relative_score < -30:
        return "gives opponent advantage"

    return "inferior to the solution"


def generate_refuted_lines(
    board: chess.Board,
    maia_moves: list[HumanMove],
    solution_first_move: str,
    stockfish_engine: chess.engine.SimpleEngine,
    limit: chess.engine.Limit,
) -> list[RefutedLine]:
    """Generate refutation lines for human-likely moves that aren't the solution.

    Args:
        board: Current puzzle position
        maia_moves: Top Maia predictions with policy %
        solution_first_move: The correct first move (SAN)
        stockfish_engine: Pre-instantiated Stockfish engine
        limit: Analysis limit (depth)

    Returns:
        List of RefutedLine objects for non-solution moves (2-ply refutations)
    """
    refuted_lines: list[RefutedLine] = []
    is_white_to_move = board.turn == chess.WHITE

    for maia_move in maia_moves:
        # Skip if this is the solution move
        if maia_move.san_move == solution_first_move:
            continue

        # Play the human move
        board_after_human = board.copy()
        move = board_after_human.parse_san(maia_move.san_move)
        board_after_human.push(move)

        # Get Stockfish's best response (2-ply: opponent response + one more)
        results = stockfish_engine.analyse(board_after_human, limit, multipv=1)

        # multipv=1 still returns a list
        if not results:
            continue
        result = results[0]

        pv: list[chess.Move] = result.get("pv", [])
        if not pv:
            continue

        # Extract 2-ply refutation line
        refutation_moves: list[str] = []
        temp_board = board_after_human.copy()
        for pv_move in pv[:2]:
            refutation_moves.append(temp_board.san(pv_move))
            temp_board.push(pv_move)

        # Extract score
        engine_score: chess.engine.PovScore | None = result.get("score")
        score: float | None = None
        mate_in: int | None = None

        if engine_score:
            pov_score = engine_score.white()
            score = pov_score.score()
            mate_in = pov_score.mate()

        explanation = build_refutation_explanation(score, mate_in, is_white_to_move)

        refuted_lines.append(
            RefutedLine(
                human_move=maia_move.san_move,
                human_policy=maia_move.policy,
                refutation_line=refutation_moves,
                score=score,
                mate_in=mate_in,
                explanation=explanation,
            )
        )

    return refuted_lines


def build_lines_exploration_pgn(
    board: chess.Board,
    solution_san: list[str],
    refuted_lines: list[RefutedLine] | None = None,
) -> str:
    """Build valid PGN with solution as main line and refuted lines as variations.

    Variations are placed immediately after the first move (the branching point)
    so PGN parsers correctly understand they are alternatives to that move.

    Example: 25. Rxe7! (25. Rxf6? Re1+ {mate}) (25. hxg3? Rxe6) Qb1+ 26. Nc1

    Args:
        board: Starting position
        solution_san: Solution moves in SAN notation
        refuted_lines: Optional refuted alternative moves

    Returns:
        Valid PGN string with main line and variations
    """
    move_number = board.fullmove_number
    is_white = board.turn == chess.WHITE
    parts: list[str] = []

    # First move with ! annotation
    if solution_san:
        first_move = solution_san[0]
        if is_white:
            parts.append(f"{move_number}. {first_move}!")
        else:
            parts.append(f"{move_number}... {first_move}!")

    # Insert variations immediately after first move (they are alternatives to it)
    if refuted_lines:
        for rl in refuted_lines:
            var_parts: list[str] = []

            if is_white:
                var_parts.append(f"{move_number}. {rl.human_move}?")
            else:
                var_parts.append(f"{move_number}... {rl.human_move}?")

            if rl.refutation_line:
                var_parts.extend(rl.refutation_line)

            var_parts.append(f"{{{rl.explanation}}}")
            parts.append(f"({' '.join(var_parts)})")

    # Continue with remaining solution moves
    if len(solution_san) > 1:
        temp_board = board.copy()
        # Push first move
        try:
            first_move_obj = temp_board.parse_san(solution_san[0])
            temp_board.push(first_move_obj)
        except (chess.InvalidMoveError, chess.AmbiguousMoveError, ValueError):
            pass

        for san in solution_san[1:]:
            if temp_board.turn == chess.WHITE:
                current_move_num = temp_board.fullmove_number
                parts.append(f"{current_move_num}. {san}")
            else:
                parts.append(san)

            try:
                move = temp_board.parse_san(san)
                temp_board.push(move)
            except (chess.InvalidMoveError, chess.AmbiguousMoveError, ValueError):
                break

    return " ".join(parts)


def identify_candidate_moves(board: chess.Board) -> tuple[str, str]:
    """Identify forcing moves: checks and captures.

    Returns:
        Tuple of (checks_str, captures_str)
    """
    checks: list[str] = []
    captures: list[str] = []

    for move in board.legal_moves:
        san = board.san(move)
        if board.gives_check(move):
            checks.append(san)
        elif board.is_capture(move):
            captures.append(san)

    checks_str = ", ".join(checks) if checks else "None"
    captures_str = ", ".join(captures) if captures else "None"
    return checks_str, captures_str


REASONING_PROMPT_TEMPLATE = """\
You are a chess instructor explaining your thought process when solving puzzles for your students.
Break down step by step how you analyze this position and arrive at your answer.

Position (FEN): {puzzle_fen}
Pieces: {piece_positions}
Candidate checks: {candidate_checks}
Candidate captures: {candidate_captures}
Most likely human moves: {human_moves}
Last move: {last_move}
To move: {side_to_move}
Themes: {themes}

Lines PGN (solution + refuted alternatives):
{lines_pgn}

IMPORTANT INSTRUCTIONS:
1. In your analysis sections (FEN parsing, Position Summary, Candidate Moves), do NOT quote the final solution.
2. For lines_pgn_with_comments: copy the Lines PGN above exactly, ONLY adding {{comments}} after key moves.
   Do NOT modify, add, or remove any chess moves - the PGN structure must remain identical.

Your task is to explain WHY the solution works and why alternatives fail."""


def build_reasoning_prompt(
    puzzle_fen: str,
    last_move_san: str,
    themes: list[str],
    solution_san: list[str],
    maia_moves: list[HumanMove] | None = None,
    refuted_lines: list[RefutedLine] | None = None,
) -> tuple[str, str]:
    """Build prompt with position context for structured output.

    Returns:
        Tuple of (prompt, lines_pgn) where lines_pgn is the pre-built valid PGN
    """
    board = chess.Board(puzzle_fen)
    side_to_move = "White" if board.turn == chess.WHITE else "Black"
    checks_str, captures_str = identify_candidate_moves(board)
    human_moves_str = format_maia_predictions(maia_moves) if maia_moves else "None"

    # Build valid PGN from solution + refuted lines
    lines_pgn = build_lines_exploration_pgn(board, solution_san, refuted_lines)

    prompt = REASONING_PROMPT_TEMPLATE.format(
        puzzle_fen=puzzle_fen,
        piece_positions=build_piece_placement_summary(board),
        candidate_checks=checks_str,
        candidate_captures=captures_str,
        human_moves=human_moves_str,
        lines_pgn=lines_pgn,
        last_move=last_move_san,
        side_to_move=side_to_move,
        themes=", ".join(themes) if themes else "none",
    )
    return prompt, lines_pgn


async def generate_reasoning_trace(
    prompt: str,
    client: AsyncOpenAI,
    model: str = "gpt-oss-20b",
) -> ReasoningTrace | None:
    """Generate structured reasoning trace using OpenAI-compatible API."""
    response = await client.beta.chat.completions.parse(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format=ReasoningTrace,
        extra_body={
            "reasoning": {
                "effort": "low"  # "high", "medium", "low"
            }
        },
        # max_tokens=1024,
        temperature=0.7,
    )
    return response.choices[0].message.parsed


def join_reasoning_trace(trace: ReasoningTrace, lines_pgn: str) -> str:
    """Join structured trace into <think> format for SFT training."""
    return f"""<think>
## Step 1: FEN parsing
{trace.fen_parsing}

## Step 2: Piece Positions
{trace.piece_positions}

## Step 3: Position Summary
{trace.position_summary}

## Step 4: Candidate Moves
{trace.candidate_moves_reasoning}
{trace.candidate_moves_csv}

## Step 5: Lines Exploration
{trace.lines_pgn_with_comments}

</think>
{lines_pgn}"""


def format_reasoning_example(
    puzzle_fen: str,
    last_move_san: str,
    themes: list[str],
    solution_san: list[str],
    trace: ReasoningTrace,
    lines_pgn: str,
    puzzle_id: str,
    rating: int,
    verification: VerificationResult | None = None,
) -> dict[str, Any]:
    """Format puzzle with structured reasoning for SFT training."""
    first_move = solution_san[0]
    question = f"Position: {puzzle_fen}\nOpponent's last move: {last_move_san}\nFind the best move."
    answer = join_reasoning_trace(trace, lines_pgn)

    result: dict[str, Any] = {
        "question": question,
        "answer": answer,
        "fen": puzzle_fen,
        "last_move": last_move_san,
        "solution": " ".join(solution_san),
        "themes": themes,
        "first_move": first_move,
        "rating": rating,
        # Structured fields for downstream flexibility
        "fen_parsing": trace.fen_parsing,
        "piece_positions": trace.piece_positions,
        "position_summary": trace.position_summary,
        "candidate_moves_reasoning": trace.candidate_moves_reasoning,
        "candidate_moves_csv": trace.candidate_moves_csv,
        "lines_exploration": lines_pgn,
        "lines_pgn_with_comments": trace.lines_pgn_with_comments,
        "source_url": f"https://lichess.org/training/{puzzle_id}",
    }

    if verification is not None:
        result["verification_score"] = verification.score
        result["sections_found"] = verification.sections_found
        result["illegal_moves"] = verification.illegal_moves
        result["first_move_correct"] = verification.first_move_correct

    return result


async def process_puzzle(
    example: dict[str, Any],
    client: AsyncOpenAI,
    model: str,
    semaphore: asyncio.Semaphore,
    maia_engine: chess.engine.SimpleEngine | None = None,
    stockfish_engine: chess.engine.SimpleEngine | None = None,
    maia_top_n: int = 3,
    refutation_depth: int = 20,
) -> dict[str, Any] | None:
    """Process a single puzzle: generate and verify reasoning trace."""
    async with semaphore:
        fen = str(example["FEN"])
        uci_moves = str(example["Moves"])
        themes = list(example.get("Themes", []))
        puzzle_id = str(example.get("PuzzleId", ""))
        rating = int(example.get("Rating", 0))

        result = extract_puzzle_position_and_solution(fen, uci_moves)
        if result is None:
            return None

        puzzle_fen, last_move_san, solution_san = result

        maia_moves: list[HumanMove] | None = None
        if maia_engine is not None:
            board = chess.Board(puzzle_fen)
            maia_moves = get_maia_predictions(board, maia_engine, maia_top_n)

        # Generate refuted lines for Maia predictions that aren't the solution
        refuted_lines: list[RefutedLine] | None = None
        if maia_moves and stockfish_engine is not None:
            board = chess.Board(puzzle_fen)
            refuted_lines = generate_refuted_lines(
                board=board,
                maia_moves=maia_moves,
                solution_first_move=solution_san[0],
                stockfish_engine=stockfish_engine,
                limit=chess.engine.Limit(depth=refutation_depth),
            )

        prompt, lines_pgn = build_reasoning_prompt(
            puzzle_fen, last_move_san, themes, solution_san, maia_moves, refuted_lines
        )

        # Validate the pre-built PGN (should always pass - bug check)
        from chess_sandbox.puzzles_trainer.reasoning_verifier import validate_pgn_lines

        lines_pgn_valid, lines_pgn_illegal = validate_pgn_lines(puzzle_fen, lines_pgn)
        if not lines_pgn_valid:
            # Bug in build_lines_exploration_pgn - log and skip
            print(f"BUG: Pre-built PGN invalid: {lines_pgn_illegal[:3]}")
            return None

        trace = await generate_reasoning_trace(prompt, client, model)

        if trace is None:
            return None

        # Validate the LLM's commented version
        lines_with_comments_valid, lines_with_comments_illegal = validate_pgn_lines(
            puzzle_fen, trace.lines_pgn_with_comments
        )

        # Join trace for verification using existing string-based verifier
        joined = join_reasoning_trace(trace, lines_pgn)
        verification = verify_reasoning_trace(puzzle_fen, joined, solution_san)

        # Add lines validation to verification result
        verification.lines_exploration_valid = lines_with_comments_valid
        verification.lines_exploration_illegal = lines_with_comments_illegal

        return format_reasoning_example(
            puzzle_fen, last_move_san, themes, solution_san, trace, lines_pgn, puzzle_id, rating, verification
        )


def get_primary_theme(puzzle_themes: list[str]) -> str:
    """Get the primary theme for quota tracking.

    Returns the first theme that matches a quota category, or 'other' if none match.
    """
    themes_lower = {t.lower() for t in puzzle_themes}
    for theme in THEME_QUOTAS:
        if theme == "other":
            continue
        if theme.lower() in themes_lower:
            return theme
    return "other"


async def generate_reasoning_dataset(
    sample_size: int = 20,
    model: str = "openai/gpt-oss-20b:free",
    min_popularity: int = 80,
    max_rating: int | None = None,
    themes: tuple[str, ...] | None = None,
    max_concurrent: int = 5,
    base_url: str | None = None,
    api_key: str | None = None,
    balanced: bool = False,
    use_maia: bool = False,
    maia_top_n: int = 3,
) -> list[dict[str, str]]:
    """Generate reasoning dataset from Lichess puzzles."""
    effective_base_url = base_url or DEFAULT_BASE_URL
    effective_api_key = api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")

    if not effective_api_key and effective_base_url == DEFAULT_BASE_URL:
        raise ValueError("Set OPENROUTER_API_KEY or OPENAI_API_KEY environment variable")

    client = AsyncOpenAI(base_url=effective_base_url, api_key=effective_api_key or "dummy")
    semaphore = asyncio.Semaphore(max_concurrent)

    # Initialize engines if requested
    maia_engine: chess.engine.SimpleEngine | None = None
    stockfish_engine: chess.engine.SimpleEngine | None = None
    if use_maia:
        maia_config = MaiaConfig.default()
        maia_engine = maia_config.instantiate()
        print(f"Maia engine initialized (top {maia_top_n} predictions)")

        # Initialize Stockfish for refutation analysis
        stockfish_config = EngineConfig.stockfish(num_lines=1, depth=20)
        stockfish_engine = stockfish_config.instantiate()
        print("Stockfish engine initialized for refutation analysis")

    print(f"Loading dataset: {DATASET_ID}")
    dataset: Dataset = load_dataset(DATASET_ID, split="train")  # type: ignore[assignment]

    # Filter and sample puzzles
    themes_lower = [t.lower() for t in themes] if themes else None
    sampled: list[dict[str, Any]] = []

    # Theme quota tracking for balanced sampling
    theme_targets: dict[str, int] = {}
    theme_counts: dict[str, int] = {}
    if balanced:
        theme_targets = {theme: int(sample_size * quota) for theme, quota in THEME_QUOTAS.items()}
        theme_counts = {theme: 0 for theme in THEME_QUOTAS}
        print(f"Balanced sampling targets: {theme_targets}")

    for example in dataset:  # pyright: ignore[reportUnknownVariableType]
        if example["Popularity"] < min_popularity:  # pyright: ignore[reportCallIssue,reportArgumentType]
            continue
        if max_rating and example["Rating"] > max_rating:  # pyright: ignore[reportCallIssue,reportArgumentType]
            continue
        if themes_lower:
            puzzle_themes = example.get("Themes", [])  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType,reportAttributeAccessIssue]
            if not any(pt.lower() in themes_lower for pt in puzzle_themes):  # pyright: ignore[reportUnknownArgumentType,reportUnknownMemberType,reportUnknownVariableType]
                continue

        # Balanced sampling: check if this puzzle's theme quota is already full
        if balanced:
            puzzle_themes_list: list[str] = list(example.get("Themes", []))  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType,reportAttributeAccessIssue,reportUnknownArgumentType]
            primary_theme = get_primary_theme(puzzle_themes_list)
            if theme_counts[primary_theme] >= theme_targets[primary_theme]:
                continue  # Skip - quota for this theme is full
            theme_counts[primary_theme] += 1

        sampled.append(dict(example))  # pyright: ignore[reportUnknownArgumentType]

        # Check termination condition
        if balanced:
            # Stop when all quotas are filled
            if all(theme_counts[t] >= theme_targets[t] for t in THEME_QUOTAS):
                break
        elif len(sampled) >= sample_size:
            break

    if balanced:
        print(f"Sampled {len(sampled)} puzzles with theme distribution: {theme_counts}")
    else:
        print(f"Sampled {len(sampled)} puzzles")

    try:
        # Process puzzles concurrently
        tasks = [
            process_puzzle(ex, client, model, semaphore, maia_engine, stockfish_engine, maia_top_n) for ex in sampled
        ]
        results: list[dict[str, Any] | None] = await tqdm_asyncio.gather(*tasks, desc="Generating reasoning")  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]

        # Filter out failures
        valid_results: list[dict[str, Any]] = [r for r in results if r is not None]
        print(f"Generated {len(valid_results)}/{len(sampled)} reasoning traces")

        return valid_results
    finally:
        # Clean up engines
        if maia_engine is not None:
            maia_engine.quit()
        if stockfish_engine is not None:
            stockfish_engine.quit()


@click.command("generate-reasoning-dataset")
@click.option("--sample-size", type=int, default=20, help="Number of puzzles to process")
@click.option(
    "--model",
    type=str,
    default="openai/gpt-5-nano",
    help="OpenRouter model ID (e.g., openai/gpt-5-nano, openai/gpt-4o-mini)",
)
@click.option("--min-popularity", type=int, default=80, help="Minimum puzzle popularity")
@click.option("--max-rating", type=int, default=None, help="Maximum puzzle rating")
@click.option("--themes", type=str, default=None, help="Comma-separated theme filter")
@click.option("--max-concurrent", type=int, default=5, help="Max concurrent API requests")
@click.option("--base-url", type=str, default=None, help="API base URL (default: OpenRouter)")
@click.option("--api-key", type=str, default=None, help="API key (default: OPENROUTER_API_KEY env var)")
@click.option("--test-split", type=float, default=0.1, help="Fraction for test set")
@click.option("--push-to-hub", is_flag=True, help="Push dataset to HuggingFace Hub")
@click.option(
    "--dataset-id",
    type=str,
    default="pilipolio/chess-reasoning-traces",
    help="HuggingFace dataset ID",
)
@click.option(
    "--min-score",
    type=float,
    default=0.6,
    help="Minimum verification score to include (0.0-1.0)",
)
@click.option(
    "--include-failed",
    is_flag=True,
    help="Include all traces with verification metadata (for analysis)",
)
@click.option(
    "--balanced",
    is_flag=True,
    help="Use quota-based theme balancing to prevent mate-puzzle overfit",
)
@click.option(
    "--use-maia",
    is_flag=True,
    help="Include Maia human move predictions in candidate moves",
)
@click.option(
    "--maia-top-n",
    type=int,
    default=3,
    help="Number of top Maia predictions to include (default: 3)",
)
@click.option(
    "--export-lichess-study-id",
    type=str,
    default=None,
    help="Export results to Lichess study with this ID",
)
def main(
    sample_size: int,
    model: str,
    min_popularity: int,
    max_rating: int | None,
    themes: str | None,
    max_concurrent: int,
    base_url: str | None,
    api_key: str | None,
    test_split: float,
    push_to_hub: bool,
    dataset_id: str,
    min_score: float,
    include_failed: bool,
    balanced: bool,
    use_maia: bool,
    maia_top_n: int,
    export_lichess_study_id: str | None,
) -> None:
    """Generate chess puzzle reasoning traces using LLMs."""
    themes_tuple = tuple(themes.split(",")) if themes else None

    results = asyncio.run(
        generate_reasoning_dataset(
            sample_size=sample_size,
            model=model,
            min_popularity=min_popularity,
            max_rating=max_rating,
            themes=themes_tuple,
            max_concurrent=max_concurrent,
            base_url=base_url,
            api_key=api_key,
            balanced=balanced,
            use_maia=use_maia,
            maia_top_n=maia_top_n,
        )
    )

    if not results:
        print("No results generated")
        return

    # Filter by verification score unless include_failed is set
    total_generated = len(results)
    if not include_failed:
        results = [r for r in results if float(r.get("verification_score", 0.0)) >= min_score]
        filtered_count = total_generated - len(results)
        click.echo(f"Filtered {filtered_count} traces below min_score={min_score}")

    if not results:
        print("No results passed verification")
        return

    # Show verification stats
    scores = [float(r.get("verification_score", 0.0)) for r in results]
    avg_score = sum(scores) / len(scores) if scores else 0.0
    click.echo(f"Verification scores: avg={avg_score:.2f}, min={min(scores):.2f}, max={max(scores):.2f}")

    import random

    random.shuffle(results)
    test_size = math.ceil(len(results) * test_split)
    train_data = results[:-test_size] if test_size > 0 else results
    test_data = results[-test_size:] if test_size > 0 else []

    click.echo(f"Applying test split: {test_split} to {len(results)} examples")

    train_dataset = Dataset.from_list(train_data)  # pyright: ignore[reportUnknownMemberType]
    test_dataset = Dataset.from_list(test_data)  # pyright: ignore[reportUnknownMemberType]
    dataset_dict = DatasetDict({"train": train_dataset, "test": test_dataset})

    click.echo(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    if push_to_hub:
        dataset_dict.push_to_hub(dataset_id)  # pyright: ignore[reportUnknownMemberType]
        click.echo(f"Pushed to: https://huggingface.co/datasets/{dataset_id}")
    elif not export_lichess_study_id:
        import json

        click.echo(json.dumps(results, indent=2))

    if export_lichess_study_id:
        from chess_sandbox.lichess_export import export_traces_to_lichess

        click.echo(f"\nExporting {len(results)} traces to Lichess study: {export_lichess_study_id}")
        export_result = export_traces_to_lichess(results, export_lichess_study_id, model)
        click.echo(f"Exported: {export_result['exported_count']}, Skipped: {export_result['skipped_count']}")
        if export_result["response"]:
            click.echo(f"Study URL: https://lichess.org/study/{export_lichess_study_id}")


if __name__ == "__main__":
    main()
