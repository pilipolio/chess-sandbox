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
import click
from datasets import (  # pyright: ignore[reportMissingTypeStubs]
    Dataset,
    DatasetDict,
    load_dataset,  # pyright: ignore[reportMissingTypeStubs,reportUnknownVariableType]
)
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

from chess_sandbox.puzzles_trainer.reasoning_verifier import VerificationResult, verify_reasoning_trace

load_dotenv()

DATASET_ID = "Lichess/chess-puzzles"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


def extract_puzzle_position_and_solution(fen: str, uci_moves: str) -> tuple[str, str, list[str]] | None:
    """Extract puzzle position, opponent's last move, and solution.

    In Lichess puzzles, the first move is the opponent's setup move.
    The puzzle position is AFTER this move, and the solution starts from move 2.

    Returns:
        Tuple of (puzzle_fen, last_move_san, solution_san_list) or None if invalid
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

        # Remaining moves are the solution
        solution_san: list[str] = []
        for uci in moves[1:]:
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                break
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


def identify_candidate_moves(board: chess.Board) -> dict[str, list[str]]:
    """Identify forcing moves: checks and captures."""
    checks: list[str] = []
    captures: list[str] = []

    for move in board.legal_moves:
        san = board.san(move)
        if board.gives_check(move):
            checks.append(san)
        elif board.is_capture(move):
            captures.append(san)

    return {"checks": checks, "captures": captures}


REASONING_PROMPT_TEMPLATE = """Position (FEN): {puzzle_fen}

{ascii_board}

Pieces:
{piece_placement}

Context:
- Opponent just played: {last_move}
- Side to move: {side_to_move}
- Themes: {themes}

Candidate moves:
- Checks: {checks}
- Captures: {captures}

Solution moves: {solution}

Analyze this position following this exact format:

<think>
## Position Analysis
Summarize material balance, king safety, and key piece activity (2-3 bullet points)

## Tactical Assessment
List the main tactical theme and why the candidate forcing moves matter

## Solution
Annotate the solution in PGN style with {{curly bracket comments}} explaining each move.
Start from move {move_number}. Format: "{move_number}. {first_move} {{explanation}} ..."
</think>
{first_move}"""


def build_reasoning_prompt(
    puzzle_fen: str,
    last_move_san: str,
    themes: list[str],
    solution_san: list[str],
) -> str:
    """Build enhanced prompt with structured context."""
    board = chess.Board(puzzle_fen)
    ascii_board = str(board)
    piece_placement = build_piece_placement_summary(board)
    candidates = identify_candidate_moves(board)
    side_to_move = "White" if board.turn == chess.WHITE else "Black"
    move_number = board.fullmove_number

    checks_str = ", ".join(candidates["checks"]) if candidates["checks"] else "none"
    captures_str = ", ".join(candidates["captures"]) if candidates["captures"] else "none"

    return REASONING_PROMPT_TEMPLATE.format(
        puzzle_fen=puzzle_fen,
        ascii_board=ascii_board,
        piece_placement=piece_placement,
        last_move=last_move_san,
        side_to_move=side_to_move,
        themes=", ".join(themes) if themes else "none",
        checks=checks_str,
        captures=captures_str,
        solution=" ".join(solution_san),
        first_move=solution_san[0],
        move_number=move_number,
    )


async def generate_reasoning_trace(
    prompt: str,
    client: AsyncOpenAI,
    model: str = "gpt-oss-20b",
) -> str:
    """Generate reasoning trace using OpenAI-compatible API."""
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.7,
    )

    message = response.choices[0].message
    content = message.content or ""

    # Some reasoning models put output in a separate reasoning field
    if not content and hasattr(message, "reasoning"):
        reasoning = getattr(message, "reasoning", None)
        if reasoning:
            content = str(reasoning)

    return content


def format_reasoning_example(
    puzzle_fen: str,
    last_move_san: str,
    themes: list[str],
    solution_san: list[str],
    reasoning: str,
    puzzle_id: str,
    verification: VerificationResult | None = None,
) -> dict[str, Any]:
    """Format puzzle with reasoning for SFT training."""
    question = f"Position: {puzzle_fen}\nOpponent's last move: {last_move_san}\nFind the best move."
    answer = f"<think>\n{reasoning}\n</think>\n{solution_san[0]}"

    result: dict[str, Any] = {
        "question": question,
        "answer": answer,
        "fen": puzzle_fen,
        "last_move": last_move_san,
        "solution": " ".join(solution_san),
        "themes": themes,
        "first_move": solution_san[0],
        "reasoning": reasoning,
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
) -> dict[str, Any] | None:
    """Process a single puzzle: generate and verify reasoning trace."""
    async with semaphore:
        try:
            fen = str(example["FEN"])
            uci_moves = str(example["Moves"])
            themes = list(example.get("Themes", []))
            puzzle_id = str(example.get("PuzzleId", ""))

            result = extract_puzzle_position_and_solution(fen, uci_moves)
            if result is None:
                return None

            puzzle_fen, last_move_san, solution_san = result

            prompt = build_reasoning_prompt(puzzle_fen, last_move_san, themes, solution_san)
            output = await generate_reasoning_trace(prompt, client, model)

            if not output.strip():
                return None

            verification = verify_reasoning_trace(puzzle_fen, output, solution_san)

            return format_reasoning_example(
                puzzle_fen, last_move_san, themes, solution_san, output, puzzle_id, verification
            )

        except Exception as e:
            print(f"Error processing puzzle: {e}")
            return None


async def generate_reasoning_dataset(
    sample_size: int = 20,
    model: str = "openai/gpt-oss-20b:free",
    min_popularity: int = 80,
    max_rating: int | None = None,
    themes: tuple[str, ...] | None = None,
    max_concurrent: int = 5,
    base_url: str | None = None,
    api_key: str | None = None,
) -> list[dict[str, Any]]:
    """Generate reasoning dataset from Lichess puzzles."""
    effective_base_url = base_url or DEFAULT_BASE_URL
    effective_api_key = api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")

    if not effective_api_key and effective_base_url == DEFAULT_BASE_URL:
        raise ValueError("Set OPENROUTER_API_KEY or OPENAI_API_KEY environment variable")

    client = AsyncOpenAI(base_url=effective_base_url, api_key=effective_api_key or "dummy")
    semaphore = asyncio.Semaphore(max_concurrent)

    print(f"Loading dataset: {DATASET_ID}")
    dataset: Dataset = load_dataset(DATASET_ID, split="train")  # type: ignore[assignment]

    # Filter and sample puzzles
    themes_lower = [t.lower() for t in themes] if themes else None
    sampled: list[dict[str, Any]] = []

    for example in dataset:  # pyright: ignore[reportUnknownVariableType]
        if example["Popularity"] < min_popularity:  # pyright: ignore[reportCallIssue,reportArgumentType]
            continue
        if max_rating and example["Rating"] > max_rating:  # pyright: ignore[reportCallIssue,reportArgumentType]
            continue
        if themes_lower:
            puzzle_themes = example.get("Themes", [])  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType,reportAttributeAccessIssue]
            if not any(pt.lower() in themes_lower for pt in puzzle_themes):  # pyright: ignore[reportUnknownArgumentType,reportUnknownMemberType,reportUnknownVariableType]
                continue

        sampled.append(dict(example))  # pyright: ignore[reportUnknownArgumentType]
        if len(sampled) >= sample_size:
            break

    print(f"Sampled {len(sampled)} puzzles")

    # Process puzzles concurrently
    tasks = [process_puzzle(ex, client, model, semaphore) for ex in sampled]
    results: list[dict[str, Any] | None] = await tqdm_asyncio.gather(*tasks, desc="Generating reasoning")  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]

    # Filter out failures
    valid_results: list[dict[str, Any]] = [r for r in results if r is not None]
    print(f"Generated {len(valid_results)}/{len(sampled)} reasoning traces")

    return valid_results


@click.command("generate-reasoning-dataset")
@click.option("--sample-size", type=int, default=20, help="Number of puzzles to process")
@click.option(
    "--model",
    type=str,
    default="openai/gpt-oss-20b:free",
    help="OpenRouter model ID (e.g., openai/gpt-oss-20b:free, openai/gpt-4o-mini)",
)
@click.option("--min-popularity", type=int, default=80, help="Minimum puzzle popularity")
@click.option("--max-rating", type=int, default=None, help="Maximum puzzle rating")
@click.option("--themes", type=str, default=None, help="Comma-separated theme filter")
@click.option("--max-concurrent", type=int, default=5, help="Max concurrent API requests")
@click.option("--base-url", type=str, default=None, help="API base URL (default: OpenRouter)")
@click.option("--api-key", type=str, default=None, help="API key (default: OPENROUTER_API_KEY env var)")
@click.option("--test-split", type=float, default=0.2, help="Fraction for test set")
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
        )
    )

    if not results:
        print("No results generated")
        return

    # Filter by verification score unless include_failed is set
    total_generated = len(results)
    if not include_failed:
        results = [r for r in results if r.get("verification_score", 0.0) >= min_score]
        filtered_count = total_generated - len(results)
        click.echo(f"Filtered {filtered_count} traces below min_score={min_score}")

    if not results:
        print("No results passed verification")
        return

    # Show verification stats
    scores = [r.get("verification_score", 0.0) for r in results]
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

    # Show sample with verification info
    if results:
        print("\n--- Sample output ---")
        sample = results[0]
        print(f"FEN: {sample['fen']}")
        print(f"Themes: {sample['themes']}")
        print(f"Solution: {sample['solution']}")
        print(f"Verification score: {sample.get('verification_score', 'N/A')}")
        print(f"First move correct: {sample.get('first_move_correct', 'N/A')}")
        print(f"Sections found: {sample.get('sections_found', 'N/A')}")
        print(f"Illegal moves: {sample.get('illegal_moves', [])}")
        print(f"Question: {sample['question']}")
        print(f"Answer: {sample['answer']}")

    if push_to_hub:
        dataset_dict.push_to_hub(dataset_id)  # pyright: ignore[reportUnknownMemberType]
        print(f"Pushed to: https://huggingface.co/datasets/{dataset_id}")
    else:
        import json

        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
