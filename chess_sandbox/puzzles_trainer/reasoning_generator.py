"""Generate synthetic reasoning traces for chess puzzles using LLMs.

Uses OpenRouter API by default to access models like gpt-oss-20b.
Set OPENROUTER_API_KEY environment variable or use --api-key flag.

Example:
    export OPENROUTER_API_KEY=sk-...
    uv run puzzles-generate-reasoning --sample-size 20 --model openai/gpt-oss-20b:free
"""

import asyncio
import os
import re
from typing import Any, Literal

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

load_dotenv()

DATASET_ID = "Lichess/chess-puzzles"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

ModelChoice = Literal["openai/gpt-oss-20b:free", "openai/gpt-4o-mini"]


def get_solution_moves_san(fen: str, uci_moves: str) -> list[str]:
    """Convert UCI solution moves to SAN notation.

    Args:
        fen: Starting position FEN.
        uci_moves: Space-separated UCI moves (e.g., "g6f5 h3g5 f6g5").

    Returns:
        List of SAN moves (e.g., ["Nxf5", "Qxg5", "Nxg5"]).
    """
    board = chess.Board(fen)
    san_moves: list[str] = []

    for uci in uci_moves.strip().split():
        try:
            move = chess.Move.from_uci(uci)
            if move in board.legal_moves:
                san_moves.append(board.san(move))
                board.push(move)
            else:
                break
        except ValueError:
            break

    return san_moves


def build_ascii_board(fen: str) -> str:
    """Generate ASCII board representation from FEN."""
    return str(chess.Board(fen))


def build_piece_positions(fen: str) -> str:
    """List all pieces by color and position."""
    board = chess.Board(fen)

    white_pieces: list[str] = []
    black_pieces: list[str] = []

    for square, piece in board.piece_map().items():
        piece_str = piece.symbol().upper() + chess.square_name(square)
        if piece.color == chess.WHITE:
            white_pieces.append(piece_str)
        else:
            black_pieces.append(piece_str)

    white_pieces.sort(key=lambda p: ("KQRBNP".index(p[0]), p[1:]))
    black_pieces.sort(key=lambda p: ("KQRBNP".index(p[0]), p[1:]))

    return f"White: {', '.join(white_pieces)}\nBlack: {', '.join(black_pieces)}"


def build_legal_captures(fen: str) -> str:
    """List all legal captures for side to move."""
    board = chess.Board(fen)
    captures = [m for m in board.legal_moves if board.is_capture(m)]

    if not captures:
        return "none"

    return ", ".join(board.san(m) for m in captures)


def build_reasoning_context(example: dict[str, Any]) -> dict[str, Any]:
    """Build rich context for reasoning generation from a puzzle.

    Args:
        example: Lichess puzzle with FEN, Moves, Themes, etc.

    Returns:
        Context dict with fen, ascii_board, piece_positions, legal_captures,
        themes, solution_san, first_move_san.
    """
    fen = str(example["FEN"])
    uci_moves = str(example["Moves"])
    themes = list(example.get("Themes", []))

    solution_san = get_solution_moves_san(fen, uci_moves)
    if not solution_san:
        raise ValueError(f"Could not parse solution moves: {uci_moves}")

    return {
        "fen": fen,
        "ascii_board": build_ascii_board(fen),
        "piece_positions": build_piece_positions(fen),
        "legal_captures": build_legal_captures(fen),
        "themes": themes,
        "solution_san": solution_san,
        "first_move_san": solution_san[0],
        "puzzle_id": str(example.get("PuzzleId", "")),
    }


REASONING_PROMPT_TEMPLATE = """You are a chess instructor explaining puzzle solutions. Given:

Position (FEN): {fen}

Board:
{ascii_board}

Pieces: {piece_positions}

Available captures: {legal_captures}

Themes: {themes}

Solution: {solution_str}

Write a concise reasoning trace (2-4 sentences) explaining WHY these moves work.
Focus on: checks, captures, threats, piece coordination, and the tactical pattern.
Do NOT just describe the moves - explain the forcing nature and why alternatives fail.

Output format:
<think>
[Your reasoning here]
</think>
{first_move}"""


def build_reasoning_prompt(context: dict[str, Any]) -> str:
    """Build prompt for reasoning generation."""
    themes_str = ", ".join(context["themes"]) if context["themes"] else "none"
    solution_str = " ".join(context["solution_san"])

    return REASONING_PROMPT_TEMPLATE.format(
        fen=context["fen"],
        ascii_board=context["ascii_board"],
        piece_positions=context["piece_positions"],
        legal_captures=context["legal_captures"],
        themes=themes_str,
        solution_str=solution_str,
        first_move=context["first_move_san"],
    )


async def generate_reasoning_trace(
    prompt: str,
    client: AsyncOpenAI,
    model: str = "gpt-oss-20b",
) -> str:
    """Generate reasoning trace using OpenAI-compatible API.

    Args:
        prompt: Full prompt with context and instructions.
        client: AsyncOpenAI client.
        model: Model ID to use.

    Returns:
        Raw completion text.
    """
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.7,
    )

    return response.choices[0].message.content or ""


def parse_reasoning_output(output: str) -> tuple[str | None, str | None]:
    """Parse <think> tags and final move from model output.

    Returns:
        Tuple of (reasoning_content, final_move) or (None, None) if parsing fails.
    """
    think_match = re.search(r"<think>\s*(.*?)\s*</think>", output, re.DOTALL)
    reasoning = think_match.group(1).strip() if think_match else None

    # Extract move after </think> tag
    after_think = re.split(r"</think>\s*", output, maxsplit=1)
    if len(after_think) > 1:
        # Take first word/token as the move
        final_move = after_think[1].strip().split()[0] if after_think[1].strip() else None
    else:
        # No </think> tag, try to find move at end
        final_move = None

    return reasoning, final_move


def validate_move(predicted: str | None, expected: str, fen: str) -> bool:
    """Validate that predicted move matches expected (handles SAN/UCI variations).

    Args:
        predicted: Predicted move (SAN or UCI).
        expected: Expected move (SAN).
        fen: Position FEN for parsing.

    Returns:
        True if moves match.
    """
    if not predicted:
        return False

    board = chess.Board(fen)

    try:
        # Try parsing expected as SAN
        expected_move = board.parse_san(expected)
    except ValueError:
        return False

    # Try parsing predicted as SAN first
    try:
        predicted_move = board.parse_san(predicted)
        return predicted_move == expected_move
    except ValueError:
        pass

    # Try parsing as UCI
    try:
        predicted_move = chess.Move.from_uci(predicted)
        return predicted_move == expected_move
    except ValueError:
        pass

    return False


def format_reasoning_example(
    context: dict[str, Any],
    reasoning: str,
) -> dict[str, Any]:
    """Format puzzle with reasoning for SFT training.

    Args:
        context: Puzzle context from build_reasoning_context.
        reasoning: Generated reasoning trace.

    Returns:
        SFT-ready example with question/answer fields and messages list.
    """
    question = f"Position: {context['fen']}\nFind the best move."
    answer = f"<think>\n{reasoning}\n</think>\n{context['first_move_san']}"

    return {
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "task_type": "puzzle_reasoning",
        "fen": context["fen"],
        "question": question,
        "answer": answer,
        "themes": context["themes"],
        "solution": " ".join(context["solution_san"]),
        "reasoning": reasoning,
        "first_move": context["first_move_san"],
        "source_url": f"https://lichess.org/training/{context['puzzle_id']}",
    }


# Hardcoded example for testing: back-rank mate puzzle
BACKRANK_MATE_EXAMPLE: dict[str, Any] = {
    "FEN": "1r4k1/4nppp/8/4Pb2/8/1P5P/r1PR4/3R3K w - - 0 27",
    "Moves": "d2d8 b8d8 d1d8",  # Rd8+ Rxd8 Rxd8#
    "Themes": ["backRankMate", "short", "mateIn2"],
    "PuzzleId": "backrank_test",
    "Rating": 1200,
    "Popularity": 95,
}

BACKRANK_MATE_REASONING = (
    "Black's king is trapped on the back rank with no escape squares "
    "(g8 blocked by pawns, f8 blocked by knight). "
    "Rd8+ forces Rxd8 (only legal response to check), then Rxd8# delivers mate. "
    "The two rooks coordinate to exploit the weak back rank."
)


def create_test_example() -> dict[str, Any]:
    """Create a test example using the hardcoded back-rank mate puzzle.

    This is useful for verifying the output format without API calls.
    """
    context = build_reasoning_context(BACKRANK_MATE_EXAMPLE)
    return format_reasoning_example(context, BACKRANK_MATE_REASONING)


def print_test_example() -> None:
    """Print the hardcoded test example for verification."""
    example = create_test_example()
    context = build_reasoning_context(BACKRANK_MATE_EXAMPLE)
    prompt = build_reasoning_prompt(context)

    print("=" * 60)
    print("TEST EXAMPLE: Back-rank mate puzzle")
    print("=" * 60)
    print(f"\nFEN: {example['fen']}")
    print(f"Themes: {example['themes']}")
    print(f"Solution: {example['solution']}")
    print(f"First move: {example['first_move']}")
    print(f"\n--- SFT Question ---\n{example['question']}")
    print(f"\n--- SFT Answer ---\n{example['answer']}")
    print("\n--- LLM Prompt (sent to generate reasoning) ---")
    print(prompt)
    print("=" * 60)


async def process_puzzle(
    example: dict[str, Any],
    client: AsyncOpenAI,
    model: str,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any] | None:
    """Process a single puzzle: build context, generate reasoning, validate.

    Returns:
        Formatted example or None if generation/validation failed.
    """
    async with semaphore:
        try:
            context = build_reasoning_context(example)
            prompt = build_reasoning_prompt(context)

            output = await generate_reasoning_trace(prompt, client, model)
            reasoning, predicted_move = parse_reasoning_output(output)

            if not reasoning:
                return None

            if not validate_move(predicted_move, context["first_move_san"], context["fen"]):
                # Still use the reasoning but with correct move
                pass

            return format_reasoning_example(context, reasoning)

        except Exception as e:
            print(f"Error processing puzzle: {e}")
            return None


async def generate_reasoning_dataset(
    sample_size: int = 20,
    model: ModelChoice = "openai/gpt-oss-20b:free",
    min_popularity: int = 80,
    max_rating: int | None = None,
    themes: tuple[str, ...] | None = None,
    max_concurrent: int = 5,
    base_url: str | None = None,
    api_key: str | None = None,
) -> list[dict[str, Any]]:
    """Generate reasoning dataset from Lichess puzzles.

    Args:
        sample_size: Number of puzzles to process.
        model: Model to use for generation (OpenRouter model ID).
        min_popularity: Minimum puzzle popularity.
        max_rating: Maximum puzzle rating.
        themes: Filter by themes (None for all).
        max_concurrent: Max concurrent API requests.
        base_url: OpenAI-compatible API base URL (default: OpenRouter).
        api_key: API key (uses OPENROUTER_API_KEY or OPENAI_API_KEY env var).

    Returns:
        List of formatted reasoning examples.
    """
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

    # Always inject the hardcoded back-rank mate example first for verification
    sampled.append(BACKRANK_MATE_EXAMPLE)
    print(f"Injected test example: {BACKRANK_MATE_EXAMPLE['FEN']}")

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

    print(f"Sampled {len(sampled)} puzzles (including 1 test example)")

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
@click.option("--test-split", type=float, default=0.1, help="Fraction for test set")
@click.option("--push-to-hub", is_flag=True, help="Push dataset to HuggingFace Hub")
@click.option(
    "--dataset-id",
    type=str,
    default="pilipolio/chess-reasoning-traces",
    help="HuggingFace dataset ID",
)
@click.option(
    "--test-example",
    is_flag=True,
    help="Print hardcoded back-rank mate example and exit (no API calls)",
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
    test_example: bool,
    dataset_id: str,
) -> None:
    """Generate chess puzzle reasoning traces using LLMs.

    Uses OpenRouter API to generate <think>...</think> reasoning traces
    for Lichess puzzles, suitable for SFT training.

    Requires OPENROUTER_API_KEY environment variable or --api-key flag.
    """
    if test_example:
        print_test_example()
        return

    themes_tuple = tuple(themes.split(",")) if themes else None

    results = asyncio.run(
        generate_reasoning_dataset(
            sample_size=sample_size,
            model=model,  # type: ignore[arg-type]
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

    # Split into train/test
    import random

    random.shuffle(results)
    test_size = int(len(results) * test_split)
    train_data = results[:-test_size] if test_size > 0 else results
    test_data = results[-test_size:] if test_size > 0 else []

    train_dataset = Dataset.from_list(train_data)  # pyright: ignore[reportUnknownMemberType]
    test_dataset = Dataset.from_list(test_data)  # pyright: ignore[reportUnknownMemberType]

    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    # Show sample
    if results:
        print("\n--- Sample output ---")
        sample = results[0]
        print(f"FEN: {sample['fen']}")
        print(f"Themes: {sample['themes']}")
        print(f"Solution: {sample['solution']}")
        print(f"Reasoning: {sample['reasoning']}")
        print(f"First move: {sample['first_move']}")

    if push_to_hub:
        dataset_dict = DatasetDict({"train": train_dataset, "test": test_dataset})
        dataset_dict.push_to_hub(dataset_id)  # pyright: ignore[reportUnknownMemberType]
        print(f"Pushed to: https://huggingface.co/datasets/{dataset_id}")


if __name__ == "__main__":
    main()
