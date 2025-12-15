"""LLM evaluation for chess puzzle reasoning with structured outputs and Weave tracking."""

import asyncio
import os

import click
import weave
from datasets import Dataset, load_dataset  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

DEFAULT_DATASET_ID = "pilipolio/chess-reasoning-traces"
DEFAULT_SAMPLE_SIZE = 100

MODAL_VLLM_URL = "https://pilipolio--chess-reasoning-vllm-serve.modal.run/v1"
OPENROUTER_URL = "https://openrouter.ai/api/v1"

MODELS: dict[str, str] = {
    "chess-reasoning": MODAL_VLLM_URL,
    # "openai/gpt-oss-20b:free": OPENROUTER_URL,
    "qwen/qwen3-32b": OPENROUTER_URL,
    "openai/gpt-5-mini": OPENROUTER_URL,
}

BENCHMARK_MODELS = list(MODELS.keys())


class ChessReasoningOutput(BaseModel):
    """Structured output for chess puzzle reasoning."""

    piece_positions: str = Field(description="Key pieces and their squares, e.g., 'White: Qh6, Re6. Black: Kh8, Qb6'")
    candidate_moves: list[str] = Field(description="List of candidate moves in SAN notation")
    themes: list[str] = Field(
        description="Tactical/strategic themes: fork, pin, skewer, back-rank, discovered attack, etc."
    )
    solution: list[str] = Field(description="Full solution as list of SAN moves")
    best_move: str = Field(description="The single best move (first move of solution) in SAN notation")


def normalize_move(move: str) -> str:
    """Normalize chess move for comparison."""
    import re

    move = move.strip().lower()
    move = re.sub(r"[+#=].*$", "", move)
    return move


def create_client(base_url: str) -> OpenAI:
    """Create OpenAI-compatible client for the given endpoint."""
    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")

    if not key and base_url == OPENROUTER_URL:
        raise ValueError("Set OPENROUTER_API_KEY or OPENAI_API_KEY environment variable")

    return OpenAI(base_url=base_url, api_key=key or "dummy")


class ReasoningModelOutput(BaseModel):
    """Output from reasoning model including raw reasoning trace."""

    best_move: str = Field(description="The best move extracted from model output")
    reasoning: str | None = Field(default=None, description="Raw reasoning trace from model")


class StructuredReasoningModel(weave.Model):
    """Weave model wrapper for chess reasoning with structured outputs."""

    model_name: str
    base_url: str

    @weave.op()
    def predict(self, question: str) -> ChessReasoningOutput | None:
        """Get structured model prediction for a question."""
        client = create_client(self.base_url)

        try:
            kwargs: dict[str, object] = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": question}],
                "response_format": ChessReasoningOutput,
                "max_tokens": 2048,
                "temperature": 0.7,
            }
            if "gpt-5" in self.model_name:
                kwargs["extra_body"] = {"reasoning": {"effort": "low"}}
            response = client.beta.chat.completions.parse(**kwargs)  # pyright: ignore[reportArgumentType]
            return response.choices[0].message.parsed
        except Exception as e:
            print(f"\nError evaluating example with {self.model_name}: {e}")
            return None


class PlainReasoningModel(weave.Model):
    """Weave model wrapper for plain text reasoning models (like fine-tuned Qwen3)."""

    model_name: str
    base_url: str

    @weave.op()
    def predict(self, question: str) -> ReasoningModelOutput | None:
        """Get plain text prediction with reasoning trace."""
        client = create_client(self.base_url)

        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": question}],
                max_tokens=2048,
                temperature=0.7,
            )
            message = response.choices[0].message
            content = message.content or ""
            reasoning = getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None)

            best_move = content.strip().split("\n")[0].strip() if content else ""

            return ReasoningModelOutput(best_move=best_move, reasoning=reasoning)
        except Exception as e:
            print(f"\nError evaluating example with {self.model_name}: {e}")
            return None


@weave.op()
def first_move_scorer(first_move: str, output: ChessReasoningOutput | None) -> dict[str, bool]:
    """Score model output against expected first move."""
    if output is None:
        return {"first_move_match": False}
    return {"first_move_match": normalize_move(output.best_move) == normalize_move(first_move)}


@weave.op()
def plain_first_move_scorer(first_move: str, output: ReasoningModelOutput | None) -> dict[str, bool]:
    """Score plain model output against expected first move."""
    if output is None:
        return {"first_move_match": False}
    return {"first_move_match": normalize_move(output.best_move) == normalize_move(first_move)}


@weave.op()
def reasoning_scorer(first_move: str, output: ReasoningModelOutput | None) -> dict[str, float | bool]:
    """Score reasoning quality."""
    if output is None or not output.reasoning:
        return {"has_reasoning": False, "reasoning_length": 0.0, "mentions_solution": False}

    return {
        "has_reasoning": True,
        "reasoning_length": float(len(output.reasoning)),
        "mentions_solution": first_move.lower() in output.reasoning.lower(),
    }


@weave.op()
def candidate_scorer(first_move: str, output: ChessReasoningOutput | None) -> dict[str, bool]:
    """Check if correct move is in candidate moves."""
    if output is None:
        return {"candidate_includes_solution": False}
    normalized_first = normalize_move(first_move)
    normalized_candidates = [normalize_move(m) for m in output.candidate_moves]
    return {"candidate_includes_solution": normalized_first in normalized_candidates}


@weave.op()
def theme_scorer(themes: list[str], output: ChessReasoningOutput | None) -> dict[str, float]:
    """Jaccard similarity between predicted and actual themes."""
    if output is None or not themes:
        return {"theme_jaccard": 0.0}
    expected = {t.lower() for t in themes}
    predicted = {t.lower() for t in output.themes}
    if not expected and not predicted:
        return {"theme_jaccard": 1.0}
    if not expected or not predicted:
        return {"theme_jaccard": 0.0}
    intersection = expected & predicted
    union = expected | predicted
    return {"theme_jaccard": len(intersection) / len(union)}


PLAIN_REASONING_MODELS = {"chess-reasoning"}


async def run_evaluation(
    dataset: Dataset,  # pyright: ignore[reportMissingTypeArgument]
    model_name: str,
) -> dict[str, object]:
    """Run evaluation on dataset using Weave."""
    base_url = MODELS.get(model_name)
    if not base_url:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")

    print(f"Evaluating {len(dataset)} examples with model: {model_name}")

    examples: list[dict[str, object]] = [dict(ex) for ex in dataset]  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]

    if model_name in PLAIN_REASONING_MODELS:
        model: weave.Model = PlainReasoningModel(model_name=model_name, base_url=base_url)
        scorers: list[object] = [plain_first_move_scorer, reasoning_scorer]
        eval_name = f"reasoning-plain-{len(examples)}:{model_name}"
    else:
        model = StructuredReasoningModel(model_name=model_name, base_url=base_url)
        scorers = [first_move_scorer, candidate_scorer, theme_scorer]
        eval_name = f"reasoning-structured-{len(examples)}:{model_name}"

    evaluation = weave.Evaluation(
        name=eval_name,
        dataset=examples,  # pyright: ignore[reportArgumentType]
        scorers=scorers,  # pyright: ignore[reportArgumentType]
    )

    results = await evaluation.evaluate(model)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    return results  # pyright: ignore[reportUnknownVariableType, reportReturnType]


async def run_benchmark(
    dataset: Dataset,  # pyright: ignore[reportMissingTypeArgument]
    models: list[str],
) -> dict[str, dict[str, object]]:
    """Run benchmark across multiple models in parallel."""
    tasks = [run_evaluation(dataset, model_name) for model_name in models]
    results_list = await asyncio.gather(*tasks)

    all_results = dict(zip(models, results_list, strict=True))
    for model_name, results in all_results.items():
        print_model_summary(model_name, results)

    return all_results


def get_metric(results: dict[str, object], scorer: str, metric: str) -> float | None:
    """Extract a metric value from Weave results."""
    scorer_data = results.get(scorer, {})
    if isinstance(scorer_data, dict):
        metric_info = scorer_data.get(metric, {})  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        if isinstance(metric_info, dict):
            return metric_info.get("true_fraction") or metric_info.get("mean", 0)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType, reportReturnType]
    return None


def print_model_summary(model: str, results: dict[str, object]) -> None:
    """Print summary for a single model from Weave evaluation results."""
    first_move = get_metric(results, "first_move_scorer", "first_move_match")
    candidates = get_metric(results, "candidate_scorer", "candidate_includes_solution")
    themes = get_metric(results, "theme_scorer", "theme_jaccard")

    parts = [f"{model}:"]
    if first_move is not None:
        parts.append(f"first_move={first_move * 100:.1f}%")
    if candidates is not None:
        parts.append(f"candidates={candidates * 100:.1f}%")
    if themes is not None:
        parts.append(f"themes={themes * 100:.1f}%")
    print(" ".join(parts))


def print_summary(results: dict[str, object]) -> None:
    """Print evaluation summary."""
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    first_move = get_metric(results, "first_move_scorer", "first_move_match")
    plain_first_move = get_metric(results, "plain_first_move_scorer", "first_move_match")
    candidates = get_metric(results, "candidate_scorer", "candidate_includes_solution")
    themes = get_metric(results, "theme_scorer", "theme_jaccard")
    has_reasoning = get_metric(results, "reasoning_scorer", "has_reasoning")
    reasoning_length = get_metric(results, "reasoning_scorer", "reasoning_length")
    mentions_solution = get_metric(results, "reasoning_scorer", "mentions_solution")

    if first_move is not None:
        print(f"First move accuracy: {first_move * 100:.1f}%")
    if plain_first_move is not None:
        print(f"First move accuracy: {plain_first_move * 100:.1f}%")
    if candidates is not None:
        print(f"Candidate includes solution: {candidates * 100:.1f}%")
    if themes is not None:
        print(f"Theme Jaccard similarity: {themes * 100:.1f}%")
    if has_reasoning is not None:
        print(f"Has reasoning: {has_reasoning * 100:.1f}%")
    if reasoning_length is not None:
        print(f"Avg reasoning length: {reasoning_length:.0f} chars")
    if mentions_solution is not None:
        print(f"Mentions solution in reasoning: {mentions_solution * 100:.1f}%")

    print("=" * 70)
    print("Full results available in Weave UI")


def print_benchmark_comparison(all_results: dict[str, dict[str, object]]) -> None:
    """Print comparison table for benchmark results."""
    print("\n" + "=" * 80)
    print("BENCHMARK COMPARISON")
    print("=" * 80)
    print(f"{'Model':<35} {'First Move':>12} {'Candidates':>12} {'Themes':>12}")
    print("-" * 80)

    for model, results in all_results.items():
        first_move = get_metric(results, "first_move_scorer", "first_move_match")
        candidates = get_metric(results, "candidate_scorer", "candidate_includes_solution")
        themes = get_metric(results, "theme_scorer", "theme_jaccard")

        fm_str = f"{first_move * 100:.1f}%" if first_move is not None else "N/A"
        cand_str = f"{candidates * 100:.1f}%" if candidates is not None else "N/A"
        theme_str = f"{themes * 100:.1f}%" if themes is not None else "N/A"

        print(f"{model:<35} {fm_str:>12} {cand_str:>12} {theme_str:>12}")

    print("=" * 80)
    print("Full results available in Weave UI")


@click.command()
@click.option("--dataset-id", default=DEFAULT_DATASET_ID, help="HuggingFace dataset ID")
@click.option("--model", default=None, help="Single model to evaluate")
@click.option("--models", default=None, help="Comma-separated models for benchmark")
@click.option("--benchmark", is_flag=True, help="Run full benchmark with default models")
@click.option("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE, help="Number of examples")
@click.option("--split", default="test", help="Dataset split to evaluate")
@click.option("--weave-project", default="chess-reasoning", help="Weave project name")
@click.option("--parallelism", type=int, default=3, help="Number of parallel Weave workers")
def main(
    dataset_id: str,
    model: str | None,
    models: str | None,
    benchmark: bool,
    sample_size: int,
    split: str,
    weave_project: str,
    parallelism: int,
) -> None:
    """Evaluate chess reasoning puzzle models with structured outputs."""
    os.environ["WEAVE_PARALLELISM"] = str(parallelism)
    weave.init(weave_project)

    print(f"Loading dataset: {dataset_id} (split: {split})")
    dataset: Dataset = load_dataset(dataset_id, split=split)  # pyright: ignore[reportAssignmentType]

    if sample_size:
        dataset = dataset.select(range(min(sample_size, len(dataset))))  # pyright: ignore[reportUnknownMemberType]

    if benchmark:
        model_list = BENCHMARK_MODELS
        all_results = asyncio.run(run_benchmark(dataset, model_list))
        print_benchmark_comparison(all_results)

    elif models:
        model_list = [m.strip() for m in models.split(",")]
        all_results = asyncio.run(run_benchmark(dataset, model_list))
        print_benchmark_comparison(all_results)

    elif model:
        results = asyncio.run(run_evaluation(dataset, model))
        print_summary(results)

    else:
        raise click.UsageError("Specify --model, --models, or --benchmark")


if __name__ == "__main__":
    main()
