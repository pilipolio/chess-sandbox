"""LLM evaluation for chess puzzle tasks with Weave tracking."""

import asyncio
import os

import click
import weave
from datasets import Dataset, load_dataset  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEFAULT_DATASET_ID = "pilipolio/lichess-puzzle-tasks"
DEFAULT_SAMPLE_SIZE = 10

MODAL_VLLM_URL = "https://pilipolio--chess-puzzle-vllm-serve.modal.run/v1"
OPENROUTER_URL = "https://openrouter.ai/api/v1"

MODELS: dict[str, str] = {
    "chess-puzzle": MODAL_VLLM_URL,
    "openai/gpt-oss-20b:free": OPENROUTER_URL,
    "qwen/qwen3-32b": OPENROUTER_URL,
    "openai/gpt-5-mini": OPENROUTER_URL,
}

BENCHMARK_MODELS = list(MODELS.keys())


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    return answer.strip().lower()


def create_client(base_url: str) -> OpenAI:
    """Create OpenAI-compatible client for the given endpoint."""
    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")

    if not key and base_url == OPENROUTER_URL:
        raise ValueError("Set OPENROUTER_API_KEY or OPENAI_API_KEY environment variable")

    return OpenAI(base_url=base_url, api_key=key or "dummy")


class PuzzleModel(weave.Model):
    """Weave model wrapper for chess puzzle evaluation."""

    model_name: str
    base_url: str

    @weave.op()
    def predict(self, question: str) -> str:
        """Get model prediction for a question."""
        client = create_client(self.base_url)
        kwargs: dict[str, object] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": question}],
        }
        if self.base_url == MODAL_VLLM_URL:
            kwargs["max_tokens"] = 256
            kwargs["temperature"] = 0.7

        try:
            response = client.chat.completions.create(**kwargs)  # pyright: ignore[reportArgumentType, reportUnknownVariableType, reportCallIssue]
            content = response.choices[0].message.content  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            return content.strip() if content else ""  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        except Exception as e:
            print(f"\nError evaluating example with {self.model_name}: {e}")
            return ""


@weave.op()
def exact_match_scorer(answer: str, output: str) -> dict[str, bool]:
    """Score model output against expected answer."""
    return {"exact_match": normalize_answer(output) == normalize_answer(answer)}


async def run_evaluation(
    dataset: Dataset,  # pyright: ignore[reportMissingTypeArgument]
    model_name: str,
) -> dict[str, object]:
    """Run evaluation on dataset using Weave."""
    base_url = MODELS.get(model_name)
    if not base_url:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")

    print(f"Evaluating {len(dataset)} examples with model: {model_name}")

    examples: list[dict[str, str]] = [dict(ex) for ex in dataset]  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]

    model = PuzzleModel(model_name=model_name, base_url=base_url)
    eval_name = f"{dataset.info.dataset_name}-{len(examples)}:{model_name}"
    evaluation = weave.Evaluation(
        name=eval_name,
        dataset=examples,  # pyright: ignore[reportArgumentType]
        scorers=[exact_match_scorer],
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


def print_model_summary(model: str, results: dict[str, object]) -> None:
    """Print summary for a single model from Weave evaluation results."""
    exact_match_data = results.get("exact_match_scorer", {})
    if isinstance(exact_match_data, dict):
        exact_match_info = exact_match_data.get("exact_match", {})  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        if isinstance(exact_match_info, dict):
            mean = exact_match_info.get("true_fraction", 0)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            print(f"{model}: {mean * 100:.1f}% exact match")
            return
    print(f"{model}: Results available in Weave UI")


def print_summary(results: dict[str, object]) -> None:
    """Print evaluation summary."""
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    exact_match_data = results.get("exact_match_scorer", {})
    if isinstance(exact_match_data, dict):
        exact_match_info = exact_match_data.get("exact_match", {})  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        if isinstance(exact_match_info, dict):
            mean = exact_match_info.get("true_fraction", 0)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            print(f"Overall exact match: {mean * 100:.1f}%")
    print("=" * 60)
    print("Full results available in Weave UI")


def print_benchmark_comparison(all_results: dict[str, dict[str, object]]) -> None:
    """Print comparison table for benchmark results."""
    print("\n" + "=" * 60)
    print("BENCHMARK COMPARISON")
    print("=" * 60)
    print(f"{'Model':<40} {'Exact Match':>15}")
    print("-" * 60)

    for model, results in all_results.items():
        exact_match_data = results.get("exact_match_scorer", {})
        if isinstance(exact_match_data, dict):
            exact_match_info = exact_match_data.get("exact_match", {})  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            if isinstance(exact_match_info, dict):
                mean = exact_match_info.get("true_fraction", 0)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
                print(f"{model:<40} {mean * 100:>14.1f}%")
                continue
        print(f"{model:<40} {'N/A':>15}")

    print("=" * 60)
    print("Full results available in Weave UI")


@click.command()
@click.option("--dataset-id", default=DEFAULT_DATASET_ID, help="HuggingFace dataset ID")
@click.option("--model", default=None, help="Single model to evaluate")
@click.option("--models", default=None, help="Comma-separated models for benchmark")
@click.option("--benchmark", is_flag=True, help="Run full benchmark with default models")
@click.option("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE, help="Number of examples")
@click.option("--split", default="test", help="Dataset split to evaluate")
@click.option("--weave-project", default="chess-puzzles", help="Weave project name")
def main(
    dataset_id: str,
    model: str | None,
    models: str | None,
    benchmark: bool,
    sample_size: int,
    split: str,
    weave_project: str,
) -> None:
    """Evaluate chess puzzle models. Supports both Modal vLLM and OpenRouter endpoints."""
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
