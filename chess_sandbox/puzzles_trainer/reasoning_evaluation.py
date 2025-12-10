"""LLM evaluation for chess puzzle reasoning with Weave tracking."""

import asyncio
import os
import re

import click
import weave
from datasets import Dataset, load_dataset  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEFAULT_DATASET_ID = "pilipolio/chess-reasoning-traces"
DEFAULT_SAMPLE_SIZE = 100

MODAL_VLLM_URL = "https://pilipolio--chess-reasoning-vllm-serve.modal.run/v1"
OPENROUTER_URL = "https://openrouter.ai/api/v1"

MODELS: dict[str, str] = {
    "chess-reasoning": MODAL_VLLM_URL,
    "qwen/qwen3-32b": OPENROUTER_URL,
    "openai/gpt-4o-mini": OPENROUTER_URL,
}

BENCHMARK_MODELS = list(MODELS.keys())

BASELINE_PROMPT_SUFFIX = """

IMPORTANT: Respond with ONLY the best move in standard algebraic notation (e.g., Nf3, Qxg2#, O-O).
Do not explain your reasoning. Just output the single best move."""


def format_prompt_for_model(question: str, model_name: str) -> str:
    """Add format instructions for baseline models."""
    if model_name == "chess-reasoning":
        return question
    return question + BASELINE_PROMPT_SUFFIX


MOVE_PATTERN = re.compile(
    r"\b([KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?|O-O(?:-O)?)\b",
    re.IGNORECASE,
)


def extract_first_move(output: str) -> str:
    """Extract the first move from model output, handling various formats."""
    text = output.strip()

    # Handle <think>...</think> format from finetuned model
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()

    # Try to find a chess move pattern
    match = MOVE_PATTERN.search(text)
    if match:
        return match.group(1)

    # Fallback: return first token
    tokens = text.split()
    if tokens:
        return tokens[0].strip()
    return ""


def normalize_move(move: str) -> str:
    """Normalize chess move for comparison."""
    move = move.strip().lower()
    move = re.sub(r"[+#=].*$", "", move)
    return move


def create_client(base_url: str) -> OpenAI:
    """Create OpenAI-compatible client for the given endpoint."""
    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")

    if not key and base_url == OPENROUTER_URL:
        raise ValueError("Set OPENROUTER_API_KEY or OPENAI_API_KEY environment variable")

    return OpenAI(base_url=base_url, api_key=key or "dummy")


class ReasoningPuzzleModel(weave.Model):
    """Weave model wrapper for chess reasoning puzzle evaluation."""

    model_name: str
    base_url: str

    @weave.op()
    def predict(self, question: str) -> str:
        """Get model prediction for a question."""
        prompt = format_prompt_for_model(question, self.model_name)
        client = create_client(self.base_url)
        kwargs: dict[str, object] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.base_url == MODAL_VLLM_URL:
            kwargs["max_tokens"] = 1024
            kwargs["temperature"] = 0.7

        try:
            response = client.chat.completions.create(**kwargs)  # pyright: ignore[reportArgumentType, reportUnknownVariableType, reportCallIssue]
            content = response.choices[0].message.content  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            return content.strip() if content else ""  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        except Exception as e:
            print(f"\nError evaluating example with {self.model_name}: {e}")
            return ""


@weave.op()
def first_move_scorer(first_move: str, output: str) -> dict[str, bool]:
    """Score model output against expected first move."""
    predicted = extract_first_move(output)
    return {"first_move_match": normalize_move(predicted) == normalize_move(first_move)}


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

    model = ReasoningPuzzleModel(model_name=model_name, base_url=base_url)
    eval_name = f"reasoning-{len(examples)}:{model_name}"
    evaluation = weave.Evaluation(
        name=eval_name,
        dataset=examples,  # pyright: ignore[reportArgumentType]
        scorers=[first_move_scorer],
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
    scorer_data = results.get("first_move_scorer", {})
    if isinstance(scorer_data, dict):
        match_info = scorer_data.get("first_move_match", {})  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        if isinstance(match_info, dict):
            mean = match_info.get("true_fraction", 0)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            print(f"{model}: {mean * 100:.1f}% first move accuracy")
            return
    print(f"{model}: Results available in Weave UI")


def print_summary(results: dict[str, object]) -> None:
    """Print evaluation summary."""
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    scorer_data = results.get("first_move_scorer", {})
    if isinstance(scorer_data, dict):
        match_info = scorer_data.get("first_move_match", {})  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        if isinstance(match_info, dict):
            mean = match_info.get("true_fraction", 0)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            print(f"Overall first move accuracy: {mean * 100:.1f}%")
    print("=" * 60)
    print("Full results available in Weave UI")


def print_benchmark_comparison(all_results: dict[str, dict[str, object]]) -> None:
    """Print comparison table for benchmark results."""
    print("\n" + "=" * 60)
    print("BENCHMARK COMPARISON")
    print("=" * 60)
    print(f"{'Model':<40} {'First Move':>15}")
    print("-" * 60)

    for model, results in all_results.items():
        scorer_data = results.get("first_move_scorer", {})
        if isinstance(scorer_data, dict):
            match_info = scorer_data.get("first_move_match", {})  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            if isinstance(match_info, dict):
                mean = match_info.get("true_fraction", 0)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
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
@click.option("--weave-project", default="chess-reasoning", help="Weave project name")
def main(
    dataset_id: str,
    model: str | None,
    models: str | None,
    benchmark: bool,
    sample_size: int,
    split: str,
    weave_project: str,
) -> None:
    """Evaluate chess reasoning puzzle models."""
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
