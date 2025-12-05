"""Lightweight LLM evaluation for chess puzzle tasks using OpenAI API."""

import os
from pathlib import Path

import click
from datasets import Dataset, load_dataset  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]
from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm

DEFAULT_DATASET_ID = "pilipolio/lichess-puzzle-tasks"
DEFAULT_MODEL = "gpt-oss-20b"


class EvalResult(BaseModel):
    """Single evaluation result."""

    task_type: str
    fen: str
    question: str
    expected: str
    predicted: str
    exact_match: bool


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    return answer.strip().lower()


ReasoningEffort = str  # "low" | "medium" | "high"


def evaluate_example(client: OpenAI, model: str, question: str, reasoning_effort: ReasoningEffort | None = None) -> str:
    """Get model prediction for a question."""
    kwargs: dict[str, object] = {
        "model": model,
        "input": question,
    }
    if reasoning_effort:
        kwargs["reasoning"] = {"effort": reasoning_effort}

    response = client.responses.create(**kwargs)  # pyright: ignore[reportArgumentType, reportUnknownVariableType, reportCallIssue]

    for item in response.output:  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        if item.type == "message":  # pyright: ignore[reportUnknownMemberType]
            for content in item.content:  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
                if content.type == "output_text":  # pyright: ignore[reportUnknownMemberType]
                    return content.text.strip()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    return ""


def run_evaluation(
    dataset_id: str = DEFAULT_DATASET_ID,
    model: str = DEFAULT_MODEL,
    sample_size: int | None = None,
    split: str = "test",
    reasoning_effort: ReasoningEffort | None = None,
) -> list[EvalResult]:
    """Run evaluation on dataset and return results."""
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    print(f"Loading dataset: {dataset_id} (split: {split})")
    dataset: Dataset = load_dataset(dataset_id, split=split)  # pyright: ignore[reportAssignmentType]

    if sample_size:
        dataset = dataset.select(range(min(sample_size, len(dataset))))  # pyright: ignore[reportUnknownMemberType]

    model_desc = model
    if reasoning_effort:
        model_desc += f" (reasoning: {reasoning_effort})"
    print(f"Evaluating {len(dataset)} examples with model: {model_desc}")

    results: list[EvalResult] = []

    for example in tqdm(dataset, desc="Evaluating"):  # pyright: ignore[reportUnknownVariableType]
        ex: dict[str, str] = dict(example)  # pyright: ignore[reportUnknownArgumentType]
        question = ex["question"]
        expected = ex["answer"]
        task_type = ex["task_type"]
        fen = ex["fen"]

        try:
            predicted = evaluate_example(client, model, question, reasoning_effort)
        except Exception as e:
            print(f"\nError evaluating example: {e}")
            predicted = ""

        exact_match = normalize_answer(predicted) == normalize_answer(expected)

        results.append(
            EvalResult(
                task_type=task_type,
                fen=fen,
                question=question,
                expected=expected,
                predicted=predicted,
                exact_match=exact_match,
            )
        )

    return results


def save_results(results: list[EvalResult], output_path: str) -> None:
    """Save results to JSONL file."""
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for result in results:
            f.write(result.model_dump_json() + "\n")

    print(f"Results saved to: {output_path}")


def print_summary(results: list[EvalResult]) -> None:
    """Print evaluation summary with per-task metrics."""
    if not results:
        print("No results to summarize.")
        return

    total = len(results)
    correct = sum(1 for r in results if r.exact_match)
    overall_rate = correct / total * 100

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total examples: {total}")
    print(f"Overall exact match: {correct}/{total} ({overall_rate:.1f}%)")
    print("-" * 60)

    task_metrics: dict[str, dict[str, int]] = {}
    for r in results:
        if r.task_type not in task_metrics:
            task_metrics[r.task_type] = {"total": 0, "correct": 0}
        task_metrics[r.task_type]["total"] += 1
        if r.exact_match:
            task_metrics[r.task_type]["correct"] += 1

    print("\nPer-task metrics:")
    for task_type, metrics in sorted(task_metrics.items()):
        rate = metrics["correct"] / metrics["total"] * 100
        print(f"  {task_type}: {metrics['correct']}/{metrics['total']} ({rate:.1f}%)")

    print("=" * 60)


@click.command()
@click.option("--dataset-id", default=DEFAULT_DATASET_ID, help="HuggingFace dataset ID")
@click.option("--model", default=DEFAULT_MODEL, help="OpenAI model name")
@click.option("--sample-size", type=int, default=None, help="Limit number of examples")
@click.option("--split", default="test", help="Dataset split to evaluate")
@click.option(
    "--reasoning-effort",
    type=click.Choice(["low", "medium", "high"]),
    default=None,
    help="Reasoning effort for thinking models",
)
@click.option("--output", type=str, default=None, help="Output path for JSONL results")
def main(
    dataset_id: str,
    model: str,
    sample_size: int | None,
    split: str,
    reasoning_effort: str | None,
    output: str | None,
) -> None:
    """Evaluate an OpenAI model on the puzzle task dataset."""
    results = run_evaluation(
        dataset_id=dataset_id,
        model=model,
        sample_size=sample_size,
        split=split,
        reasoning_effort=reasoning_effort,
    )

    print_summary(results)

    if output:
        save_results(results, output)


if __name__ == "__main__":
    main()
