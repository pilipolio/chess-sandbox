#!/usr/bin/env python3
"""
Lilypad-based Evaluation Loop for Chess Commentary

This module demonstrates how to use Lilypad to create an evaluation loop that:
1. Versions different commentator configurations
2. Traces all LLM calls automatically
3. Evaluates theme predictions against ground truth
4. Compares different versions systematically
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess
import lilypad
from mirascope.core.openai import openai_call
from pydantic import BaseModel, Field

from ..lichess import get_analysis_url
from .mirascope_commentator import MirascopeCommentator


class ThemeJudgement(BaseModel):
    """LLM judge output comparing predicted vs ground truth themes."""

    score: float = Field(description="Score from 0-100 indicating theme prediction quality")
    rationale: str = Field(description="Explanation of the score and comparison")


class EvaluationResult(BaseModel):
    """Complete evaluation result for one position."""

    config_name: str = Field(description="Configuration name")
    fen: str = Field(description="FEN string of the position")
    ground_truth_themes: list[str] = Field(description="Ground truth themes")
    predicted_themes: list[str] = Field(description="LLM predicted themes")
    score: float = Field(description="Judge score 0-100")
    rationale: str = Field(description="Judge rationale")
    trace_id: str | None = Field(default=None, description="Lilypad trace ID")


@dataclass
class EvalConfig:
    """Evaluation configuration with versioning."""

    name: str
    params: dict[str, Any]
    version: str = "v1"


# Use Mirascope's @openai_call decorator for the judge
def judge_themes(ground_truth: list[str], predicted: list[str], model: str = "gpt-4o-mini") -> ThemeJudgement:
    """
    Compare predicted themes with ground truth using semantic similarity.

    This function uses Mirascope's @openai_call decorator (added dynamically) and
    Lilypad's @lilypad.trace decorator for automatic versioning/tracing.
    """

    @openai_call(model=model, response_model=ThemeJudgement)  # type: ignore[misc]
    @lilypad.trace(versioning="automatic")  # type: ignore[misc]
    def _inner_call() -> str:
        return f"""You are an expert chess judge evaluating theme predictions.

Ground Truth Themes: {json.dumps(ground_truth)}
Predicted Themes: {json.dumps(predicted)}

Evaluate the predicted themes against ground truth on a scale of 0-100:
- 100: Perfect match (all themes captured, semantically equivalent)
- 80-99: Very good (captures main themes, minor differences in wording)
- 60-79: Good (captures most key themes, some missing or extra themes)
- 40-59: Fair (some overlap, but missing important themes or many incorrect ones)
- 20-39: Poor (little overlap, mostly incorrect themes)
- 0-19: Very poor (completely off, no relevant themes)

Consider:
1. Semantic similarity (e.g., "Opposition" matches "King Opposition")
2. Completeness (are key themes missing?)
3. Accuracy (are predicted themes actually present?)
4. Specificity (are themes appropriately detailed?)

Provide a score and clear rationale explaining your evaluation."""

    return _inner_call()


def load_ground_truth(jsonl_path: str) -> list[dict[str, Any]]:
    """Load ground truth data from JSONL file."""
    data: list[dict[str, Any]] = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_results(results: list[EvaluationResult], output_path: str) -> None:
    """Save evaluation results to JSONL file."""
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for result in results:
            f.write(result.model_dump_json() + "\n")


def print_summary(results_by_config: dict[str, list[EvaluationResult]]) -> None:
    """Print evaluation summary with average scores."""
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 70)

    for config_name, results in results_by_config.items():
        scores = [r.score for r in results]
        avg_score = sum(scores) / len(scores) if scores else 0

        print(f"\nConfiguration: {config_name}")
        print(f"  Positions Evaluated: {len(results)}")
        print(f"  Average Score: {avg_score:.2f}/100")
        print(f"  Score Range: {min(scores):.1f} - {max(scores):.1f}")
        print("-" * 70)


@lilypad.trace(versioning="automatic")
def evaluate_single_position(
    commentator: MirascopeCommentator,
    fen: str,
    ground_truth_themes: list[str],
    config_name: str,
    judge_model: str = "gpt-4o-mini",
) -> EvaluationResult:
    """
    Evaluate a single chess position.

    This function is wrapped with @lilypad.trace to track the entire evaluation
    process, including both the commentator and judge calls.
    """
    board = chess.Board(fen)

    print(f"\nAnalyzing position with {config_name}...")
    explanation = commentator.analyze(board)

    predicted_themes = explanation.themes

    print(f"Ground Truth: {ground_truth_themes}")
    print(f"Predicted: {predicted_themes}")

    # Judge the themes
    judgement = judge_themes(ground_truth=ground_truth_themes, predicted=predicted_themes, model=judge_model)

    print(f"Judge Score: {judgement.score:.1f}/100")
    print(f"Rationale: {judgement.rationale}")

    return EvaluationResult(
        config_name=config_name,
        fen=fen,
        ground_truth_themes=ground_truth_themes,
        predicted_themes=predicted_themes,
        score=judgement.score,
        rationale=judgement.rationale,
    )


def run_evaluation_loop(
    ground_truth_data: list[dict[str, Any]],
    configs: list[EvalConfig],
    judge_model: str = "gpt-4o-mini",
    max_positions: int | None = None,
) -> dict[str, list[EvaluationResult]]:
    """
    Run the evaluation loop for all configurations.

    This creates a data flywheel by:
    1. Running different commentator configurations (versioned)
    2. Evaluating predictions against ground truth (traced)
    3. Storing all results for comparison and analysis
    """
    results_by_config: dict[str, list[EvaluationResult]] = {}

    # Limit number of positions for testing
    positions_to_evaluate = ground_truth_data[:max_positions] if max_positions else ground_truth_data

    for config in configs:
        print(f"\n{'=' * 70}")
        print(f"CONFIGURATION: {config.name} ({config.version})")
        print(f"{'=' * 70}")
        print(f"Engine: depth={config.params['engine']['depth']}, num_lines={config.params['engine']['num_lines']}")
        print(f"LLM: {config.params['llm']}")
        print(f"{'=' * 70}")

        commentator = MirascopeCommentator.create(config.params)
        results: list[EvaluationResult] = []

        for i, item in enumerate(positions_to_evaluate, 1):
            fen = item["fen"]
            ground_truth_themes = item["themes"]

            print(f"\n{'=' * 70}")
            print(f"POSITION {i}/{len(positions_to_evaluate)}")
            print(f"{'=' * 70}")
            print(f"FEN: {fen}")
            print(f"Lichess: {get_analysis_url(fen)}")

            try:
                result = evaluate_single_position(
                    commentator=commentator,
                    fen=fen,
                    ground_truth_themes=ground_truth_themes,
                    config_name=config.name,
                    judge_model=judge_model,
                )

                results.append(result)

            except Exception as e:
                print(f"\n{'=' * 70}")
                print(f"ERROR: {e}")
                print(f"{'=' * 70}")
                continue

        results_by_config[config.name] = results

        # Save results for this config
        output_path = f"data/results/lilypad_eval_{config.name}.jsonl"
        save_results(results, output_path)
        print(f"\n{'=' * 70}")
        print(f"Results saved to: {output_path}")
        print(f"{'=' * 70}")

    return results_by_config


def main() -> None:
    """Main entry point for the Lilypad evaluation loop."""
    # Configure Lilypad
    lilypad_project_id = os.environ.get("LILYPAD_PROJECT_ID")
    lilypad_api_key = os.environ.get("LILYPAD_API_KEY")

    if lilypad_project_id and lilypad_api_key:
        print("Configuring Lilypad...")
        lilypad.configure(
            project_id=lilypad_project_id,
            api_key=lilypad_api_key,
            auto_llm=True,  # Automatically trace all LLM calls
        )
        print("Lilypad configured successfully!")
    else:
        print("Warning: LILYPAD_PROJECT_ID and LILYPAD_API_KEY not set.")
        print("Running without Lilypad tracing.")

    # Load ground truth data
    ground_truth_path = "data/processed/chessdotcom.jsonl"
    print(f"\nLoading ground truth from: {ground_truth_path}")
    ground_truth_data = load_ground_truth(ground_truth_path)
    print(f"Loaded {len(ground_truth_data)} positions")

    # Define evaluation configurations
    # Each configuration represents a different version to compare
    configs = [
        EvalConfig(
            name="gpt4o_mini_no_engine",
            version="v1",
            params={
                "engine": {"depth": 20, "num_lines": 0},
                "llm": {"model": "gpt-4o-mini"},
                "include_tactical_patterns": True,
                "lilypad": {"project_id": lilypad_project_id, "api_key": lilypad_api_key},
            },
        ),
        EvalConfig(
            name="gpt4o_mini_with_engine",
            version="v1",
            params={
                "engine": {"depth": 20, "num_lines": 5},
                "llm": {"model": "gpt-4o-mini"},
                "include_tactical_patterns": True,
                "lilypad": {"project_id": lilypad_project_id, "api_key": lilypad_api_key},
            },
        ),
        EvalConfig(
            name="gpt4o_with_engine_no_tactical",
            version="v1",
            params={
                "engine": {"depth": 20, "num_lines": 5},
                "llm": {"model": "gpt-4o"},
                "include_tactical_patterns": False,
                "lilypad": {"project_id": lilypad_project_id, "api_key": lilypad_api_key},
            },
        ),
    ]

    # Run evaluation loop (limit to 3 positions for POC)
    print("\n" + "=" * 70)
    print("STARTING LILYPAD EVALUATION LOOP POC")
    print("=" * 70)

    results_by_config = run_evaluation_loop(
        ground_truth_data=ground_truth_data, configs=configs, judge_model="gpt-4o-mini", max_positions=3
    )

    # Print summary
    print_summary(results_by_config)

    print("\n" + "=" * 70)
    print("Evaluation complete!")
    print("=" * 70)

    if lilypad_project_id and lilypad_api_key:
        print("\nView traces and results in your Lilypad dashboard:")
        print("https://lilypad.mirascope.com")


if __name__ == "__main__":
    main()
