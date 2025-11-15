#!/usr/bin/env python3
"""
Chess Theme Evaluation - Batch LLM Judge
Evaluates chess position theme predictions against ground truth using an LLM judge.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import chess
from openai import OpenAI
from pydantic import BaseModel, Field

from ..lichess import get_analysis_url
from ..logging_config import setup_logging
from .commentator import Commentator, print_explanation

logger = setup_logging(__name__)


class ThemeJudgement(BaseModel):
    """LLM judge output comparing predicted vs ground truth themes."""

    score: float = Field(description="Score from 0-100 indicating theme prediction quality")
    rationale: str = Field(description="Explanation of the score and comparison")


class EvaluationResult(BaseModel):
    """Complete evaluation result for one position."""

    config: str = Field(description="Configuration name")
    fen: str = Field(description="FEN string of the position")
    ground_truth_themes: List[str] = Field(description="Ground truth themes")
    predicted_themes: List[str] = Field(description="LLM predicted themes")
    score: float = Field(description="Judge score 0-100")
    rationale: str = Field(description="Judge rationale")


@dataclass
class EvalConfig:
    """Evaluation configuration."""

    name: str
    params: Dict[str, Any]


class ThemeJudge:
    """LLM judge to compare predicted themes with ground truth."""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def judge(self, ground_truth: List[str], predicted: List[str]) -> ThemeJudgement:
        """
        Compare predicted themes with ground truth using semantic similarity.
        Returns score and rationale.
        """
        prompt = f"""You are an expert chess judge evaluating theme predictions.

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

        response = self.client.responses.parse(model=self.model, input=prompt, text_format=ThemeJudgement)

        message = response.output[0]
        assert message.type == "message", "Unexpected response type"

        content = message.content[0]
        assert content.type == "output_text", "Unexpected content type"

        if not content.parsed:
            raise Exception("Could not parse LLM judge response")

        return content.parsed


def load_ground_truth(jsonl_path: str) -> List[Dict[str, Any]]:
    """Load ground truth data from JSONL file."""
    data: List[Dict[str, Any]] = []
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_results(results: List[EvaluationResult], output_path: str) -> None:
    """Save evaluation results to JSONL file."""
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for result in results:
            f.write(result.model_dump_json() + "\n")


def print_summary(results_by_config: Dict[str, List[EvaluationResult]]) -> None:
    """Print evaluation summary with average scores."""
    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION RESULTS SUMMARY")
    logger.info("=" * 70)

    for config_name, results in results_by_config.items():
        scores = [r.score for r in results]
        avg_score = sum(scores) / len(scores) if scores else 0

        logger.info(f"\nConfiguration: {config_name}")
        logger.info(f"  Positions Evaluated: {len(results)}")
        logger.info(f"  Average Score: {avg_score:.2f}/100")
        logger.info(f"  Score Range: {min(scores):.1f} - {max(scores):.1f}")
        logger.info("-" * 70)


def evaluate_position(
    commentator: Commentator,
    judge: ThemeJudge,
    fen: str,
    ground_truth_themes: List[str],
    config_name: str,
) -> EvaluationResult:
    board = chess.Board(fen)

    logger.info(f"\nAnalyzing with {config_name}...")
    explanation = commentator.analyze(board)
    print_explanation(explanation)

    predicted_themes = explanation.themes

    judgement = judge.judge(ground_truth_themes, predicted_themes)

    return EvaluationResult(
        config=config_name,
        fen=fen,
        ground_truth_themes=ground_truth_themes,
        predicted_themes=predicted_themes,
        score=judgement.score,
        rationale=judgement.rationale,
    )


def run_evaluation(
    ground_truth_data: List[Dict[str, Any]], configs: List[EvalConfig], judge: ThemeJudge
) -> Dict[str, List[EvaluationResult]]:
    """Run evaluation for all configurations."""
    results_by_config: Dict[str, List[EvaluationResult]] = {}

    for config in configs:
        logger.info(f"\n{'=' * 70}")
        logger.info(f"CONFIGURATION: {config.name}")
        logger.info(f"{'=' * 70}")
        logger.info(
            f"Engine: depth={config.params['engine']['depth']}, num_lines={config.params['engine']['num_lines']}"
        )
        logger.info(f"LLM: {config.params['llm']}")
        logger.info(f"{'=' * 70}")

        commentator = Commentator.create(config.params)
        results: List[EvaluationResult] = []

        for i, item in enumerate(ground_truth_data, 1):
            fen = item["fen"]
            ground_truth_themes = item["themes"]

            logger.info(f"\n{'=' * 70}")
            logger.info(f"POSITION {i}/{len(ground_truth_data)}")
            logger.info(f"{'=' * 70}")
            logger.info(f"Lichess: {get_analysis_url(fen)}")

            try:
                result = evaluate_position(
                    commentator,
                    judge,
                    fen,
                    ground_truth_themes,
                    config.name,
                )

                logger.info(f"\n{'=' * 70}")
                logger.info("RESULTS")
                logger.info(f"{'=' * 70}")
                logger.info(f"Ground Truth Themes: {ground_truth_themes}")
                logger.info(f"Predicted Themes: {result.predicted_themes}")
                logger.info(f"Judge Score: {result.score:.1f}/100")
                logger.info(f"Judge Rationale: {result.rationale}")

                results.append(result)

            except Exception as e:
                logger.info(f"\n{'=' * 70}")
                logger.info(f"ERROR: {e}")
                logger.info(f"{'=' * 70}")
                continue

        results_by_config[config.name] = results

        # Save results for this config
        output_path = f"data/results/eval_{config.name}.jsonl"
        save_results(results, output_path)
        logger.info(f"\n{'=' * 70}")
        logger.info(f"Results saved to: {output_path}")
        logger.info(f"{'=' * 70}")

    return results_by_config


def main() -> None:
    ground_truth_path = "data/processed/chessdotcom.jsonl"
    logger.info(f"Loading ground truth from: {ground_truth_path}")
    ground_truth_data = load_ground_truth(ground_truth_path)
    logger.info(f"Loaded {len(ground_truth_data)} positions")

    # Define evaluation configurations
    configs = [
        EvalConfig(
            name="gpt5_mini_low_no_engine",
            params={"engine": {"depth": 20, "num_lines": 0}, "llm": {"model": "gpt-5-mini", "reasoning_effort": "low"}},
        ),
        EvalConfig(
            name="gpt5_mini_low_with_engine",
            params={"engine": {"depth": 20, "num_lines": 5}, "llm": {"model": "gpt-5-mini", "reasoning_effort": "low"}},
        ),
        EvalConfig(
            name="gpt4o_with_engine", params={"engine": {"depth": 20, "num_lines": 5}, "llm": {"model": "gpt-4o"}}
        ),
        EvalConfig(
            name="gpt4o_no_engine", params={"engine": {"depth": 20, "num_lines": 0}, "llm": {"model": "gpt-4o"}}
        ),
    ]

    judge = ThemeJudge(model="gpt-4o-mini")

    results_by_config = run_evaluation(ground_truth_data, configs, judge)

    print_summary(results_by_config)

    logger.info("\n" + "=" * 70)
    logger.info("Evaluation complete!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
