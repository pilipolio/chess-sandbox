"""Validation callback for chess reasoning SFT training.

Evaluates reasoning trace outputs:
- Section completeness (5 expected sections)
- First move correctness (SAN format)
- Move legality
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from transformers import PreTrainedTokenizerBase, TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from chess_sandbox.puzzles_trainer.inference import batch_generate
from chess_sandbox.puzzles_trainer.reasoning_verifier import (
    extract_pgn_moves,
    extract_solution_section,
    normalize_move,
    parse_sections,
    validate_move_sequence,
)

if TYPE_CHECKING:
    from datasets import Dataset


class ReasoningValidationCallback(TrainerCallback):
    """Callback for validating reasoning trace outputs during training.

    Validates:
    1. Section structure (5 expected steps in <think> block)
    2. First move correctness against expected solution
    3. Move legality using python-chess
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        test_dataset: Dataset | None,
        num_validation_samples: int = 10,
        log_examples: int = 10,
        max_new_tokens: int = 3000,
    ):
        self.tokenizer = tokenizer
        self.test_dataset = test_dataset
        self.num_validation_samples = num_validation_samples
        self.log_examples = log_examples
        self.max_new_tokens = max_new_tokens

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        model = kwargs.get("model")
        if model is None or self.test_dataset is None:
            return

        import wandb

        if wandb.run is None:
            return

        print(f"\n{'='*60}")
        print(f"Running reasoning validation at step {state.global_step}")
        print(f"{'='*60}")

        was_training = model.training
        model.eval()

        num_samples = min(self.num_validation_samples, len(self.test_dataset))
        indices: list[int] = torch.randperm(len(self.test_dataset))[:num_samples].tolist()
        test_samples = self.test_dataset.select(indices)

        results = self._run_validation(model, test_samples)

        # Compute metrics
        first_move_correct = [r["first_move_correct"] for r in results]
        legal_move_rate = [r["legal_move"] for r in results]
        sections_counts = [sum(r["sections_found"].values()) for r in results]

        metrics: dict[str, Any] = {
            "validate/num_samples": len(results),
            "validate/first_move_accuracy": sum(first_move_correct) / len(first_move_correct)
            if first_move_correct
            else 0,
            "validate/legal_move_rate": sum(legal_move_rate) / len(legal_move_rate) if legal_move_rate else 0,
            "validate/avg_sections_found": sum(sections_counts) / len(sections_counts) if sections_counts else 0,
        }

        print("\nValidation Results:")
        print(f"  Samples: {metrics['validate/num_samples']}")
        print(f"  First move accuracy: {metrics['validate/first_move_accuracy']:.1%}")
        print(f"  Legal move rate: {metrics['validate/legal_move_rate']:.1%}")
        print(f"  Avg sections found: {metrics['validate/avg_sections_found']:.1f}/5")

        wandb.log(metrics, step=state.global_step)

        # Log examples table
        table_data: list[list[Any]] = []
        for r in results[: self.log_examples]:
            table_data.append(
                [
                    r["fen"],
                    r["source_url"],
                    r["expected_first_move"],
                    r["extracted_first_move"],
                    r["first_move_correct"],
                    r["legal_move"],
                    sum(r["sections_found"].values()),
                    r["output"],
                ]
            )

        table = wandb.Table(
            columns=["FEN", "Lichess", "Expected", "Extracted", "Correct", "Legal", "Sections", "Output"],
            data=table_data,
        )
        wandb.log({"validate/examples": table})

        print("Logged metrics and examples to W&B\n")

        if was_training:
            model.train()

    def _run_validation(self, model: Any, test_samples: Any) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []

        prompts: list[str] = []
        for example in test_samples:
            question = example.get("question", "")
            chat_messages = [{"role": "user", "content": question}]
            prompt = self.tokenizer.apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=True)
            prompts.append(str(prompt))

        print(f"  Running inference on {len(prompts)} samples...")

        outputs = batch_generate(
            model=model,
            tokenizer=self.tokenizer,
            prompts=prompts,
            max_new_tokens=self.max_new_tokens,
        )

        print("  Validating outputs...")

        for example, output in zip(test_samples, outputs):
            result = self._validate_output(example, output)
            results.append(result)

        return results

    def _validate_output(self, example: dict[str, Any], output: str) -> dict[str, Any]:
        fen = example.get("fen", "")
        source_url = example.get("source_url", "")
        expected_first_move = example.get("first_move", "")
        solution_str = example.get("solution", "")
        expected_solution = solution_str.split() if solution_str else []

        # Parse sections
        sections_found = parse_sections(output)

        # Extract first move from output
        extracted_first_move: str | None = None
        legal_move = False

        solution_section = extract_solution_section(output)
        if solution_section:
            pgn_moves = extract_pgn_moves(solution_section)
            valid_moves, _ = validate_move_sequence(fen, pgn_moves)
            if valid_moves:
                extracted_first_move = valid_moves[0]
                legal_move = True

        # Check first move correctness
        first_move_correct = False
        if extracted_first_move and expected_first_move:
            first_move_correct = normalize_move(extracted_first_move) == normalize_move(expected_first_move)

        return {
            "fen": fen,
            "source_url": source_url,
            "expected_first_move": expected_first_move,
            "expected_solution": expected_solution,
            "extracted_first_move": extracted_first_move,
            "first_move_correct": first_move_correct,
            "legal_move": legal_move,
            "sections_found": sections_found,
            "output": output,
        }
