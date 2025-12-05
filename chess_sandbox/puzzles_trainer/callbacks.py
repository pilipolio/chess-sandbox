"""Chess validation callback for multi-task SFT training."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import chess
from transformers import PreTrainedTokenizerBase, TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from chess_sandbox.puzzles_trainer.inference import batch_generate
from chess_sandbox.puzzles_trainer.prompts import (
    build_ascii_board_prompt,
    build_concept_detection_prompt,
    build_legal_captures_prompt,
    build_piece_captures_prompt,
    build_puzzle_prompt,
)

if TYPE_CHECKING:
    from datasets import Dataset  # pyright: ignore[reportMissingTypeStubs]


class ChessValidationCallback(TrainerCallback):
    """
    Callback to run validation during training for all task types.

    Validates predictions by:
    1. Generating predictions on test samples
    2. Computing exact match and task-specific metrics
    3. Logging metrics and examples to W&B
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        test_dataset: Dataset | None,
        num_validation_samples: int = 10,
        log_examples: int = 10,
        max_new_tokens: int | None = None,
        max_thinking_tokens: int | None = None,
    ):
        self.tokenizer = tokenizer
        self.test_dataset = test_dataset
        self.num_validation_samples = num_validation_samples
        self.log_examples = log_examples
        self.max_new_tokens = max_new_tokens
        self.max_thinking_tokens = max_thinking_tokens

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        """Run validation after each evaluation step."""
        model = kwargs.get("model")
        if model is None or self.test_dataset is None:
            return

        import wandb  # pyright: ignore[reportMissingImports]

        # Only log to W&B if there's an active run (initialized by Trainer)
        if wandb.run is None:  # pyright: ignore[reportUnknownMemberType]
            return

        print(f"\n{'='*60}")
        print(f"Running validation at step {state.global_step}")
        print(f"{'='*60}")

        was_training = model.training
        model.eval()

        # Sample from all task types
        num_samples = min(self.num_validation_samples, len(self.test_dataset))
        test_samples = self.test_dataset.select(range(num_samples))  # pyright: ignore[reportUnknownMemberType]

        validation_results = self._run_validation(model, test_samples)

        # Group results by task type
        results_by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in validation_results:
            results_by_type[r.get("task_type", "unknown")].append(r)

        # Compute overall and per-task metrics
        all_exact_matches = [r["exact_match"] for r in validation_results]
        metrics: dict[str, Any] = {
            "eval/num_samples": len(validation_results),
            "eval/exact_match": sum(all_exact_matches) / len(all_exact_matches) if all_exact_matches else 0,
        }

        print("\nValidation Results:")
        print(f"  Total samples: {metrics['eval/num_samples']}")
        print(f"  Overall exact match: {metrics['eval/exact_match']:.1%}")

        # Per-task metrics
        for task_type, results in results_by_type.items():
            exact_matches = [r["exact_match"] for r in results]
            task_exact_match = sum(exact_matches) / len(exact_matches) if exact_matches else 0
            metrics[f"eval/{task_type}/exact_match"] = task_exact_match
            metrics[f"eval/{task_type}/num_samples"] = len(results)

            # Task-specific metrics
            if task_type == "puzzle":
                legal_moves = [r.get("is_legal", False) for r in results]
                legal_rate = sum(legal_moves) / len(legal_moves) if legal_moves else 0
                metrics[f"eval/{task_type}/legal_move_rate"] = legal_rate
                print(f"  {task_type}: {task_exact_match:.1%} exact, {legal_rate:.1%} legal ({len(results)})")
            else:
                print(f"  {task_type}: {task_exact_match:.1%} exact ({len(results)})")

        wandb.log(metrics, step=state.global_step)  # pyright: ignore[reportUnknownMemberType]

        # Create examples table
        table_data: list[list[Any]] = []
        for r in validation_results[: self.log_examples]:
            table_data.append(
                [
                    r.get("task_type", ""),
                    r.get("fen", ""),
                    r.get("prompt", ""),
                    r.get("predicted", ""),
                    r.get("expected", ""),
                    r.get("exact_match", False),
                ]
            )

        table = wandb.Table(  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
            columns=["Task", "FEN", "Prompt", "Predicted", "Expected", "Exact Match"],
            data=table_data,
        )
        print(f"Logging {json.dumps(table_data, indent=2)} examples to W&B")
        wandb.log({"eval/examples": table})  # pyright: ignore[reportUnknownMemberType]

        print("Logged metrics and examples to W&B\n")

        if was_training:
            model.train()

    def _run_validation(self, model: Any, test_samples: Any) -> list[dict[str, Any]]:
        """Run validation on test samples."""
        validation_results: list[dict[str, Any]] = []

        # Build prompts using chat template
        prompts: list[str] = []
        for example in test_samples:
            task_type = example.get("task_type", "unknown")

            fen = example.get("fen", "")

            if task_type == "puzzle":
                question = build_puzzle_prompt(fen)
            elif task_type == "ascii_board":
                question = build_ascii_board_prompt(fen)
            elif task_type == "concept_detection":
                question = build_concept_detection_prompt(fen)
            elif task_type == "legal_captures":
                question = build_legal_captures_prompt(fen)
            elif task_type == "piece_captures":
                square = example.get("square", "")
                question = build_piece_captures_prompt(fen, square)
            else:
                question = example.get("question", "")
                if not question:
                    messages = example.get("messages", [])
                    if messages and len(messages) > 0:
                        question = messages[0].get("content", "")

            chat_messages = [{"role": "user", "content": question}]
            prompt = self.tokenizer.apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=True)  # pyright: ignore[reportUnknownMemberType]
            prompts.append(str(prompt))

        print(f"  Running inference on {len(prompts)} samples...")

        all_outputs = batch_generate(
            model=model,
            tokenizer=self.tokenizer,
            prompts=prompts,
            max_new_tokens=self.max_new_tokens,
            max_thinking_tokens=self.max_thinking_tokens,
        )

        print("  Validating outputs...")

        for example, output, prompt in zip(test_samples, all_outputs, prompts):
            task_type = example.get("task_type", "unknown")
            expected = self._get_expected(example)
            result = self._validate_output(task_type, example, output, expected)
            result["prompt"] = prompt
            validation_results.append(result)

        return validation_results

    def _get_expected(self, example: dict[str, Any]) -> str:
        """Extract expected output from example."""
        # First check for answer field
        if "answer" in example:
            return str(example["answer"])

        # Fall back to messages
        messages = example.get("messages", [])
        if len(messages) >= 2:
            return str(messages[1].get("content", ""))

        return ""

    def _validate_output(
        self,
        task_type: str,
        example: dict[str, Any],
        output: str,
        expected: str,
    ) -> dict[str, Any]:
        """Validate output based on task type."""
        result: dict[str, Any] = {
            "task_type": task_type,
            "fen": example.get("fen", ""),
            "predicted": output,
            "expected": expected,
            "exact_match": self._normalize(output) == self._normalize(expected),
        }

        # Task-specific validation
        if task_type == "puzzle":
            result.update(self._validate_puzzle(example, output, expected))

        return result

    def _normalize(self, text: str) -> str:
        """Normalize text for comparison."""
        return " ".join(text.lower().split())

    def _validate_puzzle(
        self,
        example: dict[str, Any],
        output: str,
        expected: str,
    ) -> dict[str, Any]:
        """Validate puzzle solution (UCI move)."""
        fen = example.get("fen", "")
        predicted_move = self._parse_uci_move(output)

        is_legal = False
        if fen and predicted_move and len(predicted_move) >= 4:
            try:
                board = chess.Board(fen)
                move = chess.Move.from_uci(predicted_move)
                is_legal = move in board.legal_moves
            except (ValueError, chess.InvalidMoveError):
                pass

        return {"is_legal": is_legal, "predicted_move": predicted_move}

    def _parse_uci_move(self, output: str) -> str:
        """Extract a single UCI move from model output."""
        output = output.strip()
        output = re.sub(r"\[.*?\]", "", output)
        output = re.sub(r"<.*?>", "", output)
        output = output.strip()

        # Pattern for UCI moves: from_square + to_square + optional_promotion
        uci_pattern = r"\b[a-h][1-8][a-h][1-8][qrbn]?\b"
        matches = re.findall(uci_pattern, output.lower())
        if matches:
            return matches[0]

        words = output.split()
        return words[0] if words else ""
