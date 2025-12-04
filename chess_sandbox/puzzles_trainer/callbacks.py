"""Chess validation callback for multi-task SFT training."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import chess
import torch
from transformers import PreTrainedTokenizerBase, TrainerCallback, TrainerControl, TrainerState, TrainingArguments

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
    ):
        self.tokenizer = tokenizer
        self.test_dataset = test_dataset
        self.num_validation_samples = num_validation_samples
        self.log_examples = log_examples

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        """Run validation after each evaluation step."""
        if not args.report_to or "wandb" not in args.report_to:
            return

        model = kwargs.get("model")
        if model is None or self.test_dataset is None:
            return

        import wandb  # pyright: ignore[reportMissingImports]

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
                    r.get("fen", "")[:40],
                    r.get("predicted", "")[:60],
                    r.get("expected", "")[:60],
                    r.get("exact_match", False),
                ]
            )

        table = wandb.Table(  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
            columns=["Task", "FEN", "Predicted", "Expected", "Exact Match"],
            data=table_data,
        )
        print(f"Logging {json.dumps(table_data, indent=2)} examples to W&B")
        print(table_data)
        wandb.log({"eval/examples": table}, step=state.global_step)  # pyright: ignore[reportUnknownMemberType]

        print("Logged metrics and examples to W&B\n")

        if was_training:
            model.train()

    def _run_validation(self, model: Any, test_samples: Any) -> list[dict[str, Any]]:
        """Run validation on test samples."""
        validation_results: list[dict[str, Any]] = []

        # Build prompts using chat template
        prompts: list[str] = []
        for example in test_samples:
            question = example.get("question", "")
            if not question:
                # Fall back to extracting from messages
                messages = example.get("messages", [])
                if messages and len(messages) > 0:
                    question = messages[0].get("content", "")

            chat_messages = [{"role": "user", "content": question}]
            prompt = self.tokenizer.apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=True)  # pyright: ignore[reportUnknownMemberType]
            prompts.append(str(prompt))

        # Batch inference
        batch_size = 8
        all_outputs: list[str] = []

        print(f"  Running inference on {len(prompts)} samples...")

        with torch.no_grad():
            for i in range(0, len(prompts), batch_size):
                batch = prompts[i : i + batch_size]

                inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True)  # pyright: ignore[reportUnknownMemberType]
                inputs = {k: v.to(model.device) for k, v in inputs.items()}  # pyright: ignore[reportUnknownVariableType,reportUnknownMemberType]

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=128,  # Increased for longer outputs like ascii_board
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,  # pyright: ignore[reportUnknownMemberType]
                )

                # Decode only the generated tokens
                for j, output in enumerate(outputs):
                    input_len = int(inputs["input_ids"][j].shape[0])  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
                    generated = self.tokenizer.decode(output[input_len:], skip_special_tokens=True)  # pyright: ignore[reportUnknownMemberType]
                    # Strip <think>...</think> blocks (Qwen3 style reasoning)
                    generated = re.sub(r"<think>.*?</think>", "", generated, flags=re.DOTALL)
                    all_outputs.append(generated.strip())

        print("  Validating outputs...")

        for example, output in zip(test_samples, all_outputs):
            task_type = example.get("task_type", "unknown")
            expected = self._get_expected(example)
            result = self._validate_output(task_type, example, output, expected)
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
