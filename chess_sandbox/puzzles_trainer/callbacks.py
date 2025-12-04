"""Chess validation callback for puzzle SFT training."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import chess
import torch
from transformers import PreTrainedTokenizerBase, TrainerCallback, TrainerControl, TrainerState, TrainingArguments

if TYPE_CHECKING:
    from datasets import Dataset  # pyright: ignore[reportMissingTypeStubs]


class ChessValidationCallback(TrainerCallback):
    """
    Callback to run chess move validation during training evaluation.

    Validates puzzle solutions by:
    1. Generating predictions on test samples
    2. Validating moves using python-chess
    3. Logging metrics and examples to W&B
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        test_dataset: Dataset | None,
        num_validation_samples: int = 50,
        log_examples: int = 20,
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
        """Run chess validation after each evaluation step."""
        if not args.report_to or "wandb" not in args.report_to:
            return

        model = kwargs.get("model")
        if model is None or self.test_dataset is None:
            return

        import wandb  # pyright: ignore[reportMissingImports]

        print(f"\n{'='*60}")
        print(f"Running chess validation at step {state.global_step}")
        print(f"{'='*60}")

        was_training = model.training
        model.eval()

        num_samples = min(self.num_validation_samples, len(self.test_dataset))  # pyright: ignore[reportArgumentType]
        test_samples = self.test_dataset.select(range(num_samples))  # pyright: ignore[reportUnknownMemberType]

        validation_results = self._run_validation(model, test_samples)
        valid_results = [r for r in validation_results if r.get("valid", False)]

        if valid_results:
            metrics = {
                "chess_puzzle/num_samples": len(validation_results),
                "chess_puzzle/num_valid": len(valid_results),
                "chess_puzzle/exact_match_accuracy": sum(r["exact_match"] for r in valid_results) / len(valid_results),
                "chess_puzzle/legal_move_rate": sum(r["is_legal"] for r in valid_results) / len(valid_results),
                "chess_puzzle/format_error_rate": sum(r["format_error"] for r in valid_results) / len(valid_results),
            }

            print("\nChess Validation Results:")
            print(f"  Valid evaluations: {metrics['chess_puzzle/num_valid']}/{metrics['chess_puzzle/num_samples']}")
            print(f"  Exact match accuracy: {metrics['chess_puzzle/exact_match_accuracy']:.1%}")
            print(f"  Legal move rate: {metrics['chess_puzzle/legal_move_rate']:.1%}")
            print(f"  Format error rate: {metrics['chess_puzzle/format_error_rate']:.1%}")

            wandb.log(metrics, step=state.global_step)  # pyright: ignore[reportUnknownMemberType]

            # Create examples table
            table_data: list[list[Any]] = []
            for i, r in enumerate(validation_results[: self.log_examples]):
                sample: dict[str, Any] = dict(test_samples[i]) if i < len(test_samples) else {}  # pyright: ignore[reportUnknownArgumentType,reportArgumentType]

                error_type = "None"
                if not r.get("valid", False):
                    error_type = r.get("error", "Unknown")
                elif r.get("format_error", False):
                    error_type = "Format"
                elif not r.get("is_legal", False):
                    error_type = "Illegal"

                table_data.append(
                    [
                        r.get("fen", "")[:50],
                        sample.get("rating", "N/A"),
                        ", ".join(sample.get("themes", [])[:3]) if "themes" in sample else "N/A",
                        r.get("predicted_move", "")[:20],
                        r.get("expected_move", "")[:20],
                        r.get("exact_match", False),
                        r.get("is_legal", False),
                        error_type,
                    ]
                )

            table = wandb.Table(  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
                columns=[
                    "FEN",
                    "Rating",
                    "Themes",
                    "Predicted Move",
                    "Expected Move",
                    "Exact Match",
                    "Is Legal",
                    "Error Type",
                ],
                data=table_data,
            )
            wandb.log({"eval/examples": table}, step=state.global_step)  # pyright: ignore[reportUnknownMemberType]

            # Log histograms
            exact_matches = [1 if r["exact_match"] else 0 for r in valid_results]
            legal_moves = [1 if r["is_legal"] else 0 for r in valid_results]
            wandb.log(  # pyright: ignore[reportUnknownMemberType]
                {
                    "chess_puzzle/exact_match_distribution": wandb.Histogram(exact_matches),  # pyright: ignore[reportUnknownMemberType]
                    "chess_puzzle/legal_move_distribution": wandb.Histogram(legal_moves),  # pyright: ignore[reportUnknownMemberType]
                },
                step=state.global_step,
            )

            print("Logged metrics, examples table, and histograms to W&B\n")
        else:
            print("No valid validation results\n")

        if was_training:
            model.train()

    def _run_validation(self, model: Any, test_samples: Any) -> list[dict[str, Any]]:
        """Run chess validation on test samples."""
        validation_results: list[dict[str, Any]] = []

        # Build prompts using chat template
        prompts: list[str] = []
        for example in test_samples:
            messages = [{"role": "user", "content": example["question"]}]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)  # pyright: ignore[reportUnknownMemberType]
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
                    max_new_tokens=64,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,  # pyright: ignore[reportUnknownMemberType]
                )

                # Decode only the generated tokens
                for j, output in enumerate(outputs):
                    input_len = int(inputs["input_ids"][j].shape[0])  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
                    generated = self.tokenizer.decode(output[input_len:], skip_special_tokens=True)  # pyright: ignore[reportUnknownMemberType]
                    all_outputs.append(generated.strip())

        print("  Validating moves with python-chess...")

        for example, output in zip(test_samples, all_outputs):
            validation = self._validate_puzzle_solution(
                fen=str(example["fen"]),
                output=output,
                expected_move=str(example.get("first_move", example.get("answer", ""))),
            )
            validation["fen"] = example["fen"]
            validation_results.append(validation)

        return validation_results

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

    def _validate_puzzle_solution(
        self,
        fen: str,
        output: str,
        expected_move: str,
    ) -> dict[str, Any]:
        """Validate puzzle solution (single UCI move)."""
        try:
            board = chess.Board(fen)
        except Exception as e:
            return {
                "valid": False,
                "error": f"Invalid FEN: {e}",
                "exact_match": False,
                "is_legal": False,
                "format_error": True,
                "predicted_move": "",
                "expected_move": expected_move,
            }

        predicted_move = self._parse_uci_move(output)

        format_error = False
        is_legal = False

        if not predicted_move or len(predicted_move) < 4:
            format_error = True
        else:
            try:
                move = chess.Move.from_uci(predicted_move)
                is_legal = move in board.legal_moves
            except (ValueError, chess.InvalidMoveError):
                format_error = True

        exact_match = predicted_move == expected_move

        return {
            "valid": True,
            "exact_match": exact_match,
            "is_legal": is_legal,
            "format_error": format_error,
            "predicted_move": predicted_move,
            "expected_move": expected_move,
        }
