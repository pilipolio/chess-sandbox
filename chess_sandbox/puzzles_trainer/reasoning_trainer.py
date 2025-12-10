"""Training logic for chess puzzle reasoning SFT.

Fine-tunes models on structured reasoning traces with <think> sections.
"""

from pathlib import Path
from typing import Any, cast

import click
import torch
from datasets import load_dataset
from datasets.load import DatasetDict
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_OUTPUT_MODEL_ID = "pilipolio/chess-reasoning-sft"
DEFAULT_DATASET_ID = "pilipolio/chess-reasoning-traces"


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model_and_tokenizer(model_id: str, use_4bit: bool = False):
    device = get_device()
    print(f"Using device: {device}")
    print(f"Loading model: {model_id}")

    model_kwargs: dict[str, Any] = {"torch_dtype": torch.float16}

    if use_4bit and device == "cuda":
        try:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            print("Using 4-bit quantization")
        except ImportError:
            print("bitsandbytes not available, using full precision")
    elif use_4bit:
        print("4-bit quantization only available on CUDA, using full precision")

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def get_lora_config() -> LoraConfig:
    return LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )


def get_training_config(
    output_model_id: str,
    max_steps: int | None = None,
    eval_steps: int | None = None,
    wandb_project: str | None = None,
    wandb_run_name: str | None = None,
) -> SFTConfig:
    device = get_device()

    config_kwargs: dict[str, Any] = {
        "output_dir": output_model_id,
        "hub_model_id": output_model_id,
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 4,  # OOM with default 8 with Qwen3-4B 4bits
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-4,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "max_length": 2048,
        "packing": False,
        "logging_steps": 10,
        "save_steps": 500,
        "save_total_limit": 2,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "bf16": device == "cuda",
        "fp16": device == "mps",
        "optim": "adamw_torch",
        "eval_strategy": "steps",
        "eval_steps": eval_steps or 100,
        "push_to_hub": True,
    }

    if wandb_project:
        config_kwargs["report_to"] = "wandb"
        config_kwargs["run_name"] = wandb_run_name
        config_kwargs["logging_first_step"] = True
    else:
        config_kwargs["report_to"] = "none"

    if max_steps:
        config_kwargs["max_steps"] = max_steps
    else:
        config_kwargs["num_train_epochs"] = 3

    return SFTConfig(**config_kwargs)


def build_formatting_func(tokenizer: Any):
    """Build formatting function that applies chat template."""

    def format_reasoning_example(example: dict[str, Any]) -> str:
        """Format reasoning trace as chat string for SFT."""
        messages = [
            {"role": "user", "content": example["question"]},
            {"role": "assistant", "content": example["answer"]},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False)

    return format_reasoning_example


def train(
    model_id: str = DEFAULT_MODEL,
    output_model_id: str | None = None,
    dataset_id: str = DEFAULT_DATASET_ID,
    use_4bit: bool = False,
    max_steps: int | None = None,
    eval_steps: int | None = None,
    wandb_project: str | None = None,
    wandb_run_name: str | None = None,
) -> None:
    """Train a chess reasoning SFT model."""
    from chess_sandbox.puzzles_trainer.reasoning_callbacks import ReasoningValidationCallback

    model_short_name = model_id.split("/")[-1].lower()
    output_model_id = output_model_id or f"{DEFAULT_OUTPUT_MODEL_ID}-{model_short_name}"
    Path(output_model_id).mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(model_id, use_4bit=use_4bit)

    print(f"Loading dataset from Hub: {dataset_id}")
    dataset_dict: DatasetDict = cast(DatasetDict, load_dataset(dataset_id))
    train_dataset = dataset_dict["train"]
    test_dataset = dataset_dict["test"]

    callbacks: list[TrainerCallback] = []
    if wandb_project:
        wandb_run_name = wandb_run_name or model_short_name
        import os

        os.environ["WANDB_PROJECT"] = wandb_project
        callbacks.append(
            ReasoningValidationCallback(
                tokenizer=tokenizer,
                test_dataset=test_dataset,
            )
        )

    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    lora_config = get_lora_config()
    training_config = get_training_config(
        output_model_id,
        max_steps=max_steps,
        eval_steps=eval_steps,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
    )

    print("\nTraining config:")
    print(f"  Output model ID: {output_model_id}")
    print(f"  Batch size: {training_config.per_device_train_batch_size}")
    print(f"  Grad accum: {training_config.gradient_accumulation_steps}")
    print(f"  Learning rate: {training_config.learning_rate}")
    print(f"  Max length: {training_config.max_length}")
    print(f"  Packing: {training_config.packing}")
    print(f"  Max steps: {training_config.max_steps or 'full epochs'}")
    print(f"  W&B: {'enabled' if wandb_project else 'disabled'}")
    print(f"  Push to Hub: {training_config.push_to_hub}")

    formatting_func = build_formatting_func(tokenizer)

    trainer = SFTTrainer(
        model=model,
        args=training_config,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
        callbacks=callbacks,
        formatting_func=formatting_func,
    )

    print("\nStarting training...")
    trainer.train()

    print(f"\nSaving model to {output_model_id}")
    trainer.save_model(output_model_id)
    tokenizer.save_pretrained(output_model_id)

    print("Done!")


@click.command("reasoning-train")
@click.option(
    "--model-id",
    type=str,
    default=DEFAULT_MODEL,
    help=f"HuggingFace model ID (default: {DEFAULT_MODEL})",
)
@click.option(
    "--dataset-id",
    type=str,
    default=DEFAULT_DATASET_ID,
    help=f"HuggingFace dataset ID (default: {DEFAULT_DATASET_ID})",
)
@click.option("--use-4bit", is_flag=True, help="Use 4-bit quantization (CUDA only)")
@click.option("--max-steps", type=int, default=None, help="Max training steps (for testing)")
@click.option("--eval-steps", type=int, default=None, help="Eval every N steps (default: 100)")
@click.option("--output-model-id", type=str, default=None, help="Hub model ID (also saves locally)")
@click.option("--wandb-project", type=str, default=None, help="W&B project name (enables W&B logging)")
@click.option("--wandb-run-name", type=str, default=None, help="W&B run name (defaults to model name)")
def main(
    model_id: str,
    dataset_id: str,
    use_4bit: bool,
    max_steps: int | None,
    eval_steps: int | None,
    output_model_id: str | None,
    wandb_project: str | None,
    wandb_run_name: str | None,
) -> None:
    """Train a chess reasoning SFT model using LoRA fine-tuning."""
    train(
        model_id=model_id,
        output_model_id=output_model_id,
        dataset_id=dataset_id,
        use_4bit=use_4bit,
        max_steps=max_steps,
        eval_steps=eval_steps,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
    )


if __name__ == "__main__":
    main()
