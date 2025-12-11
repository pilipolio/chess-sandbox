"""GRPO training for chess reasoning models.

Fine-tunes models using Group Relative Policy Optimization with verifiable
chess rewards (legality, correctness, format, piece accuracy).
"""

from pathlib import Path
from typing import Any

import click
import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from chess_sandbox.puzzles_trainer.grpo_rewards import chess_reasoning_reward

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_OUTPUT_MODEL_ID = "pilipolio/chess-reasoning-grpo"
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
    tokenizer.padding_side = "left"

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


def get_grpo_config(
    output_model_id: str,
    num_generations: int = 8,
    max_completion_length: int = 512,
    max_steps: int | None = None,
    use_vllm: bool = False,
    vllm_gpu_memory_utilization: float = 0.4,
    wandb_project: str | None = None,
    wandb_run_name: str | None = None,
) -> GRPOConfig:
    """Build GRPOConfig with chess-specific defaults.

    Key parameters:
    - num_generations: 8 completions per prompt (balance quality vs compute)
    - beta: 0.0 (KL coefficient disabled per recent GRPO best practices)
    - max_completion_length: 512 (enough for reasoning + solution)
    """
    device = get_device()

    config_kwargs: dict[str, Any] = {
        "output_dir": output_model_id,
        "hub_model_id": output_model_id,
        # GRPO-specific
        "num_generations": num_generations,
        "max_completion_length": max_completion_length,
        "beta": 0.0,
        "scale_rewards": True,
        # Training hyperparameters
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 1e-6,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "logging_steps": 10,
        "save_steps": 100,
        "save_total_limit": 3,
        # Memory optimization
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "bf16": device == "cuda",
        "fp16": device == "mps",
        # Hub - only push if we have a proper model ID
        "push_to_hub": wandb_project is not None,
    }

    # vLLM acceleration
    if use_vllm:
        config_kwargs["use_vllm"] = True
        config_kwargs["vllm_mode"] = "colocate"
        config_kwargs["vllm_gpu_memory_utilization"] = vllm_gpu_memory_utilization

    # W&B logging
    if wandb_project:
        config_kwargs["report_to"] = "wandb"
        config_kwargs["run_name"] = wandb_run_name
        config_kwargs["logging_first_step"] = True
    else:
        config_kwargs["report_to"] = "none"

    # Training duration
    if max_steps:
        config_kwargs["max_steps"] = max_steps
    else:
        config_kwargs["num_train_epochs"] = 1

    return GRPOConfig(**config_kwargs)


def prepare_dataset(dataset_id: str, split: str = "train") -> Dataset:
    """Load and prepare dataset for GRPO.

    GRPOTrainer requires 'prompt' column. Additional columns
    (fen, first_move, solution) are passed to reward function.
    """
    print(f"Loading dataset: {dataset_id} (split: {split})")
    dataset: Dataset = load_dataset(dataset_id, split=split)  # pyright: ignore[reportAssignmentType]

    # Rename 'question' to 'prompt' for TRL compatibility
    if "question" in dataset.column_names:
        dataset = dataset.rename_column("question", "prompt")

    # Keep columns needed for reward function
    keep_columns = ["prompt", "fen", "first_move", "solution", "themes"]
    available_columns = [c for c in keep_columns if c in dataset.column_names]
    dataset = dataset.select_columns(available_columns)

    print(f"Dataset columns: {dataset.column_names}")
    print(f"Dataset size: {len(dataset)}")

    return dataset


def train(
    model_id: str = DEFAULT_MODEL,
    output_model_id: str | None = None,
    dataset_id: str = DEFAULT_DATASET_ID,
    use_4bit: bool = False,
    use_vllm: bool = False,
    vllm_gpu_memory_utilization: float = 0.4,
    num_generations: int = 8,
    max_completion_length: int = 512,
    max_steps: int | None = None,
    wandb_project: str | None = None,
    wandb_run_name: str | None = None,
) -> None:
    """Train chess reasoning model with GRPO."""
    model_short_name = model_id.split("/")[-1].lower()
    output_model_id = output_model_id or f"{DEFAULT_OUTPUT_MODEL_ID}-{model_short_name}"
    Path(output_model_id).mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(model_id, use_4bit=use_4bit)

    train_dataset = prepare_dataset(dataset_id, split="train")

    lora_config = get_lora_config()
    grpo_config = get_grpo_config(
        output_model_id=output_model_id,
        num_generations=num_generations,
        max_completion_length=max_completion_length,
        max_steps=max_steps,
        use_vllm=use_vllm,
        vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
    )

    # W&B setup
    if wandb_project:
        import os

        os.environ["WANDB_PROJECT"] = wandb_project

    print("\nGRPO Training Config:")
    print(f"  Output: {output_model_id}")
    print(f"  Generations per prompt: {num_generations}")
    print(f"  Max completion length: {max_completion_length}")
    print(f"  Use vLLM: {use_vllm}")
    if use_vllm:
        print(f"  vLLM GPU memory: {vllm_gpu_memory_utilization}")
    print(f"  Learning rate: {grpo_config.learning_rate}")
    print(f"  KL coef (beta): {grpo_config.beta}")
    print(f"  Max steps: {grpo_config.max_steps or 'full epochs'}")
    print(f"  W&B: {'enabled' if wandb_project else 'disabled'}")

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        reward_funcs=chess_reasoning_reward,
        train_dataset=train_dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    print("\nStarting GRPO training...")
    trainer.train()

    print(f"\nSaving model to {output_model_id}")
    trainer.save_model(output_model_id)
    tokenizer.save_pretrained(output_model_id)

    if grpo_config.push_to_hub:
        print(f"Pushing to Hub: {output_model_id}")
        trainer.push_to_hub()

    print("Done!")


@click.command("grpo-train")
@click.option(
    "--model-id",
    type=str,
    default=DEFAULT_MODEL,
    help=f"Model to fine-tune (default: {DEFAULT_MODEL})",
)
@click.option(
    "--dataset-id",
    type=str,
    default=DEFAULT_DATASET_ID,
    help=f"HuggingFace dataset ID (default: {DEFAULT_DATASET_ID})",
)
@click.option("--output-model-id", type=str, default=None, help="Hub model ID for output")
@click.option("--use-4bit", is_flag=True, help="Use 4-bit quantization (CUDA only)")
@click.option("--use-vllm", is_flag=True, help="Use vLLM for accelerated generation")
@click.option("--vllm-gpu-memory", type=float, default=0.4, help="vLLM GPU memory utilization (default: 0.4)")
@click.option("--num-generations", type=int, default=8, help="Completions per prompt (default: 8)")
@click.option("--max-completion-length", type=int, default=512, help="Max tokens to generate (default: 512)")
@click.option("--max-steps", type=int, default=None, help="Max training steps (for testing)")
@click.option("--wandb-project", type=str, default=None, help="W&B project name")
@click.option("--wandb-run-name", type=str, default=None, help="W&B run name")
def main(
    model_id: str,
    dataset_id: str,
    output_model_id: str | None,
    use_4bit: bool,
    use_vllm: bool,
    vllm_gpu_memory: float,
    num_generations: int,
    max_completion_length: int,
    max_steps: int | None,
    wandb_project: str | None,
    wandb_run_name: str | None,
) -> None:
    """Train a chess reasoning model with GRPO (Group Relative Policy Optimization)."""
    train(
        model_id=model_id,
        output_model_id=output_model_id,
        dataset_id=dataset_id,
        use_4bit=use_4bit,
        use_vllm=use_vllm,
        vllm_gpu_memory_utilization=vllm_gpu_memory,
        num_generations=num_generations,
        max_completion_length=max_completion_length,
        max_steps=max_steps,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
    )


if __name__ == "__main__":
    main()
