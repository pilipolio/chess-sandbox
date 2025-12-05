"""Core training logic for chess puzzle SFT."""

from pathlib import Path

import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from chess_sandbox.puzzles_trainer.callbacks import ChessValidationCallback
from chess_sandbox.puzzles_trainer.dataset import load_puzzle_dataset

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"  # "Qwen/Qwen3-4B-Instruct-2507
DEFAULT_OUTPUT_MODEL_ID = "pilipolio/chess-puzzle-sft"


def get_device() -> str:
    """Detect available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model_and_tokenizer(model_id: str, use_4bit: bool = False):
    """Load model with optional 4-bit quantization."""
    device = get_device()
    print(f"Using device: {device}")
    print(f"Loading model: {model_id}")

    model_kwargs = {"torch_dtype": torch.float16}

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
    """Configure LoRA for parameter-efficient fine-tuning."""
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
    """Configure SFT training."""
    device = get_device()

    config_kwargs = {
        "output_dir": output_model_id,  # Also saves locally to this path
        "hub_model_id": output_model_id,
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 2,
        "learning_rate": 2e-4,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "max_length": 512,
        "packing": True,
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


def train(
    model_id: str = DEFAULT_MODEL,
    output_model_id: str | None = None,
    use_4bit: bool = False,
    max_steps: int | None = None,
    eval_steps: int | None = None,
    wandb_project: str | None = None,
    wandb_run_name: str | None = None,
) -> None:
    """Train a chess puzzle SFT model."""
    model_short_name = model_id.split("/")[-1].lower()
    output_model_id = output_model_id or f"{DEFAULT_OUTPUT_MODEL_ID}-{model_short_name}"
    Path(output_model_id).mkdir(parents=True, exist_ok=True)

    if wandb_project and not wandb_run_name:
        wandb_run_name = model_short_name

    # Let TRL/Trainer handle wandb initialization via report_to="wandb"
    # if wandb_project:
    #     import wandb
    #     wandb.init(project=wandb_project, name=wandb_run_name)
    if wandb_project:
        import os

        os.environ["WANDB_PROJECT"] = wandb_project

    model, tokenizer = load_model_and_tokenizer(model_id, use_4bit=use_4bit)
    train_dataset, test_dataset = load_puzzle_dataset()
    lora_config = get_lora_config()
    training_config = get_training_config(
        output_model_id,
        max_steps=max_steps,
        eval_steps=eval_steps,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
    )

    callbacks = []
    if wandb_project:
        callbacks.append(
            ChessValidationCallback(
                tokenizer=tokenizer,
                test_dataset=test_dataset,
            )
        )

    print("\nTraining config:")
    print(f"  Output model ID: {output_model_id}")
    print(f"  Batch size: {training_config.per_device_train_batch_size}")
    print(f"  Grad accum: {training_config.gradient_accumulation_steps}")
    print(f"  Learning rate: {training_config.learning_rate}")
    print(f"  Max steps: {training_config.max_steps or 'full epochs'}")
    print(f"  W&B: {'enabled' if wandb_project else 'disabled'}")
    print(f"  Push to Hub: {training_config.push_to_hub}")

    trainer = SFTTrainer(
        model=model,
        args=training_config,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    print("\nStarting training...")
    trainer.train()

    print(f"\nSaving model to {output_model_id}")
    trainer.save_model(output_model_id)
    tokenizer.save_pretrained(output_model_id)

    # if wandb_project:
    #     wandb.finish()

    print("Done!")
