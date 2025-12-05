"""Core training logic for chess puzzle SFT."""

from pathlib import Path

import click
import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
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
        "output_dir": output_model_id,
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
    from chess_sandbox.puzzles_trainer.callbacks import ChessValidationCallback
    from chess_sandbox.puzzles_trainer.dataset import load_puzzle_dataset

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


@click.group()
def main() -> None:
    """Chess puzzle trainer CLI."""


@main.command("train")
@click.option(
    "--model-id",
    type=str,
    default="Qwen/Qwen3-0.6B",
    help="HuggingFace model ID (default: Qwen/Qwen3-0.6B)",
)
@click.option("--use-4bit", is_flag=True, help="Use 4-bit quantization (CUDA only)")
@click.option("--max-steps", type=int, default=None, help="Max training steps (for testing)")
@click.option("--eval-steps", type=int, default=None, help="Eval every N steps (default: 100)")
@click.option("--output-model-id", type=str, default=None, help="Hub model ID (also saves locally)")
@click.option("--wandb-project", type=str, default=None, help="W&B project name (enables W&B logging)")
@click.option("--wandb-run-name", type=str, default=None, help="W&B run name (defaults to model name)")
def train_command(
    model_id: str,
    use_4bit: bool,
    max_steps: int | None,
    eval_steps: int | None,
    output_model_id: str | None,
    wandb_project: str | None,
    wandb_run_name: str | None,
) -> None:
    """Train a chess puzzle SFT model using LoRA fine-tuning."""
    train(
        model_id=model_id,
        output_model_id=output_model_id,
        use_4bit=use_4bit,
        max_steps=max_steps,
        eval_steps=eval_steps,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
    )


DEFAULT_DATASET_IDS = {
    "puzzle": "pilipolio/chess-puzzle-tasks",
    "toy": "pilipolio/chess-toy-tasks",
    "mixed": "pilipolio/chess-mixed-tasks",
}


@main.command("generate-samples")
@click.option("--sample-size", type=int, default=1000, help="Number of puzzles/exercises to generate")
@click.option("--test-split", type=float, default=0.1, help="Fraction for test set")
@click.option("--seed", type=int, default=42, help="Random seed")
@click.option("--image-size", type=int, default=240, help="Board image size in pixels")
@click.option("--min-popularity", type=int, default=80, help="Minimum puzzle popularity (puzzle/mixed)")
@click.option("--max-rating", type=int, default=None, help="Maximum puzzle rating (puzzle/mixed)")
@click.option("--themes", type=str, default=None, help="Comma-separated theme filter (puzzle/mixed)")
@click.option(
    "--source",
    type=click.Choice(["puzzle", "toy", "mixed"]),
    default="puzzle",
    help="Data source: puzzle (Lichess), toy (synthetic), or mixed",
)
@click.option("--toy-ratio", type=float, default=0.3, help="Fraction of toy exercises when source=mixed")
@click.option(
    "--include-toy-representation/--no-toy-representation",
    default=True,
    help="Include FEN/piece-list conversion tasks for toy exercises",
)
@click.option("--push-to-hub", is_flag=True, help="Push dataset to HuggingFace Hub")
@click.option(
    "--dataset-id",
    type=str,
    default=None,
    help="HuggingFace dataset ID (default: source-specific ID)",
)
def generate_samples(
    sample_size: int,
    test_split: float,
    seed: int,
    image_size: int,
    min_popularity: int,
    max_rating: int | None,
    themes: str | None,
    source: str,
    toy_ratio: float,
    include_toy_representation: bool,
    push_to_hub: bool,
    dataset_id: str | None,
) -> None:
    """Generate chess task dataset with board images.

    Sources:
      - puzzle: Lichess puzzles with multiple task types (SAN moves)
      - toy: Synthetic toy exercises (capture sequences, movement paths, UCI moves)
      - mixed: Combination of both (use --toy-ratio to control mix)
    """
    from chess_sandbox.puzzles_trainer.dataset import materialize_task_dataset

    themes_tuple = tuple(themes.split(",")) if themes else None

    dataset_dict = materialize_task_dataset(
        sample_size=sample_size,
        test_split=test_split,
        seed=seed,
        image_size=image_size,
        min_popularity=min_popularity,
        max_rating=max_rating,
        themes=themes_tuple,
        source=source,  # type: ignore[arg-type]
        toy_ratio=toy_ratio,
        include_toy_representation=include_toy_representation,
    )

    if push_to_hub:
        hub_dataset_id = dataset_id or DEFAULT_DATASET_IDS[source]
        dataset_dict.push_to_hub(hub_dataset_id)
        print(f"Pushed dataset to: https://huggingface.co/datasets/{hub_dataset_id}")


@main.command("generate-curriculum")
@click.option("--capture-exercises", type=int, default=100, help="Number of capture exercises")
@click.option("--movement-exercises", type=int, default=100, help="Number of movement exercises")
@click.option("--include-representation/--no-representation", default=True, help="Include representation tasks")
@click.option("--seed", type=int, default=42, help="Random seed")
@click.option("--output", type=click.Path(), required=True, help="Output JSONL file path")
@click.option("--push-to-hub", is_flag=True, help="Push dataset to HuggingFace Hub")
@click.option("--dataset-id", type=str, default=None, help="HuggingFace dataset ID for push")
def generate_curriculum_command(
    capture_exercises: int,
    movement_exercises: int,
    include_representation: bool,
    seed: int,
    output: str,
    push_to_hub: bool,
    dataset_id: str | None,
) -> None:
    """Generate synthetic toy chess curriculum dataset."""
    import json

    from datasets import Dataset

    from chess_sandbox.puzzles_trainer.toy_curriculum import create_toy_curriculum

    print("Generating toy curriculum...")
    print(f"  Capture exercises: {capture_exercises}")
    print(f"  Movement exercises: {movement_exercises}")
    print(f"  Include representation tasks: {include_representation}")
    print(f"  Seed: {seed}")

    curriculum = create_toy_curriculum(
        capture_exercises=capture_exercises,
        movement_exercises=movement_exercises,
        include_representation=include_representation,
        seed=seed,
    )

    print(f"\nGenerated {len(curriculum)} total tasks")

    task_counts: dict[str, int] = {}
    for ex in curriculum:
        task_type = ex.get("task_type", "unknown")
        task_counts[task_type] = task_counts.get(task_type, 0) + 1
    print(f"Task distribution: {task_counts}")

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for item in curriculum:
            f.write(json.dumps(item) + "\n")

    print(f"\nSaved to: {output_path}")

    if push_to_hub:
        if not dataset_id:
            raise click.UsageError("--dataset-id required when using --push-to-hub")
        dataset = Dataset.from_list(curriculum)
        dataset.push_to_hub(dataset_id)
        print(f"Pushed dataset to: https://huggingface.co/datasets/{dataset_id}")


if __name__ == "__main__":
    main()
