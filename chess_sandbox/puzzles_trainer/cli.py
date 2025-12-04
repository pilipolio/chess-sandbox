"""CLI for chess puzzle SFT training."""

from pathlib import Path

import click

from chess_sandbox.puzzles_trainer.trainer import DEFAULT_MODEL, MODELS, train


@click.command()
@click.option(
    "--model",
    type=click.Choice(list(MODELS.keys())),
    default=DEFAULT_MODEL,
    help=f"Model to fine-tune (default: {DEFAULT_MODEL})",
)
@click.option("--use-4bit", is_flag=True, help="Use 4-bit quantization (CUDA only)")
@click.option("--max-steps", type=int, default=None, help="Max training steps (for testing)")
@click.option("--output-dir", type=click.Path(path_type=Path), default=None, help="Output directory")
@click.option("--wandb-project", type=str, default=None, help="W&B project name (enables W&B logging)")
@click.option("--wandb-run-name", type=str, default=None, help="W&B run name")
def main(
    model: str,
    use_4bit: bool,
    max_steps: int | None,
    output_dir: Path | None,
    wandb_project: str | None,
    wandb_run_name: str | None,
) -> None:
    """Train a chess puzzle SFT model using LoRA fine-tuning."""
    train(
        model_name=model,
        output_dir=output_dir,
        use_4bit=use_4bit,
        max_steps=max_steps,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
    )


if __name__ == "__main__":
    main()
