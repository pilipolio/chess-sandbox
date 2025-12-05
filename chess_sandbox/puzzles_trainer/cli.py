"""CLI for chess puzzle SFT training."""

import click

from chess_sandbox.puzzles_trainer.trainer import DEFAULT_MODEL, train


@click.command()
@click.option(
    "--model-id",
    type=str,
    default=DEFAULT_MODEL,
    help=f"HuggingFace model ID (default: {DEFAULT_MODEL})",
)
@click.option("--use-4bit", is_flag=True, help="Use 4-bit quantization (CUDA only)")
@click.option("--max-steps", type=int, default=None, help="Max training steps (for testing)")
@click.option("--eval-steps", type=int, default=None, help="Eval every N steps (default: 100)")
@click.option("--output-model-id", type=str, default=None, help="Hub model ID (also saves locally)")
@click.option("--wandb-project", type=str, default=None, help="W&B project name (enables W&B logging)")
@click.option("--wandb-run-name", type=str, default=None, help="W&B run name (defaults to model name)")
def main(
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


if __name__ == "__main__":
    main()
