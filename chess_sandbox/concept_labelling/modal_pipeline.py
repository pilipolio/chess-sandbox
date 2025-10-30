"""Modal ephemeral app for chess concept labeling pipeline.

This module wraps the existing Click CLI pipeline for execution on Modal,
enabling serverless batch processing of PGN files with LLM-based concept validation.

Implementation approach (Option 2 - Tactical):
- Uses Click's CliRunner to invoke existing pipeline.main()
- Zero changes to existing pipeline code
- Fast validation of Modal integration
- Can migrate to extracted core logic (Option 1) later

Usage:
    modal run chess_sandbox/concept_labelling/modal_pipeline.py::main \\
        --input-subdir pgn_inputs/gameknot \\
        --output-filename output.jsonl \\
        --limit 10 \\
        --refine-with-llm
"""

import modal

# Build Modal image with all dependencies
image = (
    modal.Image.debian_slim()
    .uv_sync(uv_project_dir="./", frozen=True)  # Install from pyproject.toml
    .add_local_python_source("chess_sandbox")  # Include our module
)

# Create Modal app
app = modal.App(name="chess-concept-pipeline", image=image)

# Persistent volume for PGN inputs and JSONL outputs
volume = modal.Volume.from_name("chess-pgn-data", create_if_missing=True)  # type: ignore


@app.function(  # type: ignore
    timeout=3600,  # 1 hour for large batches
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("openai-secret")],  # type: ignore
)
def process_pgn_batch(
    input_subdir: str,
    output_filename: str,
    limit: int | None = None,
    refine_with_llm: bool = False,
    llm_model: str = "gpt-5-nano",
) -> dict[str, str | int]:
    """Process PGN files from volume using Click CLI wrapper.

    This function wraps the existing Click CLI pipeline without duplicating logic.
    It uses CliRunner to invoke the CLI with arguments mapped from Modal parameters.

    Args:
        input_subdir: Subdirectory in /data containing PGN files (e.g., "pgn_inputs/gameknot")
        output_filename: Output JSONL filename in /data/outputs/ (e.g., "labeled.jsonl")
        limit: Optional limit on number of PGN files to process (for testing)
        refine_with_llm: Whether to use LLM for concept validation (requires OPENAI_API_KEY)
        llm_model: LLM model to use (default: gpt-5-nano)

    Returns:
        Dictionary with processing status and CLI output text

    Raises:
        RuntimeError: If pipeline execution fails (non-zero exit code)
    """
    from click.testing import CliRunner

    from chess_sandbox.concept_labelling.pipeline import main

    # Build CLI arguments for Click command
    input_dir = f"/data/{input_subdir}"
    output = f"/data/outputs/{output_filename}"

    args = [
        "--input-dir",
        input_dir,
        "--output",
        output,
    ]

    if limit is not None:
        args.extend(["--limit", str(limit)])

    if refine_with_llm:
        args.append("--refine-with-llm")

    args.extend(["--llm-model", llm_model])

    # Invoke Click CLI (runs in-process, no subprocess overhead)
    runner = CliRunner()
    result = runner.invoke(main, args)

    # Forward CLI output to Modal logs for visibility
    print(result.output)

    # Check for errors
    if result.exit_code != 0:
        error_msg = f"Pipeline failed with exit code {result.exit_code}"
        if result.exception:
            error_msg += f"\nException: {result.exception}"
        raise RuntimeError(error_msg)

    # Commit volume changes to persist output
    volume.commit()

    return {
        "status": "success",
        "exit_code": result.exit_code,
        "output": result.output,
    }


@app.local_entrypoint()
def main(
    input_subdir: str = "pgn_inputs/gameknot",
    output_filename: str = "labeled_positions.jsonl",
    limit: int | None = None,
    refine_with_llm: bool = False,
    llm_model: str = "gpt-5-nano",
):
    """Local entrypoint for running Modal pipeline.

    This function runs on your local machine and orchestrates remote Modal execution.

    Args:
        input_subdir: Subdirectory in volume containing PGN files
        output_filename: Output JSONL filename in volume
        limit: Optional limit on PGN files to process
        refine_with_llm: Whether to use LLM refinement
        llm_model: LLM model name

    Example:
        # Process 10 files without LLM
        modal run modal_pipeline.py::main --limit 10

        # Process all files with LLM refinement
        modal run modal_pipeline.py::main --refine-with-llm
    """
    print("Starting Modal pipeline execution...")
    print(f"  Input: /data/{input_subdir}")
    print(f"  Output: /data/outputs/{output_filename}")
    if limit:
        print(f"  Limit: {limit} files")
    print(f"  LLM refinement: {'enabled' if refine_with_llm else 'disabled'}")
    if refine_with_llm:
        print(f"  LLM model: {llm_model}")
    print()

    result = process_pgn_batch.remote(
        input_subdir=input_subdir,
        output_filename=output_filename,
        limit=limit,
        refine_with_llm=refine_with_llm,
        llm_model=llm_model,
    )

    print(f"\n{'='*60}")
    print("Processing complete!")
    print(f"Status: {result['status']}")
    print(f"Output file: /data/outputs/{output_filename}")
    print(f"{'='*60}")
    print("\nDownload results with:")
    print(f"  modal volume get chess-pgn-data /outputs/{output_filename} ./{output_filename}")
