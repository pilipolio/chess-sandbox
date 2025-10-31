"""Modal ephemeral app for chess concept labeling pipeline.

Usage:
    modal run chess_sandbox/concept_labelling/modal_pipeline.py::main \\
        --input-subdir pgn_inputs/gameknot \\
        --output-filename output.jsonl \\
        --limit 10 \\
        --refine-with-llm

Setup and detailed instructions: docs/plans/modal-pipeline-execution.md
"""

import modal

image = modal.Image.debian_slim().uv_sync(uv_project_dir="./", frozen=True).add_local_python_source("chess_sandbox")

app = modal.App(name="chess-concept-pipeline", image=image)
volume = modal.Volume.from_name("chess-pgn-data", create_if_missing=True)  # type: ignore


@app.function(  # type: ignore
    timeout=7200,  # 2 hours for large batches
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("openai-secret")],  # type: ignore
    gpu=None,
)
def process_pgn_batch(
    input_subdir: str = "pgn_inputs/gameknot",
    output_filename: str = "labeled_positions.jsonl",
    limit: int | None = None,
    refine_with_llm: bool = False,
    llm_model: str = "gpt-4.1-mini",
) -> dict[str, str | int]:
    """Process PGN files from volume using CLI invocation via subprocess.

    This function invokes the Click CLI pipeline as a subprocess, allowing
    real-time log streaming to Modal's output.

    Args:
        input_subdir: Subdirectory in /data containing PGN files (e.g., "pgn_inputs/gameknot")
        output_filename: Output JSONL filename in /data/outputs/ (e.g., "labeled.jsonl")
        limit: Optional limit on number of PGN files to process (for testing)
        refine_with_llm: Whether to use LLM for concept validation (requires OPENAI_API_KEY)
        llm_model: LLM model to use (default: gpt-5-nano)

    Returns:
        Dictionary with processing status and exit code

    Raises:
        RuntimeError: If pipeline execution fails (non-zero exit code)
    """
    import subprocess

    input_dir = f"/data/{input_subdir}"
    output = f"/data/outputs/{output_filename}"

    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "chess_sandbox.concept_labelling.pipeline",
        "--input-dir",
        input_dir,
        "--output",
        output,
    ]
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    if refine_with_llm:
        cmd.append("--refine-with-llm")
    cmd.extend(["--llm-model", llm_model])

    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        raise RuntimeError(f"Pipeline failed with exit code {result.returncode}")

    volume.commit()

    return {
        "status": "success",
        "exit_code": result.returncode,
    }
