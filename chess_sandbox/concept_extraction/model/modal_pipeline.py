"""Modal ephemeral app for chess concept probe training pipeline.

Usage: See train.py CLI for parameter documentation.

Setup: Requires HuggingFace token configured as Modal secret 'huggingface-read-write-secret'
"""

import subprocess

import modal

from chess_sandbox.git import get_commit_sha

image = modal.Image.debian_slim().uv_sync(uv_project_dir="./", frozen=True).add_local_python_source("chess_sandbox")

app = modal.App(name="chess-concept-training", image=image)


@app.function(  # type: ignore
    timeout=7200,  # 2 hours
    secrets=[modal.Secret.from_name("huggingface-read-write-secret")],  # type: ignore
    cpu=8.0,
    env={"GIT_COMMIT": get_commit_sha()},
)
def train(*arglist: str):
    """Train concept probe from labeled positions via Modal subprocess invocation.

    See train.py CLI for parameter documentation.

    Raises:
        RuntimeError: If training pipeline execution fails (non-zero exit code)
        ValueError: If upload_to_hub=True but output_repo_id is not provided
    """

    cmd = ["uv", "run", "python", "-m", "chess_sandbox.concept_extraction.model.train", *arglist]
    subprocess.run(cmd, check=True)
