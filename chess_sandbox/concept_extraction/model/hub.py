"""
HuggingFace Hub integration for concept probes.

Simplified upload since probes are already saved in HF snapshot format.
"""

from pathlib import Path

from huggingface_hub import HfApi

from chess_sandbox.config import settings


def upload_probe(
    probe_dir: Path | str,
    *,
    model_name: str,
    token: str | None = None,
    commit_message: str | None = None,
) -> str:
    """
    Upload probe directory to HuggingFace Hub.

    Since probe is already saved in HF snapshot format (directory with
    joblib files + metadata), this just uploads the directory contents.

    Args:
        probe_dir: Path to probe directory (e.g., "data/models/concept_probes/probe_v1")
        model_name: Model repository name (e.g., "chess-sandbox-concept-probes")
        token: HF token (defaults to settings.HF_TOKEN)
        commit_message: Custom commit message

    Returns:
        Commit URL from HuggingFace Hub
    """
    probe_dir = Path(probe_dir)

    if not probe_dir.is_dir():
        msg = f"Probe directory does not exist: {probe_dir}"
        raise ValueError(msg)

    token = token or settings.HF_TOKEN or None
    repo_id = f"{settings.HF_ORG}/{model_name}"

    api = HfApi(token=token)
    return api.upload_folder(
        folder_path=probe_dir,
        repo_id=repo_id,
        commit_message=commit_message,
        repo_type="model",
    )
