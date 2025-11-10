import subprocess

from .config import settings


def get_commit_sha() -> str | None:
    """
    Get short git commit SHA from environment or git command.
    """
    if settings.GIT_COMMIT:
        return settings.GIT_COMMIT

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None
