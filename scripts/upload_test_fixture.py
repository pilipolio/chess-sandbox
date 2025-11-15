#!/usr/bin/env python3
"""
Upload test fixture to HuggingFace Hub.

Creates a test_fixture branch with a small dataset for fast training tests.

Requirements:
    - HF_TOKEN environment variable with WRITE access
    - Or use: huggingface-cli login

To create a write token:
    1. Go to https://huggingface.co/settings/tokens
    2. Create a new token with 'write' access
    3. Set it: export HF_TOKEN=hf_xxx
"""

import os
import sys
from pathlib import Path

from chess_sandbox.logging_config import setup_logging

logger = setup_logging(__name__)

from huggingface_hub import HfApi

# Configuration
REPO_ID = "pilipolio/chess-positions-concepts"
LOCAL_FILE = Path("data/processed/test_labeled_positions.jsonl")
HF_FILENAME = "data.jsonl"
REVISION = "test_fixture"
COMMIT_MESSAGE = "Add test fixture with 29 validated positions for fast training"

if __name__ == "__main__":
    if not LOCAL_FILE.exists():
        msg = f"Local file not found: {LOCAL_FILE}"
        raise FileNotFoundError(msg)

    # Check for HF token
    if not os.getenv("HF_TOKEN"):
        logger.error("ERROR: HF_TOKEN environment variable not set")
        logger.error("\nPlease set a HuggingFace token with WRITE access:")
        logger.error("  export HF_TOKEN=hf_xxx")
        logger.error("\nOr login with:")
        logger.error("  huggingface-cli login")
        sys.exit(1)

    logger.info(f"Uploading {LOCAL_FILE} to {REPO_ID}/{HF_FILENAME}")
    logger.info(f"Revision: {REVISION}")
    logger.info(f"File size: {LOCAL_FILE.stat().st_size:,} bytes")

    with open(LOCAL_FILE) as f:
        num_lines = sum(1 for _ in f)
    logger.info(f"Number of positions: {num_lines}")

    api = HfApi()

    try:
        # Upload file to specific revision
        # This will create the revision if it doesn't exist
        result = api.upload_file(
            path_or_fileobj=str(LOCAL_FILE),
            path_in_repo=HF_FILENAME,
            repo_id=REPO_ID,
            repo_type="dataset",
            revision=REVISION,
            commit_message=COMMIT_MESSAGE,
        )

        logger.info(f"\n✓ Success! Uploaded to: {result}")
        logger.info("\nYou can now use:")
        logger.info(f"  --dataset-repo-id {REPO_ID}")
        logger.info(f"  --dataset-filename {HF_FILENAME}")
        logger.info(f"  --dataset-revision {REVISION}")

    except Exception as e:
        logger.error(f"\n✗ Upload failed: {e}")
        logger.error("\nMake sure your HF_TOKEN has WRITE permissions")
        sys.exit(1)
