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
        print("ERROR: HF_TOKEN environment variable not set", file=sys.stderr)
        print("\nPlease set a HuggingFace token with WRITE access:", file=sys.stderr)
        print("  export HF_TOKEN=hf_xxx", file=sys.stderr)
        print("\nOr login with:", file=sys.stderr)
        print("  huggingface-cli login", file=sys.stderr)
        sys.exit(1)

    print(f"Uploading {LOCAL_FILE} to {REPO_ID}/{HF_FILENAME}")
    print(f"Revision: {REVISION}")
    print(f"File size: {LOCAL_FILE.stat().st_size:,} bytes")

    with open(LOCAL_FILE) as f:
        num_lines = sum(1 for _ in f)
    print(f"Number of positions: {num_lines}")

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

        print(f"\n✓ Success! Uploaded to: {result}")
        print("\nYou can now use:")
        print(f"  --dataset-repo-id {REPO_ID}")
        print(f"  --dataset-filename {HF_FILENAME}")
        print(f"  --dataset-revision {REVISION}")

    except Exception as e:
        print(f"\n✗ Upload failed: {e}", file=sys.stderr)
        print("\nMake sure your HF_TOKEN has WRITE permissions", file=sys.stderr)
        sys.exit(1)
