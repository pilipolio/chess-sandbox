"""
Integration tests for concept probe training pipeline.

NOTE: These tests require a valid HF_TOKEN environment variable to download
models and datasets from HuggingFace Hub. Tests will download real data.
"""

from pathlib import Path

import pytest
from click.testing import CliRunner

from chess_sandbox.concept_extraction.model.train import train

# HuggingFace references for testing
TEST_MODEL_REPO = "lczerolens/maia-1500"
TEST_MODEL_FILENAME = "model.onnx"
TEST_DATASET_REPO = "pilipolio/chess-positions-concepts"
TEST_DATASET_FILENAME = "data.jsonl"
TEST_DATASET_REVISION = "test_fixture"


@pytest.mark.parametrize("mode", ["multi-class", "multi-label"])
def test_e2e_training_and_inference_pipeline(
    tmp_path: Path,
    mode: str,
) -> None:
    """
    End-to-end integration test of training and inference pipeline with HuggingFace Hub.
    """
    output_training_directory = tmp_path / f"probe_{mode}"

    runner = CliRunner()
    result = runner.invoke(
        train,
        [
            "--dataset-repo-id",
            TEST_DATASET_REPO,
            "--dataset-filename",
            TEST_DATASET_FILENAME,
            "--dataset-revision",
            TEST_DATASET_REVISION,
            "--lc0-model-repo-id",
            TEST_MODEL_REPO,
            "--lc0-model-filename",
            TEST_MODEL_FILENAME,
            "--layer-name",
            "block3/conv2/relu",
            "--output",
            str(output_training_directory),
            "--classifier-mode",
            mode,
            "--test-split",
            "0.3",
            "--random-seed",
            "42",
        ],
    )

    # Assert
    assert result.exit_code == 0, f"CLI failed with output:\n{result.output}"

    assert output_training_directory.exists(), f"Probe directory not created at {output_training_directory}"
    assert output_training_directory.is_dir(), "Probe output should be a directory"

    # Validate model card
    readme_path = output_training_directory / "README.md"
    assert readme_path.exists(), "README.md model card not created"

    readme_content = readme_path.read_text()
    assert "datasets:" in readme_content, "Model card missing datasets field in YAML"
    assert TEST_DATASET_REPO in readme_content, "Dataset repo not linked in model card"
    assert "base_model:" in readme_content, "Model card missing base_model field in YAML"
    assert "model-index:" in readme_content, "Model card missing model-index for Papers with Code"
    assert "exact_match" in readme_content.lower(), "Model card missing exact_match metric"
    assert "micro_precision" in readme_content.lower(), "Model card missing micro_precision metric"
    assert "micro_recall" in readme_content.lower(), "Model card missing micro_recall metric"
