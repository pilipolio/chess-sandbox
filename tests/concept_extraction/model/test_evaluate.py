"""
Integration tests for concept probe evaluation CLI.

NOTE: These tests require a valid HF_TOKEN environment variable to download
models and datasets from HuggingFace Hub. Tests will download real data.
"""

from click.testing import CliRunner

from chess_sandbox.concept_extraction.model.evaluation import cli

TEST_MODEL_REPO = "pilipolio/chess-positions-extractor"
TEST_MODEL_REVISION = "0e35944267e24ddf318296ac358cfd2215087486"
TEST_DATASET_REPO = "pilipolio/chess-positions-concepts"
TEST_DATASET_FILENAME = "data.jsonl"
TEST_DATASET_REVISION = "test_fixture"
TEST_LC0_MODEL_REPO = "lczerolens/maia-1500"
TEST_LC0_MODEL_FILENAME = "model.onnx"


def test_evaluate_cli_with_sample_predictions() -> None:
    """
    End-to-end integration test of evaluation CLI with HuggingFace Hub.

    Tests the full evaluation pipeline:
    - Downloads trained probe from HF Hub
    - Downloads test dataset from HF Hub
    - Displays sample predictions
    - Calculates and displays comprehensive metrics
    """
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "evaluate",
            "--lc0-repo-id",
            TEST_LC0_MODEL_REPO,
            "--lc0-filename",
            TEST_LC0_MODEL_FILENAME,
            "--revision",
            TEST_MODEL_REVISION,
            "--model-repo-id",
            TEST_MODEL_REPO,
            "--dataset-repo-id",
            TEST_DATASET_REPO,
            "--dataset-filename",
            TEST_DATASET_FILENAME,
            "--dataset-revision",
            TEST_DATASET_REVISION,
            "--sample-size",
            "10",
            "--random-seed",
            "42",
        ],
    )

    # Assert command succeeded
    assert result.exit_code == 0, f"CLI failed with output:\n{result.output}"

    # Validate output contains expected sections
    output = result.output

    # Check for data loading messages
    assert "Loading ConceptExtractor from HuggingFace Hub" in output, "Missing model loading message"
    assert "Loading evaluation dataset from HuggingFace Hub" in output, "Missing dataset loading message"
    assert "Loaded" in output and "positions" in output, "Missing position count message"

    # Check for sample predictions section
    assert "SAMPLE PREDICTIONS" in output, "Missing sample predictions section"
    assert "FEN:" in output, "Missing FEN positions in output"
    assert "Ground Truth:" in output, "Missing ground truth labels"
    assert "Prediction:" in output, "Missing predictions"

    # Check for comprehensive metrics section
    assert "SUMMARY" in output, "Missing evaluation metrics summary"
    assert "PER-CONCEPT METRICS" in output, "Missing per-concept metrics section"
