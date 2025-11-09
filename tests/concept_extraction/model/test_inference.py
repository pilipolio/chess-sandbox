"""
Integration tests for concept probe inference pipeline.

NOTE: These tests require a valid HF_TOKEN environment variable to download
models and probes from HuggingFace Hub.
"""

from pathlib import Path

from click.testing import CliRunner

from chess_sandbox.concept_extraction.model.inference import predict

# HuggingFace references for testing
TEST_LC0_MODEL_REPO = "lczerolens/maia-1500"
TEST_LC0_MODEL_FILENAME = "model.onnx"
TEST_MODEL_REPO = "pilipolio/chess-positions-extractor"
TEST_MODEL_REVISION = "0e35944267e24ddf318296ac358cfd2215087486"

TEST_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def test_predict_single_fen(tmp_path: Path) -> None:
    """
    Test the predict command with a single FEN position using HuggingFace Hub.
    Downloads a pre-trained probe from HF Hub and uses it to predict concepts.
    Requires HF_TOKEN environment variable.
    """
    # Invoke predict CLI command with HF Hub options
    runner = CliRunner()
    result = runner.invoke(
        predict,
        [
            TEST_FEN,
            "--model-repo-id",
            TEST_MODEL_REPO,
            "--revision",
            TEST_MODEL_REVISION,
            "--lc0-repo-id",
            TEST_LC0_MODEL_REPO,
            "--lc0-filename",
            TEST_LC0_MODEL_FILENAME,
            "--cache-dir",
            str(tmp_path / "cache"),
        ],
    )

    # Assert success
    assert result.exit_code == 0, f"CLI failed with output:\n{result.output}"

    # Assert output contains predictions
    assert (
        "Predicted concepts" in result.output or "concepts" in result.output.lower()
    ), f"Expected predictions in output, got:\n{result.output}"

    # Assert output contains confidence scores (predictions should have confidence >= 0.1)
    assert any(char.isdigit() for char in result.output), f"Expected confidence scores in output, got:\n{result.output}"
