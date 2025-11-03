"""
Integration tests for concept probe training pipeline.
"""

from pathlib import Path

import pytest
from click.testing import CliRunner

from chess_sandbox.concept_extraction.model.inference import ConceptProbe
from chess_sandbox.concept_extraction.model.train import train

# Path to fixtures
FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_DATA = FIXTURES_DIR / "sample_labeled_positions.jsonl"
MODEL_PATH = FIXTURES_DIR / "maia-1500.onnx"


@pytest.fixture
def lc0_model_path() -> Path:
    """Fixture that provides path to LC0 model or skips test."""
    if not MODEL_PATH.exists():
        pytest.skip(f"LC0 model not found at {MODEL_PATH}. Please copy maia-1500.onnx to tests/fixtures/")
    return MODEL_PATH


@pytest.mark.integration
@pytest.mark.parametrize("mode", ["multi-class", "multi-label"])
def test_full_training_pipeline_with_real_model(
    tmp_path: Path,
    lc0_model_path: Path,
    mode: str,
) -> None:
    """
    End-to-end integration test of training pipeline with real LC0 model.

    Tests both multi-class and multi-label modes with actual feature extraction.
    """
    output_probe = tmp_path / f"probe_{mode}.pkl"

    runner = CliRunner()
    result = runner.invoke(
        train,
        [
            "--data-path",
            str(SAMPLE_DATA),
            "--model-path",
            str(lc0_model_path),
            "--layer-name",
            "block3/conv2/relu",
            "--output",
            str(output_probe),
            "--mode",
            mode,
            "--test-split",
            "0.3",
            "--random-seed",
            "42",
            "--model-version",
            f"test_{mode}_v1",
            "--batch-size",
            "4",
        ],
    )

    # Check CLI succeeded
    assert result.exit_code == 0, f"CLI failed with output:\n{result.output}"

    # Check expected output patterns in CLI output
    assert "Training concept probe..." in result.output
    assert f"Mode: {mode}" in result.output
    assert "Loading LC0 model..." in result.output
    assert "Loading training data..." in result.output
    assert "Extracting activations..." in result.output
    assert "Encoding labels..." in result.output
    assert "Splitting data..." in result.output
    assert "Training probe..." in result.output
    assert "[4/4] Evaluating..." in result.output
    assert "SUMMARY" in result.output

    # Verify multi-label metrics are used for both modes
    assert "Hamming Loss" in result.output
    assert "Exact Match" in result.output

    # Verify probe file was created
    assert output_probe.exists(), f"Probe file not created at {output_probe}"

    # Load and verify probe
    probe = ConceptProbe.load(output_probe)

    # Check probe attributes
    assert probe.layer_name == "block3/conv2/relu"
    assert probe.model_version == f"test_{mode}_v1"
    assert len(probe.concept_list) > 0, "Probe should have concepts"
    assert probe.classifier is not None
    assert probe.label_encoder is not None, "Encoder should be saved with probe"