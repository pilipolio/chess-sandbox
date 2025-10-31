"""Tests for Modal pipeline wrapper (Option 2 implementation)."""

from pathlib import Path

from click.testing import CliRunner

from chess_sandbox.concept_labelling.pipeline import main


def test_cli_runner_integration(tmp_path: Path) -> None:
    """Test that CliRunner can invoke the pipeline CLI successfully.

    This validates that the Modal wrapper approach (Option 2) will work,
    since Modal uses CliRunner internally to invoke the Click command.
    """
    # Create a minimal test PGN
    pgn_dir = tmp_path / "pgns"
    pgn_dir.mkdir()

    pgn_content = """[Event "Test Game"]
[Site "Test"]
[Date "2024.01.01"]
[Round "1"]
[White "Player1"]
[Black "Player2"]
[Result "1-0"]

1. e4 e5 { This move attacks the center } 1-0
"""
    (pgn_dir / "test.pgn").write_text(pgn_content)

    output_file = tmp_path / "output.jsonl"

    # Use CliRunner (same as Modal wrapper)
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--input-dir",
            str(pgn_dir),
            "--output",
            str(output_file),
            "--limit",
            "1",
        ],
    )

    # Verify success
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    assert "Parsing PGN files" in result.output
    assert "Extracted" in result.output
    assert output_file.exists(), "Output file not created"

    # Verify JSONL output
    content = output_file.read_text()
    assert len(content.strip()) > 0, "Output file is empty"
    assert '"fen":' in content, "JSONL missing FEN field"


def test_cli_runner_with_error_handling(tmp_path: Path) -> None:
    """Test that CliRunner properly handles errors (e.g., missing directory)."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--input-dir",
            "/nonexistent/path",
            "--output",
            str(tmp_path / "output.jsonl"),
        ],
    )

    # Should fail with non-zero exit code
    assert result.exit_code != 0
    assert result.exception is not None or "Error" in result.output
