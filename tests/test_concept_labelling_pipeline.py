"""Integration tests for concept extraction labelling pipeline CLI."""

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from chess_sandbox.concept_extraction.labelling.pipeline import main


@pytest.fixture
def sample_pgn_files(tmp_path: Path) -> Path:
    """Create sample PGN files with various annotated concepts."""
    pgn_dir = tmp_path / "pgn_input"
    pgn_dir.mkdir()

    pgn1 = pgn_dir / "game1.pgn"
    pgn1.write_text("""[Event "Test Tournament"]
[Site "Test"]
[Date "2024.01.01"]
[Round "1"]
[White "Player A"]
[Black "Player B"]
[Result "1-0"]

1. e4 e5 { Opening with king pawn } 2. Nf3 Nc6 { Developing the knight }
3. Bb5 a6 { Pin that knight to the king } 4. Ba4 Nf6
5. O-O Be7 { The knight forks the queen and rook } 1-0
""")

    pgn2 = pgn_dir / "game2.pgn"
    pgn2.write_text("""[Event "Test Game 2"]
[Site "Test"]
[Date "2024.01.02"]
[Round "1"]
[White "Player C"]
[Black "Player D"]
[Result "0-1"]

1. d4 d5 { Queen's pawn opening } 2. c4 c6
3. Nc3 Nf6 { A brilliant sacrifice here } 4. Nf3 e6
5. e3 Nbd7 { Creating a mating threat on the kingside } 0-1
""")

    return pgn_dir


def test_pipeline_with_limit(sample_pgn_files: Path, tmp_path: Path) -> None:
    """Test pipeline with --limit option to process only subset of files."""
    output_file = tmp_path / "output_limited.jsonl"

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--input-dir",
            str(sample_pgn_files),
            "--output",
            str(output_file),
            "--limit",
            "1",
        ],
    )

    assert result.exit_code == 0
    assert "Processing only first 1 files" in result.output
    assert output_file.exists()

    positions: list[dict[str, object]] = []
    with output_file.open() as f:
        for line in f:
            if line.strip():
                positions.append(json.loads(line))

    # Should have positions from only 2 games
    game_ids = {str(pos["game_id"]) for pos in positions}
    assert len(game_ids) == 1, f"Expected 1 game, got {len(game_ids)}"


def test_pipeline_empty_directory(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    output_file = tmp_path / "output_empty.jsonl"

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--input-dir",
            str(empty_dir),
            "--output",
            str(output_file),
        ],
    )

    assert result.exit_code == 0
    assert "Extracted 0 positions" in result.output
