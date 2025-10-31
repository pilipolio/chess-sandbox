"""Tests for batch-based LLM concept refinement."""

import json
from collections.abc import Generator
from pathlib import Path

import httpx
import pytest
import respx

from chess_sandbox.concept_labelling.batch_refiner import BatchRefiner
from chess_sandbox.concept_labelling.models import Concept, LabelledPosition


@pytest.fixture
def temp_batch_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for batch files."""
    batch_dir = tmp_path / "batches"
    batch_dir.mkdir()
    return batch_dir


@pytest.fixture
def batch_refiner(temp_batch_dir: Path) -> Generator[BatchRefiner, None, None]:
    """Create a BatchRefiner instance with temporary directory."""
    yield BatchRefiner.create(model="gpt-5-nano", batch_dir=temp_batch_dir)


@pytest.fixture
def sample_positions() -> list[LabelledPosition]:
    """Create sample positions for testing."""
    return [
        LabelledPosition(
            fen="r2q1r2/ppp1k1pp/2Bp1n2/2b1p1N1/4P1b1/8/PPPP1PPP/RNBQ1RK1 w - - 1 9",
            move_number=9,
            side_to_move="white",
            comment="I cannot play f3 because of the pin from the bishop",
            game_id="test_game_1",
            move_san="Ng5",
            previous_fen="r2q1r2/ppp1k1pp/2Bp1n2/2b1p3/4P1b1/8/PPPP1PPP/RNBQ1RK1 b - - 0 8",
            concepts=[Concept(name="pin"), Concept(name="fork")],
        ),
        LabelledPosition(
            fen="r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/3P1N2/PPP2PPP/RNBQKB1R w KQkq - 2 4",
            move_number=4,
            side_to_move="white",
            comment="Threatening the knight",
            game_id="test_game_2",
            move_san="d3",
            previous_fen="r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 3",
            concepts=[Concept(name="mating_threat")],
        ),
    ]


def test_prepare_batch_input(batch_refiner: BatchRefiner, sample_positions: list[LabelledPosition]) -> None:
    """Test preparation of batch input JSONL file."""
    batch_file = batch_refiner.prepare_batch_input(sample_positions)

    # Verify batch file was created
    assert batch_file.exists()
    assert batch_file.suffix == ".jsonl"

    # Verify mapping file was created
    timestamp = batch_file.stem.replace("batch_input_", "")
    mapping_file = batch_refiner.batch_dir / f"batch_mapping_{timestamp}.json"
    assert mapping_file.exists()

    # Load and verify batch requests
    with batch_file.open("r") as f:
        requests = [json.loads(line) for line in f]

    # Should have 3 requests (2 concepts for pos 0, 1 concept for pos 1)
    assert len(requests) == 3

    # Verify first request structure
    req = requests[0]
    assert req["custom_id"] == "pos_0_concept_0"
    assert req["method"] == "POST"
    assert req["url"] == "/v1/responses"
    assert req["body"]["model"] == "gpt-5-nano"
    assert "pin" in req["body"]["input"]
    assert "text_format" in req["body"]

    # Verify mapping
    with mapping_file.open("r") as f:
        mapping = json.load(f)

    assert "pos_0_concept_0" in mapping
    assert mapping["pos_0_concept_0"]["position_idx"] == 0
    assert mapping["pos_0_concept_0"]["concept_idx"] == 0
    assert mapping["pos_0_concept_0"]["concept_name"] == "pin"


@respx.mock
def test_submit_batch(batch_refiner: BatchRefiner, sample_positions: list[LabelledPosition]) -> None:
    """Test batch submission to OpenAI API."""
    batch_file = batch_refiner.prepare_batch_input(sample_positions)

    # Mock file upload
    respx.post("https://api.openai.com/v1/files").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "file-abc123",
                "object": "file",
                "purpose": "batch",
                "filename": str(batch_file),
                "bytes": 1000,
                "created_at": 1234567890,
            },
        )
    )

    # Mock batch creation
    respx.post("https://api.openai.com/v1/batches").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "batch-xyz789",
                "object": "batch",
                "endpoint": "/v1/responses",
                "input_file_id": "file-abc123",
                "completion_window": "24h",
                "status": "validating",
                "created_at": 1234567890,
                "request_counts": {"total": 0, "completed": 0, "failed": 0},
            },
        )
    )

    batch_id = batch_refiner.submit_batch(batch_file)

    assert batch_id == "batch-xyz789"

    # Verify metadata file was created
    metadata_file = batch_refiner.batch_dir / f"batch_{batch_id}.json"
    assert metadata_file.exists()

    with metadata_file.open("r") as f:
        metadata = json.load(f)

    assert metadata["batch_id"] == "batch-xyz789"
    assert metadata["input_file_id"] == "file-abc123"
    assert metadata["status"] == "validating"
    assert metadata["model"] == "gpt-5-nano"


@respx.mock
def test_poll_status(batch_refiner: BatchRefiner, temp_batch_dir: Path) -> None:
    """Test polling batch job status."""
    batch_id = "batch-xyz789"

    # Create metadata file
    metadata_file = temp_batch_dir / f"batch_{batch_id}.json"
    with metadata_file.open("w") as f:
        json.dump({"batch_id": batch_id, "status": "in_progress"}, f)

    # Mock batch status retrieval
    respx.get(f"https://api.openai.com/v1/batches/{batch_id}").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": batch_id,
                "status": "in_progress",
                "request_counts": {"total": 100, "completed": 50, "failed": 0},
                "created_at": 1234567890,
                "in_progress_at": 1234567900,
                "completed_at": None,
                "failed_at": None,
                "output_file_id": None,
                "error_file_id": None,
            },
        )
    )

    status_info = batch_refiner.poll_status(batch_id)

    assert status_info["batch_id"] == batch_id
    assert status_info["status"] == "in_progress"
    assert status_info["request_counts"]["total"] == 100
    assert status_info["request_counts"]["completed"] == 50
    assert status_info["request_counts"]["failed"] == 0


@respx.mock
def test_download_results(batch_refiner: BatchRefiner) -> None:
    """Test downloading batch results."""
    batch_id = "batch-xyz789"

    # Mock batch retrieval
    respx.get(f"https://api.openai.com/v1/batches/{batch_id}").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": batch_id,
                "status": "completed",
                "output_file_id": "file-output123",
                "request_counts": {"total": 3, "completed": 3, "failed": 0},
                "created_at": 1234567890,
                "in_progress_at": 1234567900,
                "completed_at": 1234567950,
                "failed_at": None,
                "error_file_id": None,
            },
        )
    )

    # Mock file content download
    results_content = b'{"custom_id": "pos_0_concept_0", "response": {"body": {"output": []}}}\n'
    respx.get("https://api.openai.com/v1/files/file-output123/content").mock(
        return_value=httpx.Response(200, content=results_content)
    )

    results_file = batch_refiner.download_results(batch_id)

    assert results_file.exists()
    assert results_file.name == f"batch_output_{batch_id}.jsonl"

    # Verify content was written
    with results_file.open("rb") as f:
        content = f.read()
    assert content == results_content


def test_process_results(batch_refiner: BatchRefiner, sample_positions: list[LabelledPosition]) -> None:
    """Test processing batch results and updating positions."""
    # Prepare batch input to create mapping file
    batch_file = batch_refiner.prepare_batch_input(sample_positions)
    batch_id = "batch-test123"

    # Create metadata file linking batch_id to batch_file
    metadata_file = batch_refiner.batch_dir / f"batch_{batch_id}.json"
    with metadata_file.open("w") as f:
        json.dump({"batch_id": batch_id, "batch_file": str(batch_file)}, f)

    # Create mock results file
    results_file = batch_refiner.batch_dir / f"batch_output_{batch_id}.jsonl"
    results = [
        {
            "custom_id": "pos_0_concept_0",
            "response": {
                "body": {
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {
                                    "type": "output_text",
                                    "parsed": {
                                        "is_valid": True,
                                        "temporal": "actual",
                                        "reasoning": "Pin is present in position",
                                    },
                                }
                            ],
                        }
                    ]
                }
            },
        },
        {
            "custom_id": "pos_0_concept_1",
            "response": {
                "body": {
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {
                                    "type": "output_text",
                                    "parsed": {
                                        "is_valid": False,
                                        "temporal": None,
                                        "reasoning": "No fork present",
                                    },
                                }
                            ],
                        }
                    ]
                }
            },
        },
        {
            "custom_id": "pos_1_concept_0",
            "response": {
                "body": {
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {
                                    "type": "output_text",
                                    "parsed": {
                                        "is_valid": False,
                                        "temporal": None,
                                        "reasoning": "False positive - no mating threat",
                                    },
                                }
                            ],
                        }
                    ]
                }
            },
        },
    ]

    with results_file.open("w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    # Process results
    updated_positions = batch_refiner.process_results(results_file, sample_positions)

    # Verify position 0 concepts
    assert len(updated_positions[0].concepts) == 2
    pin_concept = updated_positions[0].concepts[0]
    assert pin_concept.name == "pin"
    assert pin_concept.validated_by == "llm"
    assert pin_concept.temporal == "actual"
    assert pin_concept.reasoning == "Pin is present in position"

    fork_concept = updated_positions[0].concepts[1]
    assert fork_concept.name == "fork"
    assert fork_concept.validated_by is None
    assert fork_concept.temporal is None
    assert fork_concept.reasoning == "No fork present"

    # Verify position 1 concepts
    assert len(updated_positions[1].concepts) == 1
    mate_concept = updated_positions[1].concepts[0]
    assert mate_concept.name == "mating_threat"
    assert mate_concept.validated_by is None


def test_process_results_with_errors(batch_refiner: BatchRefiner, sample_positions: list[LabelledPosition]) -> None:
    """Test processing batch results with failed requests."""
    batch_file = batch_refiner.prepare_batch_input(sample_positions)
    batch_id = "batch-error123"

    # Create metadata file
    metadata_file = batch_refiner.batch_dir / f"batch_{batch_id}.json"
    with metadata_file.open("w") as f:
        json.dump({"batch_id": batch_id, "batch_file": str(batch_file)}, f)

    # Create results with one error
    results_file = batch_refiner.batch_dir / f"batch_output_{batch_id}.jsonl"
    results = [
        {
            "custom_id": "pos_0_concept_0",
            "response": {
                "body": {
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {
                                    "type": "output_text",
                                    "parsed": {
                                        "is_valid": True,
                                        "temporal": "actual",
                                        "reasoning": "Valid concept",
                                    },
                                }
                            ],
                        }
                    ]
                }
            },
        },
        {"custom_id": "pos_0_concept_1", "error": {"code": "rate_limit_exceeded", "message": "Rate limit hit"}},
    ]

    with results_file.open("w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    # Process results
    batch_refiner.process_results(results_file, sample_positions)

    # Verify error file was created
    error_file = batch_refiner.batch_dir / f"errors_{batch_id}.json"
    assert error_file.exists()

    with error_file.open("r") as f:
        errors = json.load(f)

    assert len(errors) == 1
    assert errors[0]["custom_id"] == "pos_0_concept_1"
    assert errors[0]["error"]["code"] == "rate_limit_exceeded"


def test_get_batch_metadata(batch_refiner: BatchRefiner, temp_batch_dir: Path) -> None:
    """Test retrieving batch metadata."""
    batch_id = "batch-meta123"
    metadata = {
        "batch_id": batch_id,
        "batch_file": "batch_input_20251030_120000.jsonl",
        "submitted_at": "2025-10-30T12:00:00",
        "status": "completed",
    }

    metadata_file = temp_batch_dir / f"batch_{batch_id}.json"
    with metadata_file.open("w") as f:
        json.dump(metadata, f)

    retrieved_metadata = batch_refiner.get_batch_metadata(batch_id)

    assert retrieved_metadata == metadata


def test_get_batch_metadata_not_found(batch_refiner: BatchRefiner) -> None:
    """Test error when batch metadata doesn't exist."""
    with pytest.raises(ValueError, match="No metadata found for batch"):
        batch_refiner.get_batch_metadata("batch-nonexistent")
