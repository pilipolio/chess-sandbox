"""Tests for LLM-based concept refinement."""

from collections.abc import Generator
from typing import Any

import httpx
import pytest
import respx

from chess_sandbox.concept_labelling.models import Concept, LabelledPosition
from chess_sandbox.concept_labelling.refiner import ConceptRefinement, ConceptValidation, Refiner


def create_mock_response(
    validated_concepts: list[dict[str, str]], false_positives: list[str], reasoning: str
) -> dict[str, Any]:
    """Create a mock OpenAI API response for the Reasoning API.

    The response structure mimics what client.responses.parse() returns.
    The 'text' field must be a JSON string that can be parsed into ConceptRefinement.
    """
    import json

    refinement_data = {
        "validated_concepts": validated_concepts,
        "false_positives": false_positives,
        "reasoning": reasoning,
    }

    return {
        "id": "response_123",
        "object": "response",
        "created": 1234567890,
        "model": "gpt-4o-mini",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": json.dumps(refinement_data),  # SDK will parse this JSON to create the 'parsed' field
                    }
                ],
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    }


@pytest.fixture
def refiner() -> Generator[Refiner, None, None]:
    """Create a Refiner instance with mocked OpenAI client."""
    yield Refiner.create({"llm_model": "gpt-4o-mini"})


def test_concept_refinement_model() -> None:
    """Test ConceptRefinement Pydantic model."""
    refinement = ConceptRefinement(
        validated_concepts=[ConceptValidation(concept="pin", temporal="actual")],
        false_positives=[],
        reasoning="The comment explicitly mentions a pin in the current position.",
    )

    assert len(refinement.validated_concepts) == 1
    assert refinement.validated_concepts[0].concept == "pin"
    assert refinement.validated_concepts[0].temporal == "actual"
    assert refinement.false_positives == []
    assert "pin" in refinement.reasoning


@respx.mock
def test_refiner_actual_position(refiner: Refiner) -> None:
    """Test refinement of position with actual concept."""
    # Mock the OpenAI API response
    mock_response = create_mock_response(
        validated_concepts=[{"concept": "pin", "temporal": "actual"}],
        false_positives=[],
        reasoning="The comment explicitly mentions a pin preventing f3, which exists in the current position.",
    )

    respx.post("https://api.openai.com/v1/responses").mock(return_value=httpx.Response(200, json=mock_response))

    position = LabelledPosition(
        fen="r2q1r2/ppp1k1pp/2Bp1n2/2b1p1N1/4P1b1/8/PPPP1PPP/RNBQ1RK1 w - - 1 9",
        move_number=9,
        side_to_move="white",
        comment="I cannot play f3 because of the pin from the bishop",
        game_id="test_game_1",
        move_san="Ng5",
        previous_fen="r2q1r2/ppp1k1pp/2Bp1n2/2b1p3/4P1b1/8/PPPP1PPP/RNBQ1RK1 b - - 0 8",
        concepts=[Concept(name="pin")],
    )

    refinement = refiner.refine(position)

    # Check that position was updated
    assert len(position.validated_concepts) > 0
    validated_names = [c.name for c in position.validated_concepts]
    assert "pin" in validated_names

    # Find the pin concept and check its temporal context
    pin_concept = next(c for c in position.concepts if c.name == "pin")
    assert pin_concept.validated_by == "llm"
    assert pin_concept.temporal == "actual"

    # Check refinement result
    assert len(refinement.validated_concepts) > 0
    assert any(cv.concept == "pin" for cv in refinement.validated_concepts)
    assert len(refinement.false_positives) == 0


@respx.mock
def test_refiner_threat_position(refiner: Refiner) -> None:
    """Test refinement of position with threatened concept."""
    # Mock the OpenAI API response
    mock_response = create_mock_response(
        validated_concepts=[{"concept": "passed_pawn", "temporal": "threat"}],
        false_positives=[],
        reasoning="The comment refers to pawns that can EVENTUALLY become passed pawns, indicating a future threat.",
    )

    respx.post("https://api.openai.com/v1/responses").mock(return_value=httpx.Response(200, json=mock_response))

    position = LabelledPosition(
        fen="6k1/p1r1qpp1/1p2pn2/3r4/P2n4/3B3R/1B2QPPP/3R2K1 w - - 3 27",
        move_number=27,
        side_to_move="white",
        comment="The a & b pawns can eventually be nasty passed pawns",
        game_id="test_game_2",
        move_san="a4",
        previous_fen="6k1/p1r1qpp1/1p2pn2/3r4/4n3/3B3R/1B2QPPP/3R2K1 b - - 2 26",
        concepts=[Concept(name="passed_pawn")],
    )

    _ = refiner.refine(position)  # Modifies position in-place

    # Check that position was updated
    validated_names = [c.name for c in position.validated_concepts]

    # Check that passed_pawn is marked as threat or hypothetical
    if "passed_pawn" in validated_names:
        passed_pawn_concept = next(c for c in position.concepts if c.name == "passed_pawn")
        assert passed_pawn_concept.temporal in ["threat", "hypothetical"]


@respx.mock
def test_refiner_false_positive(refiner: Refiner) -> None:
    """Test refinement detects false positives (e.g., 'material' as 'mate')."""
    # Mock the OpenAI API response - marking mating_threat as false positive
    mock_response = create_mock_response(
        validated_concepts=[],
        false_positives=["mating_threat"],
        reasoning="The comment mentions 'material exchange', not a mating threat. This is a false positive.",
    )

    respx.post("https://api.openai.com/v1/responses").mock(return_value=httpx.Response(200, json=mock_response))

    position = LabelledPosition(
        fen="r1bqkb1r/p1pn1k2/1p3npp/3NpB2/4P3/1P3N2/P1P2PPP/R1BQK2R w KQ - 0 12",
        move_number=12,
        side_to_move="white",
        comment="This leaves a straightforward material exchange",
        game_id="test_game_3",
        move_san="Nd5",
        previous_fen="r1bqkb1r/p1pn1k2/1p3npp/4pB2/4P3/1P3N2/P1P2PPP/R1BQK2R b KQ - 0 11",
        concepts=[Concept(name="mating_threat")],  # "material" wrongly matched as "mate"
    )

    _ = refiner.refine(position)  # Modifies position in-place

    # Should detect this as false positive - concept should not be validated
    mating_threat_concept = next((c for c in position.concepts if c.name == "mating_threat"), None)
    assert mating_threat_concept is not None
    assert mating_threat_concept.validated_by is None  # Should not be validated
