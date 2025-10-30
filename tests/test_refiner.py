"""Tests for LLM-based concept refinement."""

from collections.abc import Generator
from typing import Any

import httpx
import pytest
import respx

from chess_sandbox.concept_labelling.models import Concept, LabelledPosition
from chess_sandbox.concept_labelling.refiner import Refiner


def create_mock_response(is_valid: bool, temporal: str | None, reasoning: str) -> dict[str, Any]:
    """Create a mock OpenAI API response for single concept validation.

    The response structure mimics what client.responses.parse() returns.
    The 'text' field must be a JSON string that can be parsed into SingleConceptRefinement.
    """
    import json

    refinement_data = {
        "is_valid": is_valid,
        "temporal": temporal,
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


@pytest.mark.asyncio
@respx.mock
async def test_refiner_actual_position(refiner: Refiner) -> None:
    """Test refinement of position with validated concept."""
    # Mock the OpenAI API response for a valid concept
    mock_response = create_mock_response(
        is_valid=True,
        temporal="actual",
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

    # Functional approach: get new concepts
    refined_concepts = await refiner.refine(position)

    # Verify we got one refined concept back
    assert len(refined_concepts) == 1

    # Check the refined concept has correct validation metadata
    refined_concept = refined_concepts[0]
    assert refined_concept.name == "pin"
    assert refined_concept.validated_by == "llm"
    assert refined_concept.temporal == "actual"
    assert (
        refined_concept.reasoning
        == "The comment explicitly mentions a pin preventing f3, which exists in the current position."
    )


@pytest.mark.asyncio
@respx.mock
async def test_refiner_false_positive(refiner: Refiner) -> None:
    """Test refinement detects false positives (e.g., 'material' wrongly matched as 'mate')."""
    # Mock the OpenAI API response - marking concept as invalid
    mock_response = create_mock_response(
        is_valid=False,
        temporal=None,
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

    # Functional approach: get new concepts
    refined_concepts = await refiner.refine(position)

    # Verify we got one refined concept back
    assert len(refined_concepts) == 1

    # Check the refined concept is marked as NOT validated (false positive)
    refined_concept = refined_concepts[0]
    assert refined_concept.name == "mating_threat"
    assert refined_concept.validated_by is None  # Not validated
    assert refined_concept.temporal is None
    assert (
        refined_concept.reasoning
        == "The comment mentions 'material exchange', not a mating threat. This is a false positive."
    )
