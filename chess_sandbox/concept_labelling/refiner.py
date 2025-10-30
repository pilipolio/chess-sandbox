"""LLM-based concept validation and temporal context extraction."""

import os
from dataclasses import dataclass
from textwrap import dedent

from openai import OpenAI
from pydantic import BaseModel, Field

from .models import Concept, LabelledPosition


class SingleConceptRefinement(BaseModel):
    """LLM validation result for a single concept.

    This model is used for structured output from the LLM to validate
    one regex-detected concept and extract temporal context.
    """

    is_valid: bool = Field(description="Whether the concept is truly discussed in the comment (not a false positive)")
    temporal: str | None = Field(
        description=(
            "Temporal context if valid: 'actual' (exists NOW), 'threat' (future), "
            "'hypothetical' (if/could/would), 'past' (already happened). None if invalid."
        )
    )
    reasoning: str = Field(description="Brief explanation of validation decision (2-3 sentences)")


@dataclass
class Refiner:
    """Validates regex concept matches using lightweight LLM.

    Uses GPT-4o-mini or similar model to:
    1. Process each concept individually with focused LLM calls
    2. Filter false positives from regex matches
    3. Extract temporal context (actual vs threat vs hypothetical)
    4. Return new Concept objects with validation metadata and reasoning
    """

    PROMPT = dedent("""
        You are a chess expert validating a single concept label extracted from a game annotation.

        POSITION: Move {move_number}, {side_to_move} to move
        COMMENT: "{comment}"
        CONCEPT TO VALIDATE: "{concept_name}"

        Determine:

        1. **Is this concept truly discussed in the comment?**
           - FALSE POSITIVE examples: "material" wrongly matched as "mate"
           - Concept detected by regex but not actually discussed

        2. **If VALID, what is the TEMPORAL CONTEXT?**
           - 'actual': Concept exists in the current position NOW
             (e.g., "there is a pin", "has a fork", "is a passed pawn")
           - 'threat': Concept is threatened/possible in future moves
             (e.g., "threatening mate", "can fork", "will be passed pawns")
           - 'hypothetical': Discussing "if/could/would" scenarios
             (e.g., "if black plays Nf6 then there would be a pin")
           - 'past': Referring to previous moves that already happened
             (e.g., "the pin was broken", "after the fork")

        Be strict: only validate if the concept is clearly mentioned in the comment.
    """).strip()

    llm_model: str
    client: OpenAI

    @classmethod
    def create(cls, params: dict[str, str]) -> "Refiner":
        """Create Refiner from configuration parameters.

        Args:
            params: Dictionary with 'llm_model' key (e.g., {'llm_model': 'gpt-4o-mini'})

        Returns:
            Configured Refiner instance
        """
        return cls(
            llm_model=params.get("llm_model", "gpt-4o-mini"),
            client=OpenAI(api_key=os.environ.get("OPENAI_API_KEY")),
        )

    def refine(self, position: LabelledPosition) -> list[Concept]:
        """Validate and refine concept labels for a single position.

        Processes each concept individually with focused LLM calls to determine
        validity, temporal context, and reasoning.

        Args:
            position: LabelledPosition with regex-detected concepts

        Returns:
            List of new Concept objects with validation metadata and reasoning
        """
        refined_concepts: list[Concept] = []
        valid_temporal_values = {"actual", "threat", "hypothetical", "past"}

        for concept in position.concepts:
            prompt = self.PROMPT.format(
                move_number=position.move_number,
                side_to_move=position.side_to_move,
                comment=position.comment,
                concept_name=concept.name,
            )

            response = self.client.responses.parse(
                model=self.llm_model,
                input=prompt,
                text_format=SingleConceptRefinement,
            )

            # Extract parsed output from response
            # When using reasoning models, first item might be ReasoningItem
            message = next((item for item in response.output if item.type == "message"), None)  # type: ignore

            if not message:
                raise ValueError("No message found in LLM response output")

            text = message.content[0]  # type: ignore
            assert text.type == "output_text", f"Unexpected content type: {text.type}"  # type: ignore

            if not text.parsed:  # type: ignore
                raise ValueError("Could not parse LLM response into SingleConceptRefinement")

            refinement: SingleConceptRefinement = text.parsed  # type: ignore

            # Build new Concept object with refinement metadata
            if refinement.is_valid and refinement.temporal in valid_temporal_values:
                refined_concept = Concept(
                    name=concept.name,
                    validated_by="llm",
                    temporal=refinement.temporal,  # type: ignore
                    reasoning=refinement.reasoning,
                )
            else:
                # Keep concept but mark as unvalidated with reasoning explaining why
                refined_concept = Concept(
                    name=concept.name,
                    validated_by=None,
                    temporal=None,
                    reasoning=refinement.reasoning,
                )

            refined_concepts.append(refined_concept)

        return refined_concepts
