"""LLM-based concept validation and temporal context extraction."""

import os
from dataclasses import dataclass
from textwrap import dedent

from openai import OpenAI
from pydantic import BaseModel, Field

from .models import LabelledPosition


class ConceptValidation(BaseModel):
    """A concept validation result from LLM."""

    concept: str = Field(description="The chess concept name (e.g., 'pin', 'fork', 'passed_pawn')")
    temporal: str = Field(
        description=(
            "Temporal context: 'actual' (exists NOW), 'threat' (future), "
            "'hypothetical' (if/could/would), 'past' (already happened)"
        )
    )


class ConceptRefinement(BaseModel):
    """LLM-validated concept labels with temporal context.

    This model is used for structured output from the LLM to validate
    regex-detected concepts and extract temporal context.
    """

    validated_concepts: list[ConceptValidation] = Field(
        description=(
            "Concepts that exist or are discussed in the comment, with their temporal context. "
            "Each concept should specify: concept name and temporal context."
        )
    )
    false_positives: list[str] = Field(
        description=("Regex matches that are incorrect or irrelevant " "(e.g., 'material' wrongly matched as 'mate')")
    )
    reasoning: str = Field(description="Brief explanation of validation decisions (2-3 sentences)")


@dataclass
class Refiner:
    """Validates regex concept matches using lightweight LLM.

    Uses GPT-4o-mini or similar model to:
    1. Filter false positives from regex matches
    2. Extract temporal context (actual vs threat vs hypothetical)
    3. Validate concept appropriateness given the comment
    4. Update position.concepts in-place with validation metadata

    Example:
        >>> import os
        >>> from openai import OpenAI
        >>> from chess_sandbox.concept_labelling.models import Concept
        >>> # Note: requires OPENAI_API_KEY environment variable
        >>> refiner = Refiner.create({"llm_model": "gpt-4o-mini"})  # doctest: +SKIP
        >>> position = LabelledPosition(  # doctest: +SKIP
        ...     fen="6k1/p1r1qpp1/1p2pn2/3r4/P2n4/3B3R/1B2QPPP/3R2K1 w - - 3 27",
        ...     move_number=27,
        ...     side_to_move="white",
        ...     comment="...a4 is now a target and the a & b pawns can eventually be nasty passed pawns.....",
        ...     game_id="game_10000",
        ...     concepts=[Concept(name="passed_pawn")]
        ... )
        >>> refinement = refiner.refine(position)  # doctest: +SKIP
        >>> position.concepts[0].temporal  # doctest: +SKIP
        'threat'
        >>> position.concepts[0].validated_by  # doctest: +SKIP
        'llm'
    """

    PROMPT = dedent("""
        You are a chess expert validating concept labels extracted from game annotations.

        POSITION: Move {move_number}, {side_to_move} to move
        COMMENT: "{comment}"
        REGEX DETECTED: {concepts_raw}

        For each detected concept, determine:

        1. **Is it a FALSE POSITIVE?**
           - Example: "material" wrongly matched as "mate"
           - Example: Concept mentioned but not actually discussed in comment

        2. **What is the TEMPORAL CONTEXT?**
           - 'actual': Concept exists in the current position NOW
             (e.g., "there is a pin", "has a fork", "is a passed pawn")
           - 'threat': Concept is threatened/possible in future moves
             (e.g., "threatening mate", "can fork", "will be passed pawns")
           - 'hypothetical': Discussing "if/could/would" scenarios
             (e.g., "if black plays Nf6 then there would be a pin")
           - 'past': Referring to previous moves that already happened
             (e.g., "the pin was broken", "after the fork")

        Only validate concepts that are clearly mentioned in the comment.
        Be strict: if the comment doesn't clearly discuss a concept, mark it as false positive.
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

    def refine(self, position: LabelledPosition) -> ConceptRefinement:
        """Validate and refine concept labels for a single position.

        Updates position.concepts in-place by setting validated_by="llm" and
        temporal context for validated concepts.

        Args:
            position: LabelledPosition with regex-detected concepts

        Returns:
            ConceptRefinement with validated concepts and temporal context
        """
        concept_names = [c.name for c in position.concepts]
        prompt = self.PROMPT.format(
            move_number=position.move_number,
            side_to_move=position.side_to_move,
            comment=position.comment,
            concepts_raw=concept_names,
        )

        response = self.client.responses.parse(
            model=self.llm_model,
            input=prompt,
            text_format=ConceptRefinement,
        )

        # Extract parsed output from response
        # When using reasoning models, first item might be ReasoningItem
        message = next((item for item in response.output if item.type == "message"), None)  # type: ignore

        if not message:
            raise ValueError("No message found in LLM response output")

        text = message.content[0]  # type: ignore
        assert text.type == "output_text", f"Unexpected content type: {text.type}"  # type: ignore

        if not text.parsed:  # type: ignore
            raise ValueError("Could not parse LLM response into ConceptRefinement")

        refinement: ConceptRefinement = text.parsed  # type: ignore

        # Update concepts in-place
        # Only validate concepts with valid temporal values
        valid_temporal_values = {"actual", "threat", "hypothetical", "past"}
        validation_map = {
            cv.concept: cv.temporal for cv in refinement.validated_concepts if cv.temporal in valid_temporal_values
        }

        for concept in position.concepts:
            if concept.name in validation_map:
                concept.validated_by = "llm"
                concept.temporal = validation_map[concept.name]  # type: ignore

        return refinement
