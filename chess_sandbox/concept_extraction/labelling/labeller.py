"""Concept detection and labeling for chess positions."""

from dataclasses import dataclass, field
from typing import Literal

from .patterns import CONCEPT_PATTERNS


@dataclass
class Concept:
    """A chess concept label with validation metadata.

    All concepts originate from regex detection. Validation and temporal
    context are added later by LLM, probe, or human review.

    >>> c = Concept(name="pin")
    >>> c.validated_by is None
    True
    >>> c.temporal is None
    True
    >>> c2 = Concept(name="fork", validated_by="llm", temporal="actual")
    >>> c2.validated_by
    'llm'
    >>> c2.temporal
    'actual'
    >>> c3 = Concept(name="pin", validated_by="probe", temporal="actual")
    >>> c3.validated_by
    'probe'
    """

    name: str
    validated_by: Literal["llm", "human", "probe"] | None = None
    temporal: Literal["actual", "future", "hypothetical", "past"] | None = None
    reasoning: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        """Convert to dictionary for JSON serialization.

        >>> c = Concept(name="pin", validated_by="llm", temporal="actual")
        >>> d = c.to_dict()
        >>> d["name"]
        'pin'
        >>> d["validated_by"]
        'llm'
        >>> d["temporal"]
        'actual'
        """
        return {
            "name": self.name,
            "validated_by": self.validated_by,
            "temporal": self.temporal,
            "reasoning": self.reasoning,
        }

    @classmethod
    def from_dict(cls, data: dict[str, str | None]) -> "Concept":
        """Create from dictionary.

        >>> data = {"name": "pin", "validated_by": "llm", "temporal": "actual"}
        >>> c = Concept.from_dict(data)
        >>> c.name
        'pin'
        >>> c.validated_by
        'llm'
        """
        return cls(
            name=str(data["name"]),
            validated_by=data.get("validated_by"),  # type: ignore
            temporal=data.get("temporal"),  # type: ignore
            reasoning=data.get("reasoning"),  # type: ignore
        )


@dataclass
class LabelledPosition:
    """A chess position with associated annotation and detected concepts.

    Supports two-phase labeling:
    - Phase 1 (regex): Broad concept detection -> concepts with validated_by=None
    - Phase 2 (LLM): Updates concepts in-place with validated_by="llm" and temporal context

    >>> pos = LabelledPosition(
    ...     fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    ...     move_number=1,
    ...     side_to_move="white",
    ...     comment="Starting position",
    ...     game_id="test_game_1",
    ...     move_san="e4",
    ...     previous_fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    ...     concepts=[]
    ... )
    >>> pos.fen
    'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
    >>> pos.side_to_move
    'white'
    >>> pos.move_san
    'e4'
    >>> pos.concepts
    []
    >>> pos.validated_concepts
    []
    """

    fen: str
    move_number: int
    side_to_move: str
    comment: str
    game_id: str
    move_san: str
    previous_fen: str
    concepts: list[Concept] = field(default_factory=lambda: [])

    @property
    def validated_concepts(self) -> list[Concept]:
        """Get only validated concepts.

        >>> pos = LabelledPosition(
        ...     fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        ...     move_number=1,
        ...     side_to_move="white",
        ...     comment="Pin that knight",
        ...     game_id="test_game_1",
        ...     move_san="Nf3",
        ...     previous_fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        ...     concepts=[
        ...         Concept(name="pin"),
        ...         Concept(name="fork", validated_by="llm", temporal="actual")
        ...     ]
        ... )
        >>> len(pos.validated_concepts)
        1
        >>> pos.validated_concepts[0].name
        'fork'
        """
        return [c for c in self.concepts if c.validated_by is not None]

    @property
    def actual_concepts(self) -> list[Concept]:
        """Get only concepts with temporal='actual'.

        >>> pos = LabelledPosition(
        ...     fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        ...     move_number=1,
        ...     side_to_move="white",
        ...     comment="Pin that knight",
        ...     game_id="test_game_1",
        ...     move_san="Nf3",
        ...     previous_fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        ...     concepts=[
        ...         Concept(name="pin", validated_by="llm", temporal="actual"),
        ...         Concept(name="fork", validated_by="llm", temporal="future")
        ...     ]
        ... )
        >>> len(pos.actual_concepts)
        1
        >>> pos.actual_concepts[0].name
        'pin'
        """
        return [c for c in self.concepts if c.temporal == "actual"]

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        >>> pos = LabelledPosition(
        ...     fen="r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        ...     move_number=3,
        ...     side_to_move="white",
        ...     comment="Pin that knight",
        ...     game_id="gameknot_1160",
        ...     move_san="Nc6",
        ...     previous_fen="r1bqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
        ...     concepts=[Concept(name="pin")]
        ... )
        >>> d = pos.to_dict()
        >>> d['fen']
        'r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3'
        >>> d['move_san']
        'Nc6'
        >>> d['previous_fen']
        'r1bqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2'
        >>> len(d['concepts'])
        1
        >>> d['concepts'][0]['name']
        'pin'
        """
        return {
            "fen": self.fen,
            "move_number": self.move_number,
            "side_to_move": self.side_to_move,
            "comment": self.comment,
            "game_id": self.game_id,
            "move_san": self.move_san,
            "previous_fen": self.previous_fen,
            "concepts": [c.to_dict() for c in self.concepts],
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "LabelledPosition":
        """Create from dictionary.

        >>> data = {
        ...     "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        ...     "move_number": 1,
        ...     "side_to_move": "white",
        ...     "comment": "Initial position",
        ...     "game_id": "test_1",
        ...     "move_san": "e4",
        ...     "previous_fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        ...     "concepts": [{"name": "opening", "validated_by": None, "temporal": None}]
        ... }
        >>> pos = LabelledPosition.from_dict(data)
        >>> pos.game_id
        'test_1'
        >>> pos.move_san
        'e4'
        >>> len(pos.concepts)
        1
        >>> pos.concepts[0].name
        'opening'

        >>> data_with_validation = {
        ...     "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        ...     "move_number": 1,
        ...     "side_to_move": "white",
        ...     "comment": "There is a pin here",
        ...     "game_id": "test_2",
        ...     "move_san": "Nf3",
        ...     "previous_fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        ...     "concepts": [{"name": "pin", "validated_by": "llm", "temporal": "actual"}]
        ... }
        >>> pos2 = LabelledPosition.from_dict(data_with_validation)
        >>> pos2.concepts[0].name
        'pin'
        >>> pos2.concepts[0].validated_by
        'llm'
        >>> pos2.concepts[0].temporal
        'actual'
        """
        concepts_data = data.get("concepts", [])
        concepts = [Concept.from_dict(c) for c in concepts_data]  # type: ignore

        return cls(
            fen=str(data["fen"]),
            move_number=int(data["move_number"]),  # type: ignore
            side_to_move=str(data["side_to_move"]),
            comment=str(data["comment"]),
            game_id=str(data["game_id"]),
            move_san=str(data["move_san"]),
            previous_fen=str(data["previous_fen"]),
            concepts=concepts,
        )


def detect_concepts(comment: str) -> list[Concept]:
    """Detect chess concepts mentioned in a comment.

    >>> concepts = detect_concepts("Pin that knight to the bishop")
    >>> len(concepts)
    1
    >>> concepts[0].name
    'pin'
    >>> concepts[0].validated_by is None
    True
    >>> concepts = detect_concepts("The knight forks the king and rook")
    >>> concepts[0].name
    'fork'
    >>> concepts = detect_concepts("A sacrifice leads to a mating threat")
    >>> concept_names = sorted([c.name for c in concepts])
    >>> concept_names
    ['mating_threat', 'sacrifice']
    >>> detect_concepts("Just a normal move")
    []
    >>> concepts = detect_concepts("Zugzwang!")
    >>> concepts[0].name
    'zugzwang'
    """
    detected: list[Concept] = []
    comment_lower = comment.lower()

    for concept_name, patterns in CONCEPT_PATTERNS.items():
        for pattern in patterns:
            if pattern.search(comment_lower):
                detected.append(Concept(name=concept_name))
                break  # Only add each concept once

    return detected


def label_positions(positions: list[LabelledPosition]) -> list[LabelledPosition]:
    """Add concept labels to positions based on their comments.

    >>> from chess_sandbox.concept_extraction.labelling.labeller import LabelledPosition
    >>> pos1 = LabelledPosition(
    ...     fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    ...     move_number=1,
    ...     side_to_move="white",
    ...     comment="Pin the knight",
    ...     game_id="game1",
    ...     move_san="e4",
    ...     previous_fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    ...     concepts=[]
    ... )
    >>> pos2 = LabelledPosition(
    ...     fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    ...     move_number=2,
    ...     side_to_move="black",
    ...     comment="Fork with the knight",
    ...     game_id="game1",
    ...     move_san="Nf3",
    ...     previous_fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    ...     concepts=[]
    ... )
    >>> labeled = label_positions([pos1, pos2])
    >>> labeled[0].concepts[0].name
    'pin'
    >>> labeled[1].concepts[0].name
    'fork'
    """
    labeled: list[LabelledPosition] = []
    for position in positions:
        concepts = detect_concepts(position.comment)
        labeled_position = LabelledPosition(
            fen=position.fen,
            move_number=position.move_number,
            side_to_move=position.side_to_move,
            comment=position.comment,
            game_id=position.game_id,
            move_san=position.move_san,
            previous_fen=position.previous_fen,
            concepts=concepts,
        )
        labeled.append(labeled_position)
    return labeled
