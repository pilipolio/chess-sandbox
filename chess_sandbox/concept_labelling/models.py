"""Data models for chess concept labeling."""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Concept:
    """A chess concept label with validation metadata.

    All concepts originate from regex detection. Validation and temporal
    context are added later by LLM or human review.

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
    """

    name: str
    validated_by: Literal["llm", "human"] | None = None
    temporal: Literal["actual", "threat", "hypothetical", "past"] | None = None

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
        return {"name": self.name, "validated_by": self.validated_by, "temporal": self.temporal}

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
    ...     concepts=[]
    ... )
    >>> pos.fen
    'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
    >>> pos.side_to_move
    'white'
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
    concepts: list[Concept] = field(default_factory=list)

    @property
    def validated_concepts(self) -> list[Concept]:
        """Get only validated concepts.

        >>> pos = LabelledPosition(
        ...     fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        ...     move_number=1,
        ...     side_to_move="white",
        ...     comment="Pin that knight",
        ...     game_id="test_game_1",
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
        ...     concepts=[
        ...         Concept(name="pin", validated_by="llm", temporal="actual"),
        ...         Concept(name="fork", validated_by="llm", temporal="threat")
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
        ...     concepts=[Concept(name="pin")]
        ... )
        >>> d = pos.to_dict()
        >>> d['fen']
        'r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3'
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
        ...     "concepts": [{"name": "opening", "validated_by": None, "temporal": None}]
        ... }
        >>> pos = LabelledPosition.from_dict(data)
        >>> pos.game_id
        'test_1'
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
            concepts=concepts,
        )
