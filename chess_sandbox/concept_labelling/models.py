"""Data models for chess concept labeling."""

from dataclasses import dataclass, field


@dataclass
class LabelledPosition:
    """A chess position with associated annotation and detected concepts.

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
    """

    fen: str
    move_number: int
    side_to_move: str
    comment: str
    game_id: str
    concepts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        >>> pos = LabelledPosition(
        ...     fen="r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        ...     move_number=3,
        ...     side_to_move="white",
        ...     comment="Pin that knight",
        ...     game_id="gameknot_1160",
        ...     concepts=["pin"]
        ... )
        >>> d = pos.to_dict()
        >>> d['fen']
        'r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3'
        >>> d['concepts']
        ['pin']
        """
        return {
            "fen": self.fen,
            "move_number": self.move_number,
            "side_to_move": self.side_to_move,
            "comment": self.comment,
            "game_id": self.game_id,
            "concepts": self.concepts,
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
        ...     "concepts": ["opening"]
        ... }
        >>> pos = LabelledPosition.from_dict(data)
        >>> pos.game_id
        'test_1'
        >>> pos.concepts
        ['opening']
        """
        return cls(
            fen=str(data["fen"]),
            move_number=int(data["move_number"]),  # type: ignore
            side_to_move=str(data["side_to_move"]),
            comment=str(data["comment"]),
            game_id=str(data["game_id"]),
            concepts=list(data.get("concepts", [])),  # type: ignore
        )
