"""Concept detection and labeling for chess positions."""

from .models import Concept, LabelledPosition
from .patterns import CONCEPT_PATTERNS


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

    >>> from chess_sandbox.concept_labelling.models import LabelledPosition
    >>> pos1 = LabelledPosition(
    ...     fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    ...     move_number=1,
    ...     side_to_move="white",
    ...     comment="Pin the knight",
    ...     game_id="game1",
    ...     concepts=[]
    ... )
    >>> pos2 = LabelledPosition(
    ...     fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    ...     move_number=2,
    ...     side_to_move="black",
    ...     comment="Fork with the knight",
    ...     game_id="game1",
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
            concepts=concepts,
        )
        labeled.append(labeled_position)
    return labeled
