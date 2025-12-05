"""Toy chess curriculum for basic exercises."""

from chess_sandbox.puzzles_trainer.toy_curriculum.formats import (
    fen_to_editor_url,
    fen_to_piece_list,
    parse_piece_list,
    piece_list_to_fen,
)
from chess_sandbox.puzzles_trainer.toy_curriculum.generators import (
    GeneratedPosition,
    generate_capture_exercise,
    generate_movement_exercise,
)
from chess_sandbox.puzzles_trainer.toy_curriculum.paths import (
    find_shortest_path,
    knight_shortest_path,
)
from chess_sandbox.puzzles_trainer.toy_curriculum.tasks import (
    ToyTaskType,
    create_toy_curriculum,
    format_capture_sequence,
    format_movement_path,
)

__all__ = [
    "fen_to_editor_url",
    "fen_to_piece_list",
    "parse_piece_list",
    "piece_list_to_fen",
    "GeneratedPosition",
    "generate_capture_exercise",
    "generate_movement_exercise",
    "find_shortest_path",
    "knight_shortest_path",
    "ToyTaskType",
    "create_toy_curriculum",
    "format_capture_sequence",
    "format_movement_path",
]
