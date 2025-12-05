"""Helper utilities for chess puzzle trainer."""

import io
from typing import cast

import cairosvg  # pyright: ignore[reportMissingTypeStubs]
import chess
import chess.svg
from PIL import Image


def generate_board_image(fen: str, size: int = 240) -> Image.Image:
    """Generate PNG board image from FEN position, flipped for Black to move."""
    board = chess.Board(fen)
    svg_content = chess.svg.board(board, size=size, flipped=(board.turn == chess.BLACK))
    png_bytes = cast(bytes, cairosvg.svg2png(bytestring=svg_content.encode()))  # pyright: ignore[reportUnknownMemberType]
    return Image.open(io.BytesIO(png_bytes))
