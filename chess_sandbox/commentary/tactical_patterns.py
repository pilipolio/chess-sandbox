"""Tactical pattern detection using python-chess API.

This module provides hand-crafted tactical pattern detection to enhance
commentary with specific details about tactical themes like pins, forks, etc.
"""

from dataclasses import dataclass
from typing import List

import chess


@dataclass
class PinInfo:
    """Information about a pin in the position."""

    pinned_square: chess.Square
    pinned_piece: chess.Piece
    pinner_square: chess.Square
    pinner_piece: chess.Piece
    king_square: chess.Square | None  # If pinned to king
    valuable_square: chess.Square | None  # If pinned to valuable piece

    def describe(self) -> str:
        """Generate human-readable description of the pin."""
        pinned_name = chess.piece_name(self.pinned_piece.piece_type)
        pinner_name = chess.piece_name(self.pinner_piece.piece_type)
        pinned_sq = chess.square_name(self.pinned_square)
        pinner_sq = chess.square_name(self.pinner_square)

        if self.king_square:
            king_sq = chess.square_name(self.king_square)
            return (
                f"{pinned_name.capitalize()} on {pinned_sq} is pinned to the king on {king_sq} "
                f"by {pinner_name} on {pinner_sq}"
            )
        elif self.valuable_square:
            valuable_sq = chess.square_name(self.valuable_square)
            return (
                f"{pinned_name.capitalize()} on {pinned_sq} is pinned to valuable piece on {valuable_sq} "
                f"by {pinner_name} on {pinner_sq}"
            )
        else:
            return f"{pinned_name.capitalize()} on {pinned_sq} is pinned by {pinner_name} on {pinner_sq}"


@dataclass
class ForkInfo:
    """Information about a fork in the position."""

    forking_square: chess.Square
    forking_piece: chess.Piece
    attacked_squares: List[chess.Square]
    attacked_pieces: List[chess.Piece]

    def describe(self) -> str:
        """Generate human-readable description of the fork."""
        forking_name = chess.piece_name(self.forking_piece.piece_type)
        forking_sq = chess.square_name(self.forking_square)
        targets = [chess.piece_name(p.piece_type) for p in self.attacked_pieces]
        return f"{forking_name.capitalize()} on {forking_sq} attacks {', '.join(targets)}"


class TacticalPatternDetector:
    """Detects tactical patterns in chess positions using python-chess API."""

    def __init__(self, board: chess.Board):
        self.board = board

    def detect_pins(self) -> List[PinInfo]:
        """Detect all pins in the current position."""
        pins: List[PinInfo] = []
        color = self.board.turn

        # Check each square for pinned pieces
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece and piece.color == color:
                if self.board.is_pinned(color, square):
                    pin_info = self._analyze_pin(square, piece)
                    if pin_info:
                        pins.append(pin_info)

        return pins

    def _analyze_pin(self, square: chess.Square, piece: chess.Piece) -> PinInfo | None:
        """Analyze details of a pin on the given square."""
        # Find the king
        king_square = self.board.king(piece.color)
        if not king_square:
            return None

        # Find the pinner by checking all opponent pieces
        opponent_color = not piece.color
        for opponent_square in chess.SQUARES:
            opponent_piece = self.board.piece_at(opponent_square)
            if not opponent_piece or opponent_piece.color != opponent_color:
                continue

            # Check if this piece attacks through the pinned piece
            if self._is_pinning(opponent_square, opponent_piece, square, piece):
                # Determine if pinned to king or valuable piece
                king_sq = None
                valuable_sq = None

                if self._is_on_ray(opponent_square, square, king_square):
                    king_sq = king_square
                else:
                    # Check for valuable piece behind
                    valuable_sq = self._find_valuable_piece_behind(opponent_square, square, piece.color)

                return PinInfo(
                    pinned_square=square,
                    pinned_piece=piece,
                    pinner_square=opponent_square,
                    pinner_piece=opponent_piece,
                    king_square=king_sq,
                    valuable_square=valuable_sq,
                )

        return None

    def _is_pinning(
        self,
        pinner_square: chess.Square,
        pinner_piece: chess.Piece,
        pinned_square: chess.Square,
        pinned_piece: chess.Piece,
    ) -> bool:
        """Check if pinner_piece is pinning pinned_piece."""
        # Only sliding pieces can pin
        if pinner_piece.piece_type not in [
            chess.BISHOP,
            chess.ROOK,
            chess.QUEEN,
        ]:
            return False

        # Check if on same ray
        if not self._is_on_ray(pinner_square, pinned_square, None):
            return False

        # Temporarily remove the pinned piece and check if pinner attacks through
        board_copy = self.board.copy()
        board_copy.remove_piece_at(pinned_square)

        # Check if removing the piece exposes something valuable
        king_square = board_copy.king(pinned_piece.color)
        if king_square and board_copy.is_attacked_by(pinner_piece.color, king_square):
            return True

        # Check for valuable pieces behind
        valuable_behind = self._find_valuable_piece_behind(pinner_square, pinned_square, pinned_piece.color)
        return valuable_behind is not None

    def _is_on_ray(
        self,
        start: chess.Square,
        middle: chess.Square,
        end: chess.Square | None,
    ) -> bool:
        """Check if squares are on same diagonal/rank/file ray."""
        if end is None:
            # Just check if two squares are on a valid ray
            return (
                chess.square_file(start) == chess.square_file(middle)
                or chess.square_rank(start) == chess.square_rank(middle)
                or abs(chess.square_file(start) - chess.square_file(middle))
                == abs(chess.square_rank(start) - chess.square_rank(middle))
            )

        # Check if all three squares are aligned
        file_start = chess.square_file(start)
        rank_start = chess.square_rank(start)
        file_middle = chess.square_file(middle)
        rank_middle = chess.square_rank(middle)
        file_end = chess.square_file(end)
        rank_end = chess.square_rank(end)

        # Same file
        if file_start == file_middle == file_end:
            return (rank_start < rank_middle < rank_end) or (rank_start > rank_middle > rank_end)

        # Same rank
        if rank_start == rank_middle == rank_end:
            return (file_start < file_middle < file_end) or (file_start > file_middle > file_end)

        # Same diagonal
        file_diff1 = file_middle - file_start
        rank_diff1 = rank_middle - rank_start
        file_diff2 = file_end - file_middle
        rank_diff2 = rank_end - rank_middle

        if abs(file_diff1) == abs(rank_diff1) and abs(file_diff2) == abs(rank_diff2):
            # Check if directions match
            if file_diff1 * file_diff2 > 0 and rank_diff1 * rank_diff2 > 0:
                return True

        return False

    def _find_valuable_piece_behind(
        self, pinner_square: chess.Square, pinned_square: chess.Square, color: bool
    ) -> chess.Square | None:
        """Find a valuable piece behind the pinned piece on the same ray."""
        # Calculate direction from pinner to pinned
        file_diff = chess.square_file(pinned_square) - chess.square_file(pinner_square)
        rank_diff = chess.square_rank(pinned_square) - chess.square_rank(pinner_square)

        # Normalize to direction
        file_dir = 0 if file_diff == 0 else file_diff // abs(file_diff)
        rank_dir = 0 if rank_diff == 0 else rank_diff // abs(rank_diff)

        # Continue in the same direction from pinned square
        current_file = chess.square_file(pinned_square) + file_dir
        current_rank = chess.square_rank(pinned_square) + rank_dir

        while 0 <= current_file < 8 and 0 <= current_rank < 8:
            square = chess.square(current_file, current_rank)
            piece = self.board.piece_at(square)

            if piece and piece.color == color:
                # Found a piece of same color - check if valuable
                if piece.piece_type in [chess.QUEEN, chess.ROOK]:
                    return square
                # Found a piece but not valuable enough
                return None

            current_file += file_dir
            current_rank += rank_dir

        return None

    def detect_forks(self, min_value: int = 6) -> List[ForkInfo]:
        """Detect forks where a piece attacks multiple valuable targets.

        Args:
            min_value: Minimum total value of attacked pieces to consider a fork
                       (Q=9, R=5, B=3, N=3, P=1)
        """
        forks: List[ForkInfo] = []
        color = self.board.turn

        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0,  # Don't count king in fork value
        }

        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if not piece or piece.color != color:
                continue

            # Get all squares this piece attacks
            attacks = self.board.attacks(square)
            attacked_squares: List[chess.Square] = []
            attacked_pieces: List[chess.Piece] = []
            total_value = 0

            for attack_square in attacks:
                target = self.board.piece_at(attack_square)
                if target and target.color != color:
                    attacked_squares.append(attack_square)
                    attacked_pieces.append(target)
                    total_value += piece_values.get(target.piece_type, 0)

            # Consider it a fork if attacking multiple pieces worth enough
            if len(attacked_pieces) >= 2 and total_value >= min_value:
                forks.append(
                    ForkInfo(
                        forking_square=square,
                        forking_piece=piece,
                        attacked_squares=attacked_squares,
                        attacked_pieces=attacked_pieces,
                    )
                )

        return forks

    def get_tactical_context(self) -> str:
        """Generate tactical context description for the position."""
        context_parts: List[str] = []

        # Detect pins
        pins = self.detect_pins()
        if pins:
            context_parts.append("PINS DETECTED:")
            for pin in pins:
                context_parts.append(f"  - {pin.describe()}")

        # Detect forks
        forks = self.detect_forks()
        if forks:
            context_parts.append("FORKS DETECTED:")
            for fork in forks:
                context_parts.append(f"  - {fork.describe()}")

        return "\n".join(context_parts) if context_parts else ""
