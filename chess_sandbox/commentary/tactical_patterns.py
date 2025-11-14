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
        """Detect all pins using native python-chess API.

        Detects both absolute pins (to king) and relative pins (to valuable pieces).
        Uses O(1) bitboard operations instead of O(n²) iteration with board copies.
        """
        pins: List[PinInfo] = []
        color = self.board.turn
        opponent_color = not color
        king_square = self.board.king(color)

        if not king_square:
            return pins

        # Get all friendly pieces (excluding king)
        friendly_pieces = (
            self.board.pieces(chess.PAWN, color)
            | self.board.pieces(chess.KNIGHT, color)
            | self.board.pieces(chess.BISHOP, color)
            | self.board.pieces(chess.ROOK, color)
            | self.board.pieces(chess.QUEEN, color)
        )

        # Track pieces already found pinned to avoid duplicates
        pinned_to_king: set[chess.Square] = set()

        # Phase 1: Detect absolute pins to king using native API
        for square in friendly_pieces:
            if self.board.is_pinned(color, square):
                pinned_to_king.add(square)
                pin_mask = self.board.pin(color, square)

                # Find the pinner along the pin ray
                # Check opponent pieces on the pin ray for sliding pieces
                for pinner_square in pin_mask:
                    pinner_piece = self.board.piece_at(pinner_square)
                    if (
                        pinner_piece
                        and pinner_piece.color == opponent_color
                        and pinner_piece.piece_type in [chess.BISHOP, chess.ROOK, chess.QUEEN]
                    ):
                        pinned_piece = self.board.piece_at(square)
                        assert pinned_piece is not None

                        pins.append(
                            PinInfo(
                                pinned_square=square,
                                pinned_piece=pinned_piece,
                                pinner_square=pinner_square,
                                pinner_piece=pinner_piece,
                                king_square=king_square,
                                valuable_square=None,
                            )
                        )
                        break  # Only one pinner per pin

        # Phase 2: Detect relative pins to valuable pieces
        potential_pinners = (
            self.board.pieces(chess.BISHOP, opponent_color)
            | self.board.pieces(chess.ROOK, opponent_color)
            | self.board.pieces(chess.QUEEN, opponent_color)
        )

        for pinner_square in potential_pinners:
            pinner_piece = self.board.piece_at(pinner_square)
            assert pinner_piece is not None

            direct_attacks = self.board.attacks(pinner_square)

            for attacked_square in direct_attacks & friendly_pieces:
                # Skip if already pinned to king
                if attacked_square in pinned_to_king:
                    continue

                attacked_piece = self.board.piece_at(attacked_square)
                assert attacked_piece is not None

                # Calculate X-ray attacks through this piece
                temp_occupied = chess.SquareSet(self.board.occupied & ~chess.BB_SQUARES[attacked_square])
                xray_attacks = self._get_slider_attacks(pinner_square, pinner_piece.piece_type, temp_occupied)

                # Check for valuable piece behind
                ray = chess.SquareSet.ray(pinner_square, attacked_square)
                if ray:
                    for target_square in (xray_attacks & ray) - direct_attacks:
                        target_piece = self.board.piece_at(target_square)

                        if (
                            target_piece
                            and target_piece.color == color
                            and self._is_valuable_enough(target_piece, attacked_piece)
                        ):
                            pins.append(
                                PinInfo(
                                    pinned_square=attacked_square,
                                    pinned_piece=attacked_piece,
                                    pinner_square=pinner_square,
                                    pinner_piece=pinner_piece,
                                    king_square=None,
                                    valuable_square=target_square,
                                )
                            )
                            break  # Only first piece behind counts

        return pins

    def _is_more_valuable(self, valuable_piece: chess.Piece, pinned_piece: chess.Piece) -> bool:
        """Check if one piece is more valuable than another.

        Used to determine if a relative pin is tactically significant.
        """
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 100,
        }

        return piece_values.get(valuable_piece.piece_type, 0) > piece_values.get(pinned_piece.piece_type, 0)

    def _is_valuable_enough(self, valuable_piece: chess.Piece, pinned_piece: chess.Piece) -> bool:
        """Check if a piece behind is valuable enough to warrant flagging the pin.

        For relative pins, we flag pins when the piece behind is at least as valuable
        as the pinned piece (>=), not just more valuable (>).
        """
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 100,
        }

        return piece_values.get(valuable_piece.piece_type, 0) >= piece_values.get(pinned_piece.piece_type, 0)

    def _get_slider_attacks(
        self, square: chess.Square, piece_type: chess.PieceType, occupied: chess.SquareSet
    ) -> chess.SquareSet:
        """Calculate sliding piece attacks with custom occupancy for X-ray detection.

        Creates a temporary board with modified occupancy to calculate X-ray attacks.
        """
        # Create a temporary board to calculate attacks with modified occupancy
        temp_board = self.board.copy()

        # Remove all pieces except the ones in the occupied set and the sliding piece itself
        for sq in chess.SQUARES:
            if sq not in occupied and sq != square:
                temp_board.remove_piece_at(sq)

        # Get attacks from this square
        return temp_board.attacks(square)

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
