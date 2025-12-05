"""Piece list format parsing and conversion."""

import re
from dataclasses import dataclass

import chess


@dataclass(frozen=True)
class PieceSpec:
    """Specification for a piece on the board."""

    symbol: str  # 'N', 'R', 'B', 'Q', 'K', 'P' (uppercase)
    square: chess.Square
    color: chess.Color


def parse_piece_spec(spec: str) -> PieceSpec:
    """Parse a single piece spec like 'Ne4' or 'pd5'.

    Uppercase = White, lowercase = Black.

    >>> parse_piece_spec('Ne4')
    PieceSpec(symbol='N', square=28, color=True)
    >>> parse_piece_spec('pd5')
    PieceSpec(symbol='P', square=35, color=False)
    >>> parse_piece_spec('Ke1')
    PieceSpec(symbol='K', square=4, color=True)
    """
    spec = spec.strip()
    if len(spec) < 3:
        raise ValueError(f"Invalid piece spec: {spec}")

    piece_char = spec[0]
    square_str = spec[1:3].lower()

    try:
        square = chess.parse_square(square_str)
    except ValueError as e:
        raise ValueError(f"Invalid square in spec '{spec}': {square_str}") from e

    color = chess.WHITE if piece_char.isupper() else chess.BLACK
    symbol = piece_char.upper()

    if symbol not in "KQRBNP":
        raise ValueError(f"Invalid piece symbol: {piece_char}")

    return PieceSpec(symbol=symbol, square=square, color=color)


def parse_piece_list(spec: str) -> list[PieceSpec]:
    """Parse piece list string into PieceSpec list.

    Supports formats:
    - Simple: "Ne4, pd5, Qh8"
    - Explicit: "White: Ne4, Qh8 | Black: pd5, ke8"

    >>> specs = parse_piece_list("Ne4, pd5")
    >>> [(s.symbol, chess.square_name(s.square), s.color) for s in specs]
    [('N', 'e4', True), ('P', 'd5', False)]

    >>> specs = parse_piece_list("White: Ne4 | Black: pd5")
    >>> [(s.symbol, chess.square_name(s.square), s.color) for s in specs]
    [('N', 'e4', True), ('P', 'd5', False)]
    """
    spec = spec.strip()
    pieces: list[PieceSpec] = []

    if "|" in spec:
        parts = spec.split("|")
        for part in parts:
            part = part.strip()
            color_match = re.match(r"(White|Black):\s*(.*)", part, re.IGNORECASE)
            if color_match:
                color_str, piece_specs = color_match.groups()
                is_white = color_str.lower() == "white"
                if piece_specs.strip():
                    for p in piece_specs.split(","):
                        p = p.strip()
                        if not p:
                            continue
                        ps = parse_piece_spec(p)
                        if is_white and not ps.color:
                            ps = PieceSpec(ps.symbol, ps.square, chess.WHITE)
                        elif not is_white and ps.color:
                            ps = PieceSpec(ps.symbol, ps.square, chess.BLACK)
                        pieces.append(ps)
            else:
                for p in part.split(","):
                    p = p.strip()
                    if p:
                        pieces.append(parse_piece_spec(p))
    else:
        for p in spec.split(","):
            p = p.strip()
            if p:
                pieces.append(parse_piece_spec(p))

    return pieces


def piece_list_to_fen(pieces: list[PieceSpec], turn: chess.Color = chess.WHITE) -> str:
    """Convert piece list to FEN string.

    >>> specs = [PieceSpec('N', chess.E4, chess.WHITE), PieceSpec('P', chess.D5, chess.BLACK)]
    >>> piece_list_to_fen(specs)
    '8/8/8/3p4/4N3/8/8/8 w - - 0 1'

    >>> specs = parse_piece_list("Rc3, pb4, pa5")
    >>> piece_list_to_fen(specs)
    '8/8/8/p7/1p6/2R5/8/8 w - - 0 1'
    """
    board = chess.Board.empty()
    board.turn = turn

    for ps in pieces:
        piece_type = chess.PIECE_SYMBOLS.index(ps.symbol.lower())
        piece = chess.Piece(piece_type, ps.color)
        board.set_piece_at(ps.square, piece)

    return board.fen()


def fen_to_piece_list(fen: str) -> str:
    """Convert FEN to piece list string.

    >>> fen_to_piece_list('8/8/8/3p4/4N3/8/8/8 w - - 0 1')
    'White: Ne4 | Black: pd5'

    >>> fen_to_piece_list('8/8/8/p7/1p6/2R5/8/8 w - - 0 1')
    'White: Rc3 | Black: pa5, pb4'

    >>> fen_to_piece_list('8/8/8/8/8/8/8/8 w - - 0 1')
    'White:  | Black: '
    """
    board = chess.Board(fen)

    white_pieces: list[str] = []
    black_pieces: list[str] = []

    for square, piece in board.piece_map().items():
        square_name = chess.square_name(square)
        if piece.piece_type == chess.PAWN:
            piece_str = "P" + square_name if piece.color else "p" + square_name
        else:
            symbol = piece.symbol().upper()
            piece_str = symbol + square_name if piece.color else symbol.lower() + square_name

        if piece.color == chess.WHITE:
            white_pieces.append(piece_str)
        else:
            black_pieces.append(piece_str)

    piece_order = {"K": 0, "Q": 1, "R": 2, "B": 3, "N": 4, "P": 5}
    white_pieces.sort(key=lambda p: (piece_order.get(p[0].upper(), 6), p[1:]))
    black_pieces.sort(key=lambda p: (piece_order.get(p[0].upper(), 6), p[1:]))

    white_str = ", ".join(white_pieces) if white_pieces else ""
    black_str = ", ".join(black_pieces) if black_pieces else ""
    return f"White: {white_str} | Black: {black_str}"


def fen_to_editor_url(fen: str) -> str:
    """Convert FEN to Lichess editor URL.

    >>> fen_to_editor_url("8/8/2K3k1/8/1P1p4/8/8/8 w - - 0 58")
    'https://lichess.org/editor/8/8/2K3k1/8/1P1p4/8/8/8_w_-_-_0_58'
    """
    return f"https://lichess.org/editor/{fen.replace(' ', '_')}"
