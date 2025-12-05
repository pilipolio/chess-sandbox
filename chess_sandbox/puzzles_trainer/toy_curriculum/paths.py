"""BFS shortest path algorithms for chess pieces."""

from collections import deque

import chess


def knight_shortest_path(start: chess.Square, target: chess.Square) -> list[str]:
    """Find shortest knight path using BFS. Returns UCI moves.

    >>> knight_shortest_path(chess.A1, chess.B3)
    ['a1b3']

    >>> knight_shortest_path(chess.A1, chess.C2)
    ['a1c2']

    >>> len(knight_shortest_path(chess.A1, chess.H8))
    6

    >>> knight_shortest_path(chess.E4, chess.E4)
    []
    """
    if start == target:
        return []

    knight_offsets = [
        (-2, -1),
        (-2, 1),
        (-1, -2),
        (-1, 2),
        (1, -2),
        (1, 2),
        (2, -1),
        (2, 1),
    ]

    queue: deque[tuple[chess.Square, list[str]]] = deque([(start, [])])
    visited: set[chess.Square] = {start}

    while queue:
        current, path = queue.popleft()

        current_file = chess.square_file(current)
        current_rank = chess.square_rank(current)

        for df, dr in knight_offsets:
            new_file = current_file + df
            new_rank = current_rank + dr

            if 0 <= new_file < 8 and 0 <= new_rank < 8:
                new_square = chess.square(new_file, new_rank)

                if new_square not in visited:
                    move_uci = chess.square_name(current) + chess.square_name(new_square)
                    new_path = path + [move_uci]

                    if new_square == target:
                        return new_path

                    visited.add(new_square)
                    queue.append((new_square, new_path))

    return []  # Should never reach here on a valid board


def rook_shortest_path(start: chess.Square, target: chess.Square) -> list[str]:
    """Find shortest rook path (1-2 moves on empty board).

    >>> rook_shortest_path(chess.A1, chess.A8)
    ['a1a8']

    >>> rook_shortest_path(chess.A1, chess.H1)
    ['a1h1']

    >>> sorted(rook_shortest_path(chess.A1, chess.H8))
    ['a1a8', 'a8h8']

    >>> rook_shortest_path(chess.E4, chess.E4)
    []
    """
    if start == target:
        return []

    start_file = chess.square_file(start)
    start_rank = chess.square_rank(start)
    target_file = chess.square_file(target)
    target_rank = chess.square_rank(target)

    start_name = chess.square_name(start)
    target_name = chess.square_name(target)

    # Same file or rank = 1 move
    if start_file == target_file or start_rank == target_rank:
        return [start_name + target_name]

    # Otherwise 2 moves: go via corner
    intermediate = chess.square(start_file, target_rank)
    intermediate_name = chess.square_name(intermediate)
    return [start_name + intermediate_name, intermediate_name + target_name]


def bishop_shortest_path(start: chess.Square, target: chess.Square) -> list[str] | None:
    """Find shortest bishop path (1-2 moves if reachable).

    Returns None if target is on opposite color complex.

    >>> bishop_shortest_path(chess.A1, chess.H8)
    ['a1h8']

    >>> bishop_shortest_path(chess.A1, chess.A2) is None
    True

    >>> bishop_shortest_path(chess.C1, chess.H6)
    ['c1h6']

    >>> path = bishop_shortest_path(chess.A1, chess.B4)
    >>> len(path) if path else 0
    2

    >>> bishop_shortest_path(chess.E4, chess.E4)
    []
    """
    if start == target:
        return []

    start_file = chess.square_file(start)
    start_rank = chess.square_rank(start)
    target_file = chess.square_file(target)
    target_rank = chess.square_rank(target)

    # Check same color complex
    start_color = (start_file + start_rank) % 2
    target_color = (target_file + target_rank) % 2
    if start_color != target_color:
        return None

    start_name = chess.square_name(start)
    target_name = chess.square_name(target)

    # Same diagonal = 1 move
    if abs(target_file - start_file) == abs(target_rank - start_rank):
        return [start_name + target_name]

    # 2 moves: find intersection of diagonals
    # Diagonal equations: rank - file = c1 (start), rank + file = c2 (target)
    # or rank + file = c1 (start), rank - file = c2 (target)

    # Try both diagonal combinations
    for d1, d2 in [
        (start_rank - start_file, target_rank + target_file),
        (start_rank + start_file, target_rank - target_file),
    ]:
        # Solve: r - f = d1, r + f = d2 => r = (d1+d2)/2, f = (d2-d1)/2
        r_sum = d1 + d2
        f_diff = d2 - d1
        if r_sum % 2 == 0 and f_diff % 2 == 0:
            r = r_sum // 2
            f = f_diff // 2
            if 0 <= r < 8 and 0 <= f < 8:
                intermediate = chess.square(f, r)
                intermediate_name = chess.square_name(intermediate)
                return [start_name + intermediate_name, intermediate_name + target_name]

    return None  # Should not reach here if same color


def queen_shortest_path(start: chess.Square, target: chess.Square) -> list[str]:
    """Find shortest queen path (always 1-2 moves on empty board).

    >>> queen_shortest_path(chess.A1, chess.H8)
    ['a1h8']

    >>> queen_shortest_path(chess.A1, chess.H1)
    ['a1h1']

    >>> queen_shortest_path(chess.E4, chess.E4)
    []

    >>> len(queen_shortest_path(chess.A1, chess.B4))
    2
    """
    if start == target:
        return []

    start_file = chess.square_file(start)
    start_rank = chess.square_rank(start)
    target_file = chess.square_file(target)
    target_rank = chess.square_rank(target)

    start_name = chess.square_name(start)
    target_name = chess.square_name(target)

    # Same rank, file, or diagonal = 1 move
    if start_file == target_file or start_rank == target_rank:
        return [start_name + target_name]
    if abs(target_file - start_file) == abs(target_rank - start_rank):
        return [start_name + target_name]

    # 2 moves: prefer rook-like path (via rank then file)
    intermediate = chess.square(start_file, target_rank)
    intermediate_name = chess.square_name(intermediate)
    return [start_name + intermediate_name, intermediate_name + target_name]


def find_shortest_path(piece_type: chess.PieceType, start: chess.Square, target: chess.Square) -> list[str] | None:
    """Unified shortest path finder for any piece type.

    >>> find_shortest_path(chess.KNIGHT, chess.A1, chess.B3)
    ['a1b3']

    >>> find_shortest_path(chess.ROOK, chess.A1, chess.H8)
    ['a1a8', 'a8h8']

    >>> find_shortest_path(chess.BISHOP, chess.A1, chess.A2) is None
    True

    >>> find_shortest_path(chess.QUEEN, chess.A1, chess.H1)
    ['a1h1']

    >>> find_shortest_path(chess.PAWN, chess.E2, chess.E4) is None
    True
    """
    if piece_type == chess.KNIGHT:
        return knight_shortest_path(start, target)
    elif piece_type == chess.ROOK:
        return rook_shortest_path(start, target)
    elif piece_type == chess.BISHOP:
        return bishop_shortest_path(start, target)
    elif piece_type == chess.QUEEN:
        return queen_shortest_path(start, target)
    else:
        return None  # King and Pawn have complex rules
