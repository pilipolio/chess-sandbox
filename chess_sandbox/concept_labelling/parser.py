"""PGN parsing and position extraction for concept labeling."""

from pathlib import Path

import chess.pgn

from .models import LabelledPosition


def parse_pgn_file(pgn_path: Path) -> list[chess.pgn.Game]:
    """Parse a PGN file and return all games.

    >>> import tempfile
    >>> pgn_content = '''[Event "Test"]
    ... [Site "Test Site"]
    ... [Date "2024.01.01"]
    ... [Round "1"]
    ... [White "Player1"]
    ... [Black "Player2"]
    ... [Result "1-0"]
    ...
    ... 1. e4 e5 { Starting position } 2. Nf3 Nc6 1-0
    ... '''
    >>> with tempfile.NamedTemporaryFile(mode='w', suffix='.pgn', delete=False) as f:
    ...     _ = f.write(pgn_content)
    ...     temp_path = Path(f.name)
    >>> games = parse_pgn_file(temp_path)
    >>> len(games)
    1
    >>> games[0].headers['Event']
    'Test'
    >>> temp_path.unlink()  # cleanup
    """
    games: list[chess.pgn.Game] = []
    with pgn_path.open() as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            games.append(game)
    return games


def extract_positions(game: chess.pgn.Game, game_id: str) -> list[LabelledPosition]:
    """Extract all annotated positions from a game.

    >>> import io
    >>> pgn_content = '''[Event "Test"]
    ... [White "Player1"]
    ... [Black "Player2"]
    ...
    ... 1. e4 e5 { French defense } 2. Nf3 { Developing } Nc6 1-0
    ... '''
    >>> game = chess.pgn.read_game(io.StringIO(pgn_content))
    >>> positions = extract_positions(game, "test_game_1")  # type: ignore
    >>> len(positions)
    2
    >>> positions[0].comment
    'French defense'
    >>> positions[0].move_number
    2
    >>> positions[1].comment
    'Developing'
    >>> positions[1].move_number
    2
    """
    positions: list[LabelledPosition] = []
    board = game.board()

    for node in game.mainline():
        board.push(node.move)
        comment = node.comment.strip()

        if comment:
            position = LabelledPosition(
                fen=board.fen(),
                move_number=board.fullmove_number,
                side_to_move="white" if board.turn == chess.WHITE else "black",
                comment=comment,
                game_id=game_id,
                concepts=[],
            )
            positions.append(position)

    return positions


def parse_pgn_directory(input_dir: Path, limit: int | None = None) -> list[LabelledPosition]:
    """Parse all PGN files in a directory and extract positions.

    >>> import tempfile
    >>> temp_dir = Path(tempfile.mkdtemp())
    >>> pgn1 = temp_dir / "game1.pgn"
    >>> _ = pgn1.write_text('''[Event "Game1"]\\n\\n1. e4 e5 { Opening } 1-0\\n''')
    >>> pgn2 = temp_dir / "game2.pgn"
    >>> _ = pgn2.write_text('''[Event "Game2"]\\n\\n1. d4 d5 { Queen pawn } 1-0\\n''')
    >>> positions = parse_pgn_directory(temp_dir, limit=1)
    >>> len(positions)
    1
    >>> positions = parse_pgn_directory(temp_dir)
    >>> len(positions)
    2
    >>> import shutil
    >>> shutil.rmtree(temp_dir)
    """
    all_positions: list[LabelledPosition] = []
    pgn_files = sorted(input_dir.glob("*.pgn"))

    if limit:
        pgn_files = pgn_files[:limit]

    for pgn_path in pgn_files:
        game_id = pgn_path.stem
        games = parse_pgn_file(pgn_path)

        for game in games:
            positions = extract_positions(game, game_id)
            all_positions.extend(positions)

    return all_positions
