"""Lichess study export and PGN validation utilities.

This module provides functions for:
- Validating PGN move sequences against a FEN position
- Building annotated PGN from reasoning traces
- Exporting to Lichess studies via API
"""

import io
from typing import Any

import chess
import chess.pgn
import click
import httpx

from chess_sandbox.config import settings


def validate_pgn_moves(fen: str, pgn_text: str) -> tuple[bool, list[str]]:
    """Validate all moves in PGN are legal from the given FEN.

    Parses moves manually and validates each one against the board position.

    Args:
        fen: Starting position FEN
        pgn_text: PGN movetext (may include variations and comments)

    Returns:
        (is_valid, illegal_moves) - True if all legal, list of illegal move strings found

    >>> validate_pgn_moves("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "1. e4 e5 2. Nf3")
    (True, [])
    >>> validate_pgn_moves("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "1. e4 e5 (1... Nf6) 2. Nf3")
    (True, [])
    >>> valid, illegal = validate_pgn_moves("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "1. e5")
    >>> valid
    False
    >>> "e5" in illegal[0]
    True
    """
    illegal_moves: list[str] = []

    def validate_moves_recursive(pgn: str, board: chess.Board) -> None:
        """Validate moves in PGN text, handling variations recursively."""
        import re

        text = pgn.strip()
        if not text:
            return

        pos = 0
        while pos < len(text):
            char = text[pos]

            if char == "{":
                end = text.find("}", pos)
                if end == -1:
                    break
                pos = end + 1
                continue

            if char == "(":
                depth = 1
                start = pos + 1
                end = start
                while end < len(text) and depth > 0:
                    if text[end] == "(":
                        depth += 1
                    elif text[end] == ")":
                        depth -= 1
                    end += 1
                variation_text = text[start : end - 1]
                board_copy = board.copy()
                if board_copy.move_stack:
                    board_copy.pop()
                validate_moves_recursive(variation_text, board_copy)
                pos = end
                continue

            if char in "0123456789.… \t\n":
                pos += 1
                continue

            move_match = re.match(r"([KQRBNP]?[a-h]?[1-8]?x?[a-h][1-8](=[QRBN])?[+#]?|O-O-O|O-O)", text[pos:])
            if move_match:
                san = move_match.group(0)
                try:
                    move = board.parse_san(san)
                    if move not in board.legal_moves:
                        illegal_moves.append(f"{san} (illegal at {board.fen()})")
                    else:
                        board.push(move)
                except ValueError:
                    illegal_moves.append(f"{san} (invalid at {board.fen()})")
                pos += len(san)
                continue

            pos += 1

    board = chess.Board(fen)
    validate_moves_recursive(pgn_text, board)

    return len(illegal_moves) == 0, illegal_moves


def build_annotated_pgn(
    fen: str,
    themes: list[str],
    position_summary: str,
    candidate_moves_reasoning: str,
    lines_exploration: str,
    source_url: str,
    model_name: str,
) -> tuple[str, list[str]]:
    """Build annotated PGN for Lichess import from reasoning trace data.

    Args:
        fen: Puzzle position FEN
        themes: List of puzzle themes for Event header
        position_summary: Position summary for initial comment
        candidate_moves_reasoning: Candidate analysis for initial comment
        lines_exploration: PGN-formatted lines with variations and comments
        source_url: Puzzle source URL for Site header
        model_name: Model name for Annotator header

    Returns:
        (pgn_string, illegal_moves) - Complete PGN text and list of any illegal moves found

    >>> fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    >>> pgn, illegal = build_annotated_pgn(
    ...     fen=fen,
    ...     themes=["opening"],
    ...     position_summary="White played e4",
    ...     candidate_moves_reasoning="e5 is solid",
    ...     lines_exploration="1... e5 2. Nf3",
    ...     source_url="https://lichess.org/training/test",
    ...     model_name="test-model",
    ... )
    >>> "[Event" in pgn
    True
    >>> "e5" in pgn
    True
    >>> len(illegal)
    0
    """
    is_valid, illegal_moves = validate_pgn_moves(fen, lines_exploration)

    game = chess.pgn.Game()
    game.headers["Event"] = ", ".join(themes) if themes else "Chess Puzzle"
    game.headers["Site"] = source_url
    game.headers["FEN"] = fen
    game.headers["Annotator"] = model_name
    del game.headers["Date"]
    del game.headers["Round"]
    del game.headers["White"]
    del game.headers["Black"]
    del game.headers["Result"]

    header_comment = f"Position Summary: {position_summary}\n\nCandidate Moves: {candidate_moves_reasoning}"
    game.comment = header_comment

    if is_valid:
        pgn_with_fen = f'[FEN "{fen}"]\n\n{lines_exploration}'
        parsed = chess.pgn.read_game(io.StringIO(pgn_with_fen))
        if parsed is not None:
            node = game
            for parsed_node in parsed.mainline():
                node = node.add_variation(parsed_node.move)
                if parsed_node.comment:
                    node.comment = parsed_node.comment
                for var in parsed_node.parent.variations[1:] if parsed_node.parent else []:
                    _copy_variation(node.parent, var) if node.parent else None

    exporter = chess.pgn.StringExporter(headers=True, variations=True, comments=True)
    pgn_string = game.accept(exporter)

    return pgn_string, illegal_moves


def _copy_variation(target_node: chess.pgn.GameNode, source_var: chess.pgn.GameNode) -> None:
    """Recursively copy a variation from source to target node."""
    if source_var.move is None:
        return
    var_node = target_node.add_variation(source_var.move)
    if source_var.comment:
        var_node.comment = source_var.comment
    for child in source_var.variations:
        _copy_variation(var_node, child)


def import_pgn_to_lichess(study_id: str, pgn_content: str) -> dict[str, Any]:
    """Import PGN content to a Lichess study.

    Args:
        study_id: The Lichess study ID to import into
        pgn_content: The PGN content to import

    Returns:
        Response from Lichess API

    Raises:
        ValueError: If LICHESS_API_TOKEN not set
        httpx.HTTPStatusError: If the API request fails
    """
    if not settings.LICHESS_API_TOKEN:
        msg = "LICHESS_API_TOKEN not set in environment"
        raise ValueError(msg)

    url = f"https://lichess.org/api/study/{study_id}/import-pgn"
    headers = {"Authorization": f"Bearer {settings.LICHESS_API_TOKEN}"}
    data = {"pgn": pgn_content}

    response = httpx.post(url, headers=headers, data=data, timeout=30)
    response.raise_for_status()

    return response.json()


def export_traces_to_lichess(
    traces: list[dict[str, Any]],
    study_id: str,
    model_name: str,
) -> dict[str, Any]:
    """Export reasoning traces to a Lichess study.

    Args:
        traces: List of reasoning trace dicts with keys:
            fen, themes, position_summary, candidate_moves_reasoning,
            lines_exploration, source_url
        study_id: Lichess study ID to export to
        model_name: Model name for PGN Annotator header

    Returns:
        Summary dict with exported_count, skipped_count, and API response
    """
    pgn_chapters: list[str] = []
    skipped = 0
    illegal_moves_all: list[tuple[str, list[str]]] = []

    for trace in traces:
        pgn, illegal = build_annotated_pgn(
            fen=trace["fen"],
            themes=trace.get("themes", []),
            position_summary=trace.get("position_summary", ""),
            candidate_moves_reasoning=trace.get("candidate_moves_reasoning", ""),
            lines_exploration=trace.get("lines_exploration", ""),
            source_url=trace.get("source_url", ""),
            model_name=model_name,
        )

        if illegal:
            click.echo(f"Warning: Skipping trace with illegal moves: {illegal[:3]}", err=True)
            illegal_moves_all.append((trace.get("source_url", "unknown"), illegal))
            skipped += 1
        else:
            pgn_chapters.append(pgn)

    if not pgn_chapters:
        return {
            "exported_count": 0,
            "skipped_count": skipped,
            "illegal_moves": illegal_moves_all,
            "response": None,
        }

    combined_pgn = "\n\n".join(pgn_chapters)
    response = import_pgn_to_lichess(study_id, combined_pgn)

    return {
        "exported_count": len(pgn_chapters),
        "skipped_count": skipped,
        "illegal_moves": illegal_moves_all,
        "response": response,
    }
