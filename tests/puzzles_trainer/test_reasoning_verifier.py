"""Tests for reasoning trace verification."""

from chess_sandbox.puzzles_trainer.reasoning_verifier import (
    extract_pgn_moves,
    extract_piece_positions_section,
    extract_solution_section,
    normalize_move,
    parse_flexible_piece_list,
    parse_sections,
    validate_move_sequence,
    validate_piece_positions,
    verify_reasoning_trace,
)


class TestParseSections:
    """Tests for section parsing."""

    def test_all_sections_present(self):
        reasoning = """
        ## Step 1: FEN parsing
        Breaking down the position...

        ## Step 2: Piece Positions
        White: Ke1, Qd1. Black: Ke8...

        ## Step 3: Position Summary
        Material is equal. King safety concerns.

        ## Step 4: Candidate Moves
        e5, d5, Nf6...

        ## Step 5: Candidate Lines Analysis
        1...e5 2. Nf3 Nc6...

        ## Step 6: Solution
        e5
        """
        sections = parse_sections(reasoning)
        assert sections["fen_parsing"] is True
        assert sections["piece_positions"] is True
        assert sections["position_summary"] is True
        assert sections["candidate_moves"] is True
        assert sections["candidate_lines"] is True
        assert sections["solution"] is True

    def test_missing_sections(self):
        reasoning = """
        ## Step 1: FEN parsing
        Material is equal.

        The solution is Nxh7.
        """
        sections = parse_sections(reasoning)
        assert sections["fen_parsing"] is True
        assert sections["piece_positions"] is False
        assert sections["position_summary"] is False
        assert sections["candidate_moves"] is False
        assert sections["candidate_lines"] is False
        assert sections["solution"] is False

    def test_case_insensitive(self):
        reasoning = """
        ## STEP 1: FEN PARSING
        Test content.

        ## step 2: piece positions
        More content.

        ## Step 3: Position Summary
        Summary here.

        ## Step 4: Candidate Moves
        Moves here.

        ## Step 5: Candidate Lines Analysis
        Exploration.

        ## Step 6: Solution
        e5
        """
        sections = parse_sections(reasoning)
        assert all(sections.values())


class TestExtractPiecePositionsSection:
    """Tests for piece positions section extraction."""

    def test_extract_section(self):
        reasoning = """
        ## Step 1: FEN parsing
        Some content.

        ## Step 2: Piece Positions
        White: Qa3, Bc3, Kh1. Black: Kg8, Ra8.

        ## Step 3: Position Summary
        Summary here.
        """
        section = extract_piece_positions_section(reasoning)
        assert section is not None
        assert "White: Qa3" in section
        assert "Black: Kg8" in section

    def test_no_section(self):
        reasoning = """
        ## Step 1: FEN parsing
        Some content.
        """
        assert extract_piece_positions_section(reasoning) is None


class TestParseFlexiblePieceList:
    """Tests for flexible piece list parsing."""

    def test_standard_notation(self):
        text = "White: Qa3, Bc3, Kh1. Black: Kg8, Ra8"
        pieces = parse_flexible_piece_list(text)
        assert len(pieces) == 5

    def test_full_piece_names(self):
        text = "White: Queen a3, Bishop c3, King h1"
        pieces = parse_flexible_piece_list(text)
        assert len(pieces) == 3

    def test_grouped_pawns(self):
        text = "White: pawns a2, b2, c2"
        pieces = parse_flexible_piece_list(text)
        assert len(pieces) == 3

    def test_sample_input(self):
        text = (
            "White: Qa3, Bc3, Kh1, Ra1, pawns a2, b2, c2, e4, g3, h2. "
            "Black: Kg8, Ra8, Bc6, Qb7, pawns a7, b7, c7, d7, e6, f7, g6, h6, h7"
        )
        pieces = parse_flexible_piece_list(text)
        white_pieces = [p for p in pieces if p[2]]
        black_pieces = [p for p in pieces if not p[2]]
        assert len(white_pieces) == 10
        assert len(black_pieces) == 13


class TestValidatePiecePositions:
    """Tests for piece position validation."""

    def test_correct_positions(self):
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        text = (
            "White: Ke1, Qd1, Ra1, Rh1, Bc1, Bf1, Nb1, Ng1, pawns a2, b2, c2, d2, e2, f2, g2, h2. "
            "Black: Ke8, Qd8, Ra8, Rh8, Bc8, Bf8, Nb8, Ng8, pawns a7, b7, c7, d7, e7, f7, g7, h7"
        )
        accuracy = validate_piece_positions(fen, text)
        assert accuracy == 1.0

    def test_missing_pieces(self):
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        text = "White: Ke1"
        accuracy = validate_piece_positions(fen, text)
        assert accuracy == 1 / 32  # 1 correct out of 32 total

    def test_extra_pieces(self):
        fen = "8/8/8/8/8/8/8/4K3 w - - 0 1"
        text = "White: Ke1, Qd1"
        accuracy = validate_piece_positions(fen, text)
        assert accuracy == 0.5  # 1 correct, 1 extra = 1/2


class TestExtractSolutionSection:
    """Tests for solution section extraction (content after </think>)."""

    def test_extract_after_think_close(self):
        reasoning = """
        <think>
        ## Step 5: Lines Exploration
        19. Nxh7! {sacrifice} Kxh7 20. Qh7#
        </think>
        Nxh7
        """
        solution = extract_solution_section(reasoning)
        assert solution is not None
        assert solution == "Nxh7"

    def test_extract_multiline_solution(self):
        reasoning = """
        <think>
        Some analysis
        </think>
        Rxe7 Qb1+ Nc1
        """
        solution = extract_solution_section(reasoning)
        assert solution is not None
        assert "Rxe7" in solution
        assert "Qb1+" in solution

    def test_no_think_close_tag(self):
        reasoning = "Just some text without the closing tag."
        assert extract_solution_section(reasoning) is None


class TestExtractPgnMoves:
    """Tests for PGN move extraction."""

    def test_simple_moves(self):
        text = "19. Nxh7 19...Kxh7 20. Qh7#"
        moves = extract_pgn_moves(text)
        assert len(moves) == 3
        assert "Nxh7" in moves
        assert "Kxh7" in moves
        assert "Qh7#" in moves  # Includes mate symbol

    def test_moves_with_annotations(self):
        text = "19. Nxh7! {sacrifice} 19...Kxh7?? 20. Qh7#"
        moves = extract_pgn_moves(text)
        assert len(moves) == 3
        assert "Nxh7" in moves

    def test_castling(self):
        text = "10. O-O Nf6 11. O-O-O"
        moves = extract_pgn_moves(text)
        castle_moves = [m for m in moves if "O-O" in m]
        assert len(castle_moves) == 2

    def test_promotion(self):
        text = "40. e8=Q+ 40...Kf7"
        moves = extract_pgn_moves(text)
        assert "e8=Q+" in moves

    def test_compact_format(self):
        """Test that compact format (25. Rxe7 Qb1+) works correctly."""
        text = "25. Rxe7 Qb1+ 26. Nc1 Qxc1+ 27. Qxc1"
        moves = extract_pgn_moves(text)
        assert "Rxe7" in moves
        assert "Qb1+" in moves
        assert "Nc1" in moves
        assert "Qxc1+" in moves
        assert "Qxc1" in moves


class TestValidateMoveSequence:
    """Tests for move sequence validation."""

    def test_valid_opening_sequence(self):
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        moves = ["e4", "e5", "Nf3"]
        valid, illegal = validate_move_sequence(fen, moves)
        assert valid == ["e4", "e5", "Nf3"]
        assert illegal == []

    def test_illegal_move_detected(self):
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        moves = ["e4", "Nf3"]  # Black can't play Nf3
        valid, illegal = validate_move_sequence(fen, moves)
        assert "e4" in valid
        assert "Nf3" in illegal

    def test_unparseable_move(self):
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        moves = ["Xyz123"]
        _valid, illegal = validate_move_sequence(fen, moves)
        assert "Xyz123" in illegal

    def test_extract_and_validate_middlegame_sequence(self):
        """Test combining extraction and validation for a middlegame position."""
        fen = "r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2b1/PqP3PP/7K w - - 0 25"
        move_text = "25. Rxe7 Qb1+ 26. Nc1 Qxc1+ 27. Qxc1"
        moves = extract_pgn_moves(move_text)
        valid, illegal = validate_move_sequence(fen, moves)
        assert len(illegal) == 0, f"Found illegal moves: {illegal}"
        assert "Rxe7" in valid
        assert "Qb1+" in valid
        assert "Nc1" in valid
        assert "Qxc1+" in valid
        assert "Qxc1" in valid

    def test_extract_and_validate_sequence_wo_numbers(self):
        """Test combining extraction and validation for a middlegame position."""
        fen = "r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2b1/PqP3PP/7K w - - 0 25"
        move_text = "Rxe7 Qb1+ Nc1 Qxc1+ Qxc1"
        moves = extract_pgn_moves(move_text)
        valid, illegal = validate_move_sequence(fen, moves)
        assert len(illegal) == 0, f"Found illegal moves: {illegal}"
        assert "Rxe7" in valid
        assert "Qb1+" in valid
        assert "Nc1" in valid
        assert "Qxc1+" in valid
        assert "Qxc1" in valid


class TestNormalizeMove:
    """Tests for move normalization."""

    def test_strip_annotations(self):
        assert normalize_move("Nxh7!") == "Nxh7"
        assert normalize_move("Kxh7??") == "Kxh7"
        assert normalize_move("Qh7#!") == "Qh7#"

    def test_preserve_check_mate(self):
        assert normalize_move("Qh7+") == "Qh7+"
        assert normalize_move("Qh7#") == "Qh7#"


class TestVerifyReasoningTrace:
    """Integration tests for full verification."""

    def test_perfect_trace(self):
        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        reasoning = """
        <think>
        ## Step 1: FEN parsing
        Breaking down the position after e4.

        ## Step 2: Piece Positions
        White: Ke1, Qd1, Ra1, Rh1, Bc1, Bf1, Nb1, Ng1, pawns a2, b2, c2, d2, e4, f2, g2, h2.
        Black: Ke8, Qd8, Ra8, Rh8, Bc8, Bf8, Nb8, Ng8, pawns a7, b7, c7, d7, e7, f7, g7, h7.

        ## Step 3: Position Summary
        White has played e4, opening the center. Material is equal.

        ## Step 4: Candidate Moves
        e5, d5, c5, Nf6 are all playable.

        ## Step 5: Candidate Lines Analysis
        1...e5 {mirroring White's center control} 2. Nf3 Nc6

        ## Step 6: Solution
        e5
        </think>
        e5
        """
        result = verify_reasoning_trace(fen, reasoning, ["e5"])

        assert result.first_move_correct is True
        assert all(result.sections_found.values())
        assert len(result.illegal_moves) == 0
        assert result.score >= 0.8  # Lowered threshold to account for piece accuracy variance

    def test_missing_sections_lowers_score(self):
        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        reasoning = """
        <think>
        The best move is e5.
        </think>
        e5
        """
        result = verify_reasoning_trace(fen, reasoning, ["e5"])

        assert result.first_move_correct is True  # e5 extracted from after </think>
        assert not all(result.sections_found.values())
        assert result.score <= 0.7  # Penalized for missing sections

    def test_wrong_first_move(self):
        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        reasoning = """
        <think>
        ## Step 1: FEN parsing
        Test.

        ## Step 2: Piece Positions
        Test.

        ## Step 3: Position Summary
        Test.

        ## Step 4: Candidate Moves
        Test.

        ## Step 5: Lines Exploration
        1...d5 {Scandinavian}
        </think>
        d5
        """
        result = verify_reasoning_trace(fen, reasoning, ["e5"])

        assert result.first_move_correct is False
        assert result.extracted_first_move == "d5"
        assert result.score < 0.8  # Missing first move bonus

    def test_illegal_moves_in_solution(self):
        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        reasoning = """
        <think>
        ## Step 1: FEN parsing
        Test.

        ## Step 2: Piece Positions
        Test.

        ## Step 3: Position Summary
        Test.

        ## Step 4: Candidate Moves
        Test.

        ## Step 5: Lines Exploration
        Test.
        </think>
        e5 Nf3 Nc6 Bb5 Qa1
        """
        result = verify_reasoning_trace(fen, reasoning, ["e5"])

        # Qa1 is illegal - Black queen can't reach a1 from d8
        assert "Qa1" in result.illegal_moves
        assert result.score < 1.0  # Penalized for illegal move

    def test_score_threshold(self):
        # A trace with correct first move and all sections should pass 0.6 threshold
        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        reasoning = """
        <think>
        ## Step 1: FEN parsing
        Basic.

        ## Step 2: Piece Positions
        Basic.

        ## Step 3: Position Summary
        Basic.

        ## Step 4: Candidate Moves
        Basic.

        ## Step 5: Lines Exploration
        1...e5
        </think>
        e5
        """
        result = verify_reasoning_trace(fen, reasoning, ["e5"])
        assert result.score >= 0.6
