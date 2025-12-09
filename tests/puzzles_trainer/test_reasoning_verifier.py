"""Tests for reasoning trace verification."""

from chess_sandbox.puzzles_trainer.reasoning_verifier import (
    extract_final_move,
    extract_pgn_moves,
    extract_solution_section,
    normalize_move,
    parse_sections,
    validate_move_sequence,
    verify_reasoning_trace,
)


class TestParseSections:
    """Tests for section parsing."""

    def test_all_sections_present(self):
        reasoning = """
        ## Position Analysis
        Material is equal. King safety concerns.

        ## Tactical Assessment
        Back rank weakness.

        ## Solution
        19. Nxh7! Kxh7 20. Qh7#
        """
        sections = parse_sections(reasoning)
        assert sections["position_analysis"] is True
        assert sections["tactical_assessment"] is True
        assert sections["solution"] is True

    def test_missing_sections(self):
        reasoning = """
        ## Position Analysis
        Material is equal.

        The solution is Nxh7.
        """
        sections = parse_sections(reasoning)
        assert sections["position_analysis"] is True
        assert sections["tactical_assessment"] is False
        assert sections["solution"] is False

    def test_case_insensitive(self):
        reasoning = """
        ## POSITION ANALYSIS
        Test content.

        ## tactical assessment
        More content.

        ## SOLUTION
        Moves here.
        """
        sections = parse_sections(reasoning)
        assert all(sections.values())


class TestExtractSolutionSection:
    """Tests for solution section extraction."""

    def test_extract_between_headers(self):
        reasoning = """
        ## Tactical Assessment
        Some content.

        ## Solution
        19. Nxh7! {sacrifice} Kxh7 20. Qh7#

        ## Another Section
        More stuff.
        """
        solution = extract_solution_section(reasoning)
        assert solution is not None
        assert "19. Nxh7!" in solution
        assert "Another Section" not in solution

    def test_extract_before_think_close(self):
        reasoning = """
        ## Solution
        19. Nxh7! Kxh7 20. Qh7#
        </think>
        Nxh7
        """
        solution = extract_solution_section(reasoning)
        assert solution is not None
        assert "Nxh7!" in solution
        assert "</think>" not in solution

    def test_no_solution_section(self):
        reasoning = "Just some text without proper sections."
        assert extract_solution_section(reasoning) is None


class TestExtractPgnMoves:
    """Tests for PGN move extraction."""

    def test_simple_moves(self):
        text = "19. Nxh7 19...Kxh7 20. Qh7#"
        moves = extract_pgn_moves(text)
        assert len(moves) == 3
        assert (19, True, "Nxh7") in moves
        assert (19, False, "Kxh7") in moves
        assert (20, True, "Qh7#") in moves  # Includes mate symbol

    def test_moves_with_annotations(self):
        text = "19. Nxh7! {sacrifice} 19...Kxh7?? 20. Qh7#"
        moves = extract_pgn_moves(text)
        assert len(moves) == 3
        assert (19, True, "Nxh7") in moves

    def test_castling(self):
        text = "10. O-O Nf6 11. O-O-O"
        moves = extract_pgn_moves(text)
        castle_moves = [(m, w, s) for m, w, s in moves if "O-O" in s]
        assert len(castle_moves) == 2

    def test_promotion(self):
        text = "40. e8=Q+ 40...Kf7"
        moves = extract_pgn_moves(text)
        assert any(s == "e8=Q+" for _, _, s in moves)


class TestValidateMoveSequence:
    """Tests for move sequence validation."""

    def test_valid_opening_sequence(self):
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        moves = [(1, True, "e4"), (1, False, "e5"), (2, True, "Nf3")]
        valid, illegal = validate_move_sequence(fen, moves)
        assert valid == ["e4", "e5", "Nf3"]
        assert illegal == []

    def test_illegal_move_detected(self):
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        moves = [(1, True, "e4"), (1, False, "Nf3")]  # Black can't play Nf3
        valid, illegal = validate_move_sequence(fen, moves)
        assert "e4" in valid
        assert "Nf3" in illegal

    def test_unparseable_move(self):
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        moves = [(1, True, "Xyz123")]
        _valid, illegal = validate_move_sequence(fen, moves)
        assert "Xyz123" in illegal


class TestExtractFinalMove:
    """Tests for final move extraction."""

    def test_after_think_tag(self):
        reasoning = """
        <think>
        Some analysis here.
        </think>
        Nxh7
        """
        assert extract_final_move(reasoning) == "Nxh7"

    def test_with_check_symbol(self):
        reasoning = """
        </think>
        Qh7+
        """
        assert extract_final_move(reasoning) == "Qh7+"

    def test_castling_move(self):
        reasoning = """
        </think>
        O-O-O
        """
        assert extract_final_move(reasoning) == "O-O-O"

    def test_no_think_tag_fallback(self):
        reasoning = """
        The best move is
        Nxe5
        """
        assert extract_final_move(reasoning) == "Nxe5"


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
        ## Position Analysis
        White has played e4, opening the center.

        ## Tactical Assessment
        Standard opening position. e5 is a solid response.

        ## Solution
        1...e5 {mirroring White's center control}
        </think>
        e5
        """
        result = verify_reasoning_trace(fen, reasoning, ["e5"])

        assert result.first_move_correct is True
        assert all(result.sections_found.values())
        assert len(result.illegal_moves) == 0
        assert result.score >= 0.9

    def test_missing_sections_lowers_score(self):
        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        reasoning = """
        <think>
        The best move is e5.
        </think>
        e5
        """
        result = verify_reasoning_trace(fen, reasoning, ["e5"])

        assert result.first_move_correct is True
        assert not all(result.sections_found.values())
        assert result.score <= 0.7  # Penalized for missing sections

    def test_wrong_first_move(self):
        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        reasoning = """
        <think>
        ## Position Analysis
        Test.

        ## Tactical Assessment
        Test.

        ## Solution
        1...d5 {Scandinavian}
        </think>
        d5
        """
        result = verify_reasoning_trace(fen, reasoning, ["e5"])

        assert result.first_move_correct is False
        assert result.extracted_first_move == "d5"
        assert result.score < 0.8  # Missing first move bonus

    def test_illegal_moves_in_analysis(self):
        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        reasoning = """
        <think>
        ## Position Analysis
        Test.

        ## Tactical Assessment
        Test.

        ## Solution
        1...e5 2. Nf3 2...Nc6 3. Bb5 3...Qa1
        </think>
        e5
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
        ## Position Analysis
        Basic.

        ## Tactical Assessment
        Basic.

        ## Solution
        1...e5
        </think>
        e5
        """
        result = verify_reasoning_trace(fen, reasoning, ["e5"])
        assert result.score >= 0.6
