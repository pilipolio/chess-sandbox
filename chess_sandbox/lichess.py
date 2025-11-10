def get_analysis_url(fen: str, color: str = "white") -> str:
    """Generate a Lichess analysis link for a given FEN position."""
    fen_encoded = fen.replace(" ", "_")
    return f"https://lichess.org/analysis/{fen_encoded}?color={color}"
