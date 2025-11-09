"""
Configuration module for chess_sandbox.
"""

from pydantic_settings import BaseSettings

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


class Settings(BaseSettings):
    STOCKFISH_PATH: str = "/usr/local/bin/stockfish"
    LC0_PATH: str = "/opt/homebrew/bin/lc0"
    MAIA_WEIGHTS_PATH: str = "data/raw/maia-1500.pb.gz"

    # Lichess settings
    LICHESS_API_TOKEN: str = ""

    # HuggingFace settings
    HF_TOKEN: str = ""
    HF_CACHE_DIR: str = ""

    # Concept extractor settings
    HF_CONCEPT_EXTRACTOR_REPO_ID: str = "pilipolio/chess-positions-extractor"
    HF_CONCEPT_EXTRACTOR_REVISION: str = "main"

    # Git settings
    GIT_COMMIT: str = ""


settings = Settings()
