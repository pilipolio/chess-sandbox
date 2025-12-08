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
    LC0_PATH: str = "/opt/homebrew/bin/lc0"  # macOS default; override with LC0_PATH env var
    MAIA_WEIGHTS_PATH: str = "data/maia-1100.pb.gz"  # Docker uses /app/data/maia-1100.pb.gz

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

    # LLM API settings
    OPENROUTER_API_KEY: str = ""


settings = Settings()
