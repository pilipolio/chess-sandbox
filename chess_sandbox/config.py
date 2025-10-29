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
    LICHESS_API_TOKEN: str = ""


settings = Settings()
