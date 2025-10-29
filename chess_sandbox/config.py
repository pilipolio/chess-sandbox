"""
Configuration module for chess_sandbox.
"""

from dotenv import load_dotenv
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    STOCKFISH_PATH: str = "/usr/local/bin/stockfish"
    LC0_PATH: str = "/opt/homebrew/bin/lc0"
    MAIA_WEIGHTS_PATH: str = "data/raw/maia-1500.pb.gz"
    LICHESS_API_TOKEN: str = ""


load_dotenv()

settings = Settings()
