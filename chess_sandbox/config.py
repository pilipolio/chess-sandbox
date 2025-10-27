"""
Configuration module for chess_sandbox.
"""

from dotenv import load_dotenv
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    STOCKFISH_PATH: str = "/usr/local/bin/stockfish"


load_dotenv()

settings = Settings()
