# chess-sandbox

Experimenting with chess engines and LLMs.

## Features

- **Chess Engine Analysis**: Analyze chess positions using Stockfish
- **Configurable Engine Path**: Works across different platforms (macOS, Linux, Docker)
- **Docker Support**: Self-contained Docker image with Stockfish pre-compiled
- **CLI Tool**: Analyze positions from the command line

## Tech stack

 * Project scaffolding based off https://github.com/carderne/postmodern-python

## Quick Start

### Local Development

#### Prerequisites
- Python 3.13+
- [uv](https://docs.astral.sh/uv/) for dependency management
- [Stockfish](https://stockfishchess.org/download/) chess engine

#### Setup

1. Install dependencies:
```bash
uv sync
```

2. Set environment variables:
```bash
cp .env.example .env

# Edit .env and set your Stockfish path
# For macOS Homebrew: STOCKFISH_PATH=/opt/homebrew/bin/stockfish
# For Linux: STOCKFISH_PATH=/usr/bin/stockfish
```

3. Analyze a chess position:
```bash
export STOCKFISH_PATH=/opt/homebrew/bin/stockfish

# Analyze starting position
uv run python -m chess_sandbox.engine_analysis "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
```

### Docker

```bash
docker build -t chess-sandbox .

docker run --rm chess-sandbox \
  /app/.venv/bin/python -m chess_sandbox.engine_analysis \
  "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

docker run --rm chess-sandbox \
  /app/.venv/bin/python -m pytest /app/chess_sandbox/engine_analysis.py -v
```

## CI/CD

The project uses GitHub Actions for:
- **PR Checks** ([pr.yml](.github/workflows/pr.yml)): Formatting, linting, type checking, and tests

## License

MIT
