# chess-sandbox

Experimenting with chess engines and LLMs.

## Features

- **Chess Engine Analysis**: Analyze chess positions using Stockfish

```
 ▐▛███▜▌   Claude Code v2.0.27
▝▜█████▛▘  Sonnet 4.5 · Claude Max
  ▘▘ ▝▝    /Users/guillaumeallain/D/chess-sandbox

> Analyze this chess position 8/8/2K5/p1p5/P1P5/1k6/8/8 w - -
0 58

⏺ I'll analyze this chess position for you using the
  chess-analysis skill.

> The "chess-analysis" skill is running

⏺ I'll analyze this endgame position for you.

⏺ Bash(uv run python -m chess_sandbox.engine_analysis
      "8/8/2K5/p1p5/P1P5/1k6/8/8 w - - 0 58")
  ⎿  POSITION:
     . . . . . . . .
     . . . . . . . .
     … +31 lines (ctrl+o to expand)

⏺ Position Analysis

  . . . . . . . .
  . . . . . . . .
  . . K . . . . .
  p . p . . . . .
  P . P . . . . .
  . k . . . . . .
  . . . . . . . .
  . . . . . . . .

  Position Assessment: This is a critical king and pawn
  endgame. White has one winning move (Kb5), but the position
  is a dead draw with any other move. The evaluation swings
  dramatically from +13.35 (winning) to 0.00 (draw) or even
  -13.35 (losing) depending on White's choice.

  Best Move: Kb5!

  ...
  ```

## Tech stack

 * Project scaffolding templated from https://github.com/carderne/postmodern-python

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
