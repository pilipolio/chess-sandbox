# chess-sandbox

Chess commentary generation using LLMs and Stockfish engine analysis.

## Features

1. **Chess Commentator Skill** - A Claude [Skill](https://www.anthropic.com/news/skills) providing Stockfish chess engine as a tool for interactive position analysis with human-in-the-loop commentary
2. **Chess Commentator LLM Workflow** - Commentary generation pipeline using OpenAI models

## Quick Start

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) for dependency management
- [Stockfish](https://stockfishchess.org/download/) chess engine
- OpenAI API key (for automated commentator and evaluation)

### Setup

1. Install dependencies:
```bash
uv sync
```

2. Set environment variables:
```bash
cp .env.example .env

# Edit .env and set:
# STOCKFISH_PATH=/opt/homebrew/bin/stockfish  # macOS Homebrew
# OPENAI_API_KEY=your-api-key-here           # For OpenAI features
```

## Usage

### Approach 1: Claude Skill

The Claude Code skill provides interactive chess analysis directly in your conversation with Claude Code or Claude.AI

**Example:**
```
> Analyze this chess position 8/8/2K5/p1p5/P1P5/1k6/8/8 w - - 0 58
```

Claude will automatically use the `chess-commentator` skill to:
1. Run Stockfish analysis on the position
2. Provide natural language commentary on best moves
3. Explain strategic/tactical themes
4. Present key variations with annotations

The skill is located in `.claude/skills/chess-commentator/` and automatically triggers when you provide FEN positions or ask for position analysis.

### Approach 2: LLM Workflow

Run automated commentary generation using OpenAI models:

```bash
# Basic usage - analyze a single position
uv run python -m chess_sandbox.commentator
```

Batch evaluation using LLM as judges:

```bash
uv run python -m chess_sandbox.evaluation
```

## Project Structure

```
chess_sandbox/
├── engine_analysis.py    # Stockfish analysis foundation
├── commentator.py        # OpenAI-based automated commentary
├── data_scraper.py       # HTML scraping for ground truth data
├── evaluation.py         # Batch evaluation with LLM judges
└── config.py            # Settings management

.claude/skills/chess-commentator/  # Chess Commentator skill for interactive analysis
```

## Tech Stack

Project scaffolding templated from [postmodern-python](https://github.com/carderne/postmodern-python)

- **Package Management:** uv
- **Type Safety:** Pydantic models with strict type checking (pyright)
- **Code Quality:** ruff (formatting + linting), pytest (testing)
- **Chess Engine:** [Stockfish](https://stockfishchess.org/)
- **LLM Providers:** OpenAI (GPT-4o, GPT-5-mini), Claude (via Claude Code)
- **Chess Library:** [python-chess](https://python-chess.readthedocs.io/)

## Development

### Pre-commit Checks

Run all checks before committing:
```bash
uv run poe all
```

Individual commands:
```bash
uv run poe fmt     # ruff format
uv run poe lint    # ruff check --fix
uv run poe check   # pyright type checking
uv run poe test    # pytest (unit tests)
```

See `CLAUDE.md` for AI agent instructions and `pyproject.toml` for tool configurations.

## Docker

Build and run using Docker:

```bash
docker build -t chess-sandbox .

# Run engine analysis
docker run --rm chess-sandbox \
  /app/.venv/bin/python -m chess_sandbox.engine_analysis \
  "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# Run tests
docker run --rm chess-sandbox:test /app/.venv/bin/python -m pytest -m integration -v
```

## CI/CD

GitHub Actions workflows:
- **PR Checks** ([.github/workflows/pr.yml](.github/workflows/pr.yml)): Formatting, linting, type checking, and unit/integration tests

## License

MIT
