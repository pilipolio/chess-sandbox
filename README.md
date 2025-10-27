# chess-sandbox

Chess commentary generation using LLMs and Stockfish engine analysis.

## Features

### Two Approaches to Chess Commentary

This project demonstrates two distinct approaches to generating natural language chess commentary:

1. **Claude Skill (Interactive)** - Claude Code skill for interactive position analysis with human-in-the-loop commentary
2. **OpenAI Commentator (Automated)** - End-to-end automated commentary pipeline using OpenAI models

Both approaches combine Stockfish engine analysis with LLM reasoning to explain chess positions.

### Additional Capabilities

- **Engine Analysis Foundation** - Stockfish-based position evaluation and principal variation analysis
- **Data Collection Pipeline** - Scrape chess positions with themes from HTML blogs
- **Evaluation System** - Batch evaluation and benchmarking of LLM commentary quality using LLM judges

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

### Approach 1: Claude Skill (Interactive Analysis)

The Claude Code skill provides interactive chess analysis directly in your conversation with Claude.

**Example:**
```
> Analyze this chess position 8/8/2K5/p1p5/P1P5/1k6/8/8 w - - 0 58
```

Claude will automatically use the `chess-analysis` skill to:
1. Run Stockfish analysis on the position
2. Provide natural language commentary on best moves
3. Explain strategic/tactical themes
4. Present key variations with annotations

The skill is located in `.claude/skills/chess-analysis/` and automatically triggers when you provide FEN positions or ask for position analysis.

### Approach 2: OpenAI Commentator (Automated Pipeline)

Run automated commentary generation using OpenAI models:

```bash
# Basic usage - analyze a single position
uv run python -m chess_sandbox.commentator
```

The commentator combines:
- Stockfish engine analysis (configurable depth and lines)
- OpenAI LLM reasoning (supports GPT-4o, GPT-5-mini, reasoning models)
- Structured output with themes, variations, and best moves

**Configuration example:**
```python
params = {
    "engine": {"depth": 20, "num_lines": 5},
    "llm": {"model": "gpt-4o", "reasoning_effort": "low"}
}
```

### Engine Analysis (Foundation)

Both commentary approaches build on the engine analysis module:

```bash
# Analyze starting position
export STOCKFISH_PATH=/opt/homebrew/bin/stockfish
uv run python -m chess_sandbox.engine_analysis "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# Analyze position after a specific move
uv run python -m chess_sandbox.engine_analysis "8/8/2K5/p1p5/P1P5/1k6/8/8 w - - 0 58" --next-move Kb5

# Custom depth and number of lines
uv run python -m chess_sandbox.engine_analysis "<FEN>" --depth 25 --num-lines 3
```

## Advanced Usage

### Data Collection

Scrape chess positions with ground truth themes from chess blog HTML:

```bash
# Process HTML files from data/raw/ and output to data/processed/
uv run python -m chess_sandbox.data_scraper
```

This extracts:
- Chess positions in FEN format
- Associated strategic/tactical themes
- Outputs structured JSONL data for evaluation

### Batch Evaluation

Evaluate commentary quality using LLM judges:

```bash
uv run python -m chess_sandbox.evaluation
```

The evaluation pipeline:
1. Loads ground truth positions and themes
2. Generates commentary using different configurations
3. Uses GPT-4o-mini as a judge to score theme predictions (0-100)
4. Produces evaluation reports with average scores and rationale

**Evaluation configurations** can compare:
- Different LLM models (GPT-4o vs GPT-5-mini)
- Reasoning effort levels (low, medium, high)
- With/without engine analysis input
- Different engine depths and line counts

Results are saved to `data/results/` as JSONL files.

## Project Structure

```
chess_sandbox/
├── engine_analysis.py    # Stockfish analysis foundation
├── commentator.py        # OpenAI-based automated commentary
├── data_scraper.py       # HTML scraping for ground truth data
├── evaluation.py         # Batch evaluation with LLM judges
└── config.py            # Settings management

.claude/skills/chess-analysis/  # Claude Code skill for interactive analysis
```

## Tech Stack

- **Chess Engine:** [Stockfish](https://stockfishchess.org/)
- **LLM Providers:** OpenAI (GPT-4o, GPT-5-mini), Claude (via Claude Code)
- **Chess Library:** [python-chess](https://python-chess.readthedocs.io/)
- **Type Safety:** Pydantic models with strict type checking (pyright)
- **Code Quality:** ruff (formatting + linting), pytest (testing)
- **Package Management:** uv

Project scaffolding templated from [postmodern-python](https://github.com/carderne/postmodern-python)

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
docker run --rm chess-sandbox \
  /app/.venv/bin/python -m pytest /app/chess_sandbox/engine_analysis.py -v
```

## CI/CD

GitHub Actions workflows:
- **PR Checks** ([.github/workflows/pr.yml](.github/workflows/pr.yml)): Formatting, linting, type checking, and tests

## License

MIT
