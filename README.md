# chess-sandbox

Chess commentary generation using LLMs and Stockfish engine analysis.

## Features

1. **Chess Commentator Skill** - A [Skill](https://www.anthropic.com/news/skills) to augment claude.ai or Claude Code with a chess engine tool for interactive position analysis and discussion.
2. **Chess Commentator LLM Workflow** - Commentary generation pipeline using OpenAI models
3. **Concept Labeling Pipeline** - Extract and label chess positions with tactical/strategic concepts from annotated PGN files

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

### Approach 3: Concept Labeling Pipeline

Build a dataset of chess positions labeled with tactical and strategic concepts.

**Download the dataset:**
```bash
# Download annotated PGN files from Hugging Face
wget -P data/raw https://huggingface.co/datasets/Waterhorse/chess_data/resolve/main/chessclip_data/annotated_pgn/annotated_pgn_free.tar.gz

# Extract the archive
tar -xzf data/raw/annotated_pgn_free.tar.gz -C data/raw
```

**Process the dataset:**
```bash
# Parse PGN files and label positions with detected concepts
uv run python -m chess_sandbox.concept_labelling.pipeline \
  --input-dir data/raw/annotated_pgn_free/gameknot \
  --output data/processed/concept_labelling/positions_labeled.jsonl \
  --stats data/processed/concept_labelling/concept_stats.json \
  --limit 5  # Optional: process only first N files

# Export samples to Lichess-compatible PGN format
uv run python -m chess_sandbox.concept_labelling.lichess_export \
  --input data/processed/concept_labelling/positions_labeled.jsonl \
  --output-pgn data/exports/lichess_study_sample.pgn \
  --n-samples 64
```

Detected concepts include tactical themes (pin, fork, skewer, sacrifice) and strategic themes (passed pawn, outpost, weak square, zugzwang). See [docs/plans/concept-labelling-pipeline.md](docs/plans/concept-labelling-pipeline.md) for details.

## Project Structure

```
chess_sandbox/
├── config.py               # Settings management
├── engine                  # Wrapper around stockfish/lc0 engines
    ├── ...
├── commentator.py          # OpenAI-based automated commentary
├── data_scraper.py         # HTML scraping for ground truth data
├── evaluation.py           # Batch evaluation with LLM judges
└── concept_labelling/      # Position labeling pipeline
    ├── ...

docs
├── adrs                    # Architectural Decision Records based on [madr](https://adr.github.io/adr-templates/#markdown-architectural-decision-records-madr) template
└── plans                   # LLM generated/implemented plans, for references

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
- **Modal Based Serverless deployment** See [docs/adrs/20251029-use-modal-for-serverless-endpoints.md](docs/adrs/20251029-use-modal-for-serverless-endpoints.md) for rationale.

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

## Modal API Deployment

Deploy the chess analysis endpoint as a serverless API using Modal:

### Prerequisites

1. Create a Modal account at https://modal.com
2. Generate API token at https://modal.com/settings/tokens
3. Authenticate: `modal token set --token-id <ID> --token-secret <SECRET>`

### Local Testing

```bash
modal serve chess_sandbox/endpoints.py

# Test endpoint
curl "http://localhost:8000/analyze?fen=rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR%20w%20KQkq%20-%200%201&depth=20&num_lines=5"
```

### Production Deployment

```bash
# Manual deployment
modal deploy chess_sandbox/endpoints.py

# Automatic deployment on GitHub releases
# Configured in .github/workflows/release.yml
```

## Integration tests

Build and run using Docker. To replace with Modal at some point

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
- **Release** ([.github/workflows/release.yml](.github/workflows/release.yml)): Automatic Modal deployment on GitHub releases

## License

MIT
