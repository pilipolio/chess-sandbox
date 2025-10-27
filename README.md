# chess-sandbox

Experimenting with chess engines and LLMs. This project provides tools for analyzing chess positions using the Stockfish engine, with plans to integrate LLM-based analysis and commentary.

## Features

- **Chess Engine Analysis**: Analyze chess positions using Stockfish
- **Configurable Engine Path**: Works across different platforms (macOS, Linux, Docker)
- **Docker Support**: Self-contained Docker image with Stockfish pre-compiled
- **CLI Tool**: Analyze positions from the command line

## Quick Start

### Local Development

#### Prerequisites
- Python 3.13+
- [uv](https://docs.astral.sh/uv/) for dependency management
- Stockfish chess engine

#### Install Stockfish

**macOS (Homebrew):**
```bash
brew install stockfish
```

**Linux (apt):**
```bash
sudo apt-get install stockfish
```

**Other platforms:** Download from [official-stockfish.github.io](https://official-stockfish.github.io/)

#### Setup

1. Clone the repository:
```bash
git clone https://github.com/pilipolio/chess-sandbox.git
cd chess-sandbox
```

2. Install dependencies:
```bash
uv sync
```

3. Configure Stockfish path:
```bash
# Copy the example file
cp .env.example .env

# Edit .env and set your Stockfish path
# For macOS Homebrew: STOCKFISH_PATH=/opt/homebrew/bin/stockfish
# For Linux: STOCKFISH_PATH=/usr/bin/stockfish
```

4. Run tests to verify setup:
```bash
export STOCKFISH_PATH=/opt/homebrew/bin/stockfish  # or your path
uv run poe test
```

#### Usage

Analyze a chess position:
```bash
export STOCKFISH_PATH=/opt/homebrew/bin/stockfish

# Analyze starting position
uv run python -m chess_sandbox.engine_analysis "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# Analyze a specific position with custom depth
uv run python -m chess_sandbox.engine_analysis "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3" --depth 25 --num-lines 3
```

### Docker

The Docker image includes a pre-compiled Stockfish binary optimized for the host architecture.

#### Build and Run

```bash
# Build the image
docker build -t chess-sandbox .

# Run analysis (STOCKFISH_PATH is already set in the container)
docker run --rm chess-sandbox \
  /app/.venv/bin/python -m chess_sandbox.engine_analysis \
  "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# Run tests
docker run --rm chess-sandbox \
  /app/.venv/bin/python -m pytest /app/chess_sandbox/engine_analysis.py -v
```

## Development

### Running Checks

```bash
uv run poe fmt      # Format code with ruff
uv run poe lint     # Lint code with ruff
uv run poe check    # Type check with pyright
uv run poe test     # Run tests with pytest
uv run poe all      # Run all checks sequentially
```

### Project Structure

```
chess-sandbox/
├── chess_sandbox/
│   ├── engine_analysis.py    # Stockfish integration and analysis
│   ├── server.py              # Placeholder server entrypoint
│   └── ...
├── Dockerfile                 # Multi-stage build with Stockfish
├── pyproject.toml             # Project configuration
└── .env.example               # Example environment configuration
```

## Configuration

The project uses environment variables for configuration:

- `STOCKFISH_PATH`: Path to the Stockfish binary (required)

See `.env.example` for platform-specific examples.

## CI/CD

The project uses GitHub Actions for:
- **PR Checks** ([.github/workflows/pr.yml](.github/workflows/pr.yml)): Formatting, linting, type checking, and tests
- **Docker Integration Tests**: Validates Stockfish works correctly in containerized environment

## Docker Build Optimization

The Dockerfile uses a 3-stage build process optimized for caching:

1. **Stage 1 (`stockfish-builder`)**: Compiles Stockfish (rarely changes, cached longer)
2. **Stage 2 (`python-deps`)**: Installs Python dependencies (changes more frequently)
3. **Stage 3 (`runner`)**: Final runtime image

This architecture provides ~2-5 minute build time savings when only Python dependencies change.

## License

MIT

## Contributing

Pull requests are welcome! Please ensure all checks pass:
```bash
uv run poe all
```
