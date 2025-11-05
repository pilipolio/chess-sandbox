# chess-sandbox

Chess commentary generation using LLMs and Stockfish engine analysis.

## Features

1. **Chess Commentator Skill** - A [Skill](https://www.anthropic.com/news/skills) to augment claude.ai or Claude Code with a chess engine tool for interactive position analysis and discussion.
2. **Chess Commentator LLM Workflow** - Commentary generation pipeline using OpenAI models
3. **Concept Extraction Pipeline** - Extract and label chess positions with tactical/strategic concepts from annotated PGN files using regex patterns, LLM validation, and ML-based concept probes

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

### Approach 3: Concept Extraction Pipeline

Build a dataset of chess positions labeled with tactical and strategic concepts using regex detection, LLM validation, and train ML probes.

**Download the dataset:**
```bash
# Download annotated PGN files from Hugging Face
wget -P data/raw https://huggingface.co/datasets/Waterhorse/chess_data/resolve/main/chessclip_data/annotated_pgn/annotated_pgn_free.tar.gz

# Extract the archive
tar -xzf data/raw/annotated_pgn_free.tar.gz -C data/raw
```

**Label positions with regex + LLM:**
```bash
uv run python -m chess_sandbox.concept_extraction.labelling.pipeline \
  --input-dir data/raw/annotated_pgn_free/gameknot \
  --output data/processed/concept_extraction/positions_labeled.jsonl \
  --limit 5  # Optional: process only first N files
  --refine-with-llm  # Optional: validate with LLM

uv run python -m chess_sandbox.concept_extraction.labelling.lichess_export \
  --input data/processed/concept_extraction/positions_labeled.jsonl \
  --study-id YOUR_STUDY_ID \
  --n-samples 64
```

**Train & evaluate ML concept extractor:**
```bash
uv run python -m chess_sandbox.concept_extraction.model.train \
  --dataset-repo-id pilipolio/chess-positions-concepts \
  --lc0-model-repo-id lczerolens/maia-1500 \
  --layer-name block3/conv2/relu \
  --mode multi-label \
  --upload-to-hub --save-splits \
  --output-repo-id pilipolio/chess-positions-extractor \
  --n-jobs 4 --verbose 1
  --output-revision test_fixture

uv run python -m chess_sandbox.concept_extraction.model.evaluation evaluate \
      --model-repo-id pilipolio/chess-positions-extractor \
      --dataset-repo-id pilipolio/chess-positions-concepts \
      --dataset-filename test.jsonl
      --sample-size 10
```

Detected concepts include tactical themes (pin, fork, skewer, sacrifice) and strategic themes (passed pawn, outpost, weak square, zugzwang). See [docs/plans/concept-labelling-pipeline.md](docs/plans/concept-labelling-pipeline.md) for details.

**Running on Modal (serverless):**

See `chess_sandbox/concept_extraction/labelling/modal_pipeline.py` for pre-requisites, then run:

```bash
modal run --detach chess_sandbox/concept_extraction/labelling/modal_pipeline.py::process_pgn_batch \
    --refine-with-llm --llm-model gpt-4.1-mini \
    --output-filename gpt-4.1-mini_labeled_positions_all.jsonl

modal volume get chess-pgn-data outputs/gpt-4.1-mini_labeled_positions_all.jsonl \
  data/processed/concept_extraction/gpt-4.1-mini_labeled_positions_all.jsonl
```

## Project Structure

```
chess_sandbox/
├── config.py                  # Settings management
├── engine/                    # Wrapper around stockfish/lc0 engines
│   ├── ...
├── commentator.py             # OpenAI-based automated commentary
├── data_scraper.py            # HTML scraping for ground truth data
├── evaluation.py              # Batch evaluation with LLM judges
└── concept_extraction/        # Concept extraction pipeline
    ├── labelling/             # Regex + LLM labeling
    │   ├── labeller.py        # Core labeling (includes Concept, LabelledPosition models)
    │   ├── parser.py          # PGN parsing
    │   ├── patterns.py        # Regex patterns
    │   ├── refiner.py         # LLM validation
    │   ├── pipeline.py        # CLI for labeling
    │   └── modal_pipeline.py  # Modal deployment
    └── model/                 # ML-based concept detection
        ├── features.py        # LC0 activation extraction
        ├── train.py           # Training CLI (includes ModelTrainingOutput)
        ├── inference.py       # ConceptProbe, ConceptExtractor
        ├── evaluation.py      # Metrics calculation
        └── hub.py             # HuggingFace Hub upload

docs/
├── adrs/                      # Architectural Decision Records (MADR template)
└── plans/                     # LLM generated/implemented plans

.claude/skills/chess-commentator/  # Chess Commentator skill
```

## Tech Stack

Project scaffolding templated from [postmodern-python](https://github.com/carderne/postmodern-python)

- **Package Management:** uv
- **Type Safety:** Pydantic models with strict type checking (pyright)
- **Code Quality:** ruff (formatting + linting), pytest (testing)
- **Chess Engine:** [Stockfish](https://stockfishchess.org/)
- **LLM Providers:** OpenAI (GPT-4o, GPT-5-mini), Claude (via Claude Code)
- **Chess Library:** [python-chess](https://python-chess.readthedocs.io/)
- **HTTP Client:** httpx (modern async/sync HTTP client)
- **Testing:** respx (HTTP mocking for httpx)
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

curl "http://localhost:8000/analyze?fen=rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR%20w%20KQkq%20-%200%201&depth=20&num_lines=5"
```

```bash
modal serve chess_sandbox/endpoints.py

curl "https://pilipolio--chess-concept-extraction-extract-concepts.modal.run?fen=rnbqkbnr%2Fpppppppp%2F8%2F8%2F4P3%2F8%2FPPPP1PPP%2FRNBQKBNR+b+KQkq+e3+0+1&threshold=0.1"
```

### Production Deployment

Automatic deployment with `modal deploy` on GitHub [releases action](.github/workflows/release.yml)
```
curl "https://pilipolio--chess-analysis-analyze.modal.run?fen=rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR%20w%20KQkq%20-%200%201&depth=20&num_lines=5"

curl "https://pilipolio--chess-concept-extraction-extract-concepts.modal.run?fen=rnbqkbnr%2Fpppppppp%2F8%2F8%2F4P3%2F8%2FPPPP1PPP%2FRNBQKBNR+b+KQkq+e3+0+1&threshold=0.1"
```


## Integration tests

Build and run using Docker.

```bash
docker build -t chess-sandbox .

docker run --rm chess-sandbox \
  /app/.venv/bin/python -m chess_sandbox.engine.analysis \
  "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# Run tests
docker run --rm chess-sandbox:test /app/.venv/bin/python -m pytest -m integration -v
```

## CI/CD

GitHub Actions workflows:
- **PR Checks** ([.github/workflows/pr.yml](.github/workflows/pr.yml)): Formatting, linting, type checking, and unit/integration tests
- **Release** ([.github/workflows/release.yml](.github/workflows/release.yml)): Automatic Modal deployment on GitHub releases

## Bibliography and Resources

### Research Papers

- **Jhamtani & Hovy (2018)** - Learning to Generate Move-by-Move Commentary for Chess Games from Large-Scale Social Forum Data  
  - ACL 2018. Introduces a dataset of 298K chess move-commentary pairs and methods for generating move-by-move commentary. [Paper](https://www.cs.cmu.edu/~hovy/papers/18ACL-chess-commentary.pdf) | [ACL Anthology](https://aclanthology.org/P18-1154/)

- **ChessGPT (2023)** - Bridging Policy Learning and Language Modeling  
  - NeurIPS 2023. Integrates policy learning with language modeling for chess commentary generation. [arXiv:2306.09200](https://arxiv.org/abs/2306.09200) | [Code](https://github.com/waterhorse1/chessgpt)

- **Kim et al. (2025)** - Bridging the Gap between Expert and Language Models: Concept-guided Chess Commentary Generation and Evaluation  
  - NAACL 2025. Integrates expert models with LLMs through concept-guided explanations for accurate and fluent commentary generation. [arXiv:2410.20811](https://arxiv.org/abs/2410.20811) | [Code](https://github.com/ml-postech/concept-guided-chess-commentary)

- **Caissa-AI (2025)** - Neurosymbolic AI for Chess  
  - KI 2025 conference. Modern implementation using LangGraph and Prolog for chess commentary. [Paper](https://link.springer.com/chapter/10.1007/978-3-032-02813-6_11) | [Code](https://github.com/MazenS0liman/Caissa-AI)

### Chess Engines and Tools

- **Leela Chess Zero (lc0)** - Neural network-based chess engine  
  - [Website](https://lczero.org/) | [Repository](https://github.com/LeelaChessZero/lc0) | [Python wrapper](https://pypi.org/project/lcz/) | [JavaScript](https://github.com/frpays/lc0-js/)

- **Maia** - Human-like neural network chess engine  
  - Trained to predict human moves, providing "average" strength gameplay. [Repository](https://github.com/CSSLab/maia-chess) | [Website](https://www.maiachess.com/)

- **lczerolens** - Python library for interpreting lc0 models  
  - PyTorch-based tools for loading, manipulating, and probing lc0 neural network weights. [Repository](https://github.com/Xmaster6y/lczerolens) | [Documentation](https://lczerolens.readthedocs.io/) | [Report](https://hal.science/hal-05321380v1/file/Report-Exploring_capabilities_of_chess_playing_models_V1.pdf) | [Concept probing notebook](https://colab.research.google.com/github/Xmaster6y/lczerolens/blob/main/docs/source/notebooks/features/probe-concepts.ipynb)

### Datasets

- **Gameknot Games** - Social forum chess games  
  - Used in Jhamtani & Hovy (2018) for commentary generation. [Crawler](https://github.com/ml-postech/concept-guided-chess-commentary/tree/master/gameknot_crawler)

- **Annotated PGN Dataset** - Waterhorse chess data  
  - Large collection of annotated chess games from gameknot.com. [Dataset](https://huggingface.co/datasets/Waterhorse/chess_data) | [Download](https://huggingface.co/datasets/Waterhorse/chess_data/resolve/main/chessclip_data/annotated_pgn/annotated_pgn_free.tar.gz)

- **Kaggle Chess Commentary Dataset** - AI-generated commentaries  
  - [Dataset](https://www.kaggle.com/datasets/jayanthrajg/chess-commentary-dataset)

### Additional Resources

- **Awesome Explainable AI** - Curated resources on explainable AI  
  - [Repository](https://github.com/rushrukh/awesome-explainable-ai)

- **Fine-tuning Chess Commentary Models** - Medium article  
  - [Article](https://medium.com/@jasonyip_77999/fine-tuning-a-chess-commentary-model-d8ec8f44a022)

## License

MIT
