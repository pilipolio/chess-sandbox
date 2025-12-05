# chess-sandbox

Implements research ideas for chess commentary generation in a lightweight production-grade ML/AI engineering architecture.

## Research Context

This project builds on recent advances in concept-guided chess commentary generation, specifically:

**[Kim et al. (2025) - "Bridging the Gap between Expert and Language Models"](https://arxiv.org/html/2410.20811v2)** introduced the approach of extracting chess concepts from neural network activations to ground LLM-generated commentary. Their method uses:
- **Leela Chess Zero (LC0) neural network activations** to build vector representations of chess positions
- **Stockfish 8 evaluation heuristics** as concept labels (~24 engine-internal features: material, imbalance, king safety, mobility, threats, etc.)
- **Linear SVM classifiers** for concept detection framed as a multi-class problem (see [code](https://github.com/ml-postech/concept-guided-chess-commentary/blob/master/01_probing_svm.py#L133))

Our implementation extends this approach with a key distinction:

**Human-Grounded Concepts** (our contribution):
- Instead of being limited to Stockfish 8's evaluation heuristics, we rely on **12 concepts from human game commentary**, including tactical motifs such pin, fork, skewer, but alo positional and strategical ones such as outpost, weak square, initiative
- **Two-stage labelling pipeline**: regex pattern matching + LLM validation to extract concepts from annotated PGN games in the [Gameknot dataset](https://huggingface.co/datasets/Waterhorse/chess_data) introduced by [Feng et al. (2023) - ChessGPT](https://arxiv.org/html/2306.09200v2)
- **Annotated Dataset**: [97,995 labelled positions](https://huggingface.co/datasets/pilipolio/chess-positions-concepts) (28K train, 70K test) available on HuggingFace Hub

This enables concept detection and commentary generation grounded in how chess players naturally describe positions, rather than being constrained to engine evaluation heuristics.

**Key Features:**
- Interactive Claude skill for analyzing chess positions in natural language
- Multi‑label classificater using a one-vs-rest layer of logistic regressions detecting concepts from LC0 neural network activations
- Rigorous model evaluation with per-concept precision/recall/AUC metrics
- Serverless inference endpoint for real-time concept extraction
- Reproducible labelling and training pipelines with version-controlled datasets and models

## Architecture

![Architecture Diagram](docs/chess-sandbox-architecture.png)

[View/Edit diagram in Excalidraw](https://excalidraw.com/#room=281bb9f3e913a7cd6a7d,PVceQUonY0HTvcz6fcSLkQ)

### High-level

1. **Chess Commentary Skill** - A [Claude skill](https://www.claude.com/blog/skills) that enables a user to ask natural language query about a chess position (with FEN notation or game references like "Paul Morphy's Opera game") and an LLM to generate commentary grounded in position evaluation and chess concepts:
   - Chess engine evaluation and principal variations ([Stockfish](https://stockfishchess.org/))
   - SVG visualization of positions ([python-chess](https://python-chess.readthedocs.io/))
   - Concept extraction from a bespoke ML exposed via HTTP endpoint
2. **Extract-Concepts Endpoint** - Serverless [Modal.com](https://modal.com/) HTTP endpoint returning detected concepts and confidence scores from a ML classifier.
3. **Training Pipeline** - ML model training (local or Modal) for concept extraction:
   - Trained on [Leela Chess Zero (LC0)](https://lczero.org/) neural network activations
   - Logistic Regression classifiers using [scikit-learn](https://scikit-learn.org/) and [lczerolens](https://lczerolens.readthedocs.io/).
4. **Models and datasets repository** - [HuggingFace](https://huggingface.co/) as centralized storage for datasets and trained models with associated model cards and evaluation metrics.
5. **CI/CD** - GitHub Actions for code quality checks and gated code/model updates.

### Future Works

 * Human and LLM-as-Judge evaluation of generated chess commentaries using standardized quality metrics
 * Data-centric improvements by collecting chess experts' feedback and annotations using https://prodi.gy/
 * Additional board features relevant to extracted concepts
 * Agentic "chess analyst" to explore moves and engine/human lines

### WIP: Chess Puzzle Fine-tuning (`puzzles_trainer`)

Experimental module for fine-tuning small LLMs on chess puzzles using SFT with LoRA. See [docs/chess-llm-finetuning.md](docs/chess-llm-finetuning.md) for the full approach.

**Datasets:**
- [pilipolio/chess-puzzle-tasks](https://huggingface.co/datasets/pilipolio/chess-puzzle-tasks) - Lichess puzzles with multiple task types (puzzle solving, ASCII board, legal moves/captures, piece positions)
- [pilipolio/chess-toy-tasks](https://huggingface.co/datasets/pilipolio/chess-toy-tasks) - Synthetic toy exercises (capture sequences, movement paths, FEN/piece-list conversions, UCI legal moves)
- [pilipolio/chess-mixed-tasks](https://huggingface.co/datasets/pilipolio/chess-mixed-tasks) - Combination of puzzle and toy tasks for curriculum learning

**Dataset Preparation:**

```bash
# Install dependencies (requires system cairo library: brew install cairo)
uv sync --group prepare-data

# Generate puzzle dataset from Lichess puzzles
HF_TOKEN=your-token puzzles-generate-tasks-dataset generate-samples \
    --source puzzle \
    --sample-size 1000 \
    --max-rating 1500 \
    --push-to-hub

# Generate toy curriculum (synthetic exercises)
HF_TOKEN=your-token puzzles-generate-tasks-dataset generate-samples \
    --source toy \
    --sample-size 500 \
    --push-to-hub

# Generate mixed dataset (70% puzzle, 30% toy by default)
HF_TOKEN=your-token puzzles-generate-tasks-dataset generate-samples \
    --source mixed \
    --sample-size 500 \
    --toy-ratio 0.3 \
    --push-to-hub
```

**LLM Evaluation:**

```bash
# Evaluate an OpenAI-compatible model on puzzle tasks
uv run python -m chess_sandbox.puzzles_trainer.llm_evaluation \
    --dataset-id pilipolio/lichess-puzzle-tasks \
    --model gpt-4o-mini \
    --sample-size 100 \
    --output results.jsonl
```

**Training on Modal:**

```bash
modal run chess_sandbox/puzzles_trainer/modal_pipeline.py::train \
    --max-steps 500 \
    --eval-steps 50 \
    --wandb-project chess-puzzles \
    --model-id Qwen/Qwen3-4B-Instruct-2507 \
    --output-model-id pilipolio/chess-puzzle-sft-qwen3-4b
```

**Local Training:**

```bash
uv sync --group sft --group prepare-data
puzzles-generate-tasks-dataset train \
    --model-id Qwen/Qwen3-0.6B \
    --max-steps 100 \
    --eval-steps 20
```

## Project 

Scaffolding templated from [postmodern-python](https://github.com/carderne/postmodern-python)

### Stack
- **Python 3.13+**
- **Package Management:** [uv](https://docs.astral.sh/uv/)
- **Type Safety:** Pydantic models, pyright (strict mode)
- **Code Quality:** ruff (format/lint), pytest (unit tests)
- **CI/CD:** GitHub Actions

### Code Structure

```
chess_sandbox/
├── config.py                  # Settings management
├── engine/                    # Chess engine wrappers (Stockfish, LC0)
│   ├── ...
├── concept_extraction/        # ML-based concept detection
│   ├── endpoints.py           # Modal HTTP endpoint for concept extraction
│   ├── model/                 # Model training and inference
│   │   ├── ...
│   └── labelling/             # Dataset creation (regex + LLM)
│       ├── ...
└── commentary/                # Experimental: OpenAI-based commentary
    └── ...

.claude/skills/chess-commentator/  # Claude skill for interactive analysis
    ├── skill.md               # Skill prompt and instructions
    └── scripts/               # Tools: engine analysis, SVG, concept extraction

docs/
├── adrs/                      # Architectural Decision Records
└── plans/                     # Implementation plans
```

## Quick Start

### Claude Code Skill

Requires to clone the repository and have Claude Code installed:

```bash
git clone git@github.com:pilipolio/chess-sandbox.git
cd chess-sandbox
claude

...

> Analyse this position
r2qk2r/p1p2p2/p2p1n1p/3Pp1p1/1P1bP3/P1N2QBP/2P2PP1/R4RK1 w kq - 2 15

⏺ I'll analyze this chess position using the chess-commentator skill. Let me
  start by querying the analysis endpoints.

⏺ Bash(cd /Users/guillaumeallain/D/chess-sandbox/.claude/skills/chess-commentator
      && python3 scripts/query_analysis.py
      "r2qk2r/p1p2p2/p2p1n1p/3Pp1p1/1P1bP3/P1N2QBP/2P2…)
  ⎿  Running…
...

  Thematic Insights

  This middlegame position features strong pin pressure (60.8% confidence) and
   potential fork tactics (24.8% confidence). Black's bishop on d4 is
  well-centralized and actively pinning the Nc3, creating coordination
  challenges for White. The position revolves around king safety, piece
  activity, and control of key central squares.```

...
```

### Development 

**Required:**
- [Stockfish](https://stockfishchess.org/download/) chess engine

**Optional (for advanced features):**
- [Modal](https://modal.com/) account - Required only for serverless deployment and running training pipelines on Modal
- [HuggingFace](https://huggingface.co/) account - Required only for training models or pushing models/datasets to the Hub
- OpenAI API key - Required only for dataset creation with LLM validation

### Setup

1. Install dependencies:

**CPU-only (default, ~100MB - recommended for most users):**
```bash
uv sync
```

> **Note:** PyTorch installs as CPU-only by default (~100MB), saving ~6.9GB compared to GPU builds. This is sufficient for most use cases including inference and model evaluation.

**GPU with CUDA support (~7GB - only if you have NVIDIA GPU):**
```bash
UV_INDEX_URL=https://download.pytorch.org/whl/cu124 uv sync
```

> **Note:** GPU support requires overriding the default PyTorch index. Replace `cu124` with your desired CUDA version. Only needed for GPU-accelerated training or large-scale inference.

2. Set environment variables:
```bash
cp .env.example .env

# Edit .env and set:
# STOCKFISH_PATH=/opt/homebrew/bin/stockfish  # macOS Homebrew
# OPENAI_API_KEY=your-api-key-here           # Optional: for dataset creation only
```

3. Verify installation:

```bash
uv run poe all
```

4. (Optional) Activate virtual environment to run commands using `python` instead of `uv run python`

```bash
source .venv/bin/activate
```

5. (Optional) Test local concept extraction

```bash
# Downloads models from HuggingFace Hub (public models, no auth required)
uv run python -m chess_sandbox.concept_extraction.model.inference predict "r2qk2r/p1p2p2/p2p1n1p/3Pp1p1/1P1bP3/P1N2QBP/2P2PP1/R4RK1 w kq - 2 15"
```

Note: If you encounter authentication issues, you may need to create a HuggingFace model and set-up local credentials:

```bash
uv run hf
```

## Usage

### Interactive Commentary via Claude Skill

The `chess-commentator` skill provides interactive chess analysis in your conversation with Claude Code or Claude.AI:

**Example:**
```
> Analyze this chess position: 8/8/2K5/p1p5/P1P5/1k6/8/8 w - - 0 58
```

Claude automatically uses the skill to:
1. Run Stockfish analysis on the position
2. Extract tactical/strategic concepts via the ML endpoint
3. Generate an SVG visualization
4. Provide natural language commentary explaining best moves, key variations, and chess concepts

The skill is located in `.claude/skills/chess-commentator/` and triggers automatically when you provide FEN positions or reference specific games.

**Installation:**

- **Claude Code users**: The skill is automatically available when working in this repository
- **Claude.AI users**: Zip the `.claude/skills/chess-commentator/` directory and add to `https://claude.ai/settings/capabilities`

## Concept Extraction

### Inference

To inspect concept predictions locally, re-use the Quick Start command and review the JSON output:
```bash
uv run python -m chess_sandbox.concept_extraction.model.inference predict "r2qk2r/p1p2p2/p2p1n1p/3Pp1p1/1P1bP3/P1N2QBP/2P2PP1/R4RK1 w kq - 2 15"
```

Sample (truncated) output:
```json
{
  "input_fen": "r2qk2r/p1p2p2/p2p1n1p/3Pp1p1/1P1bP3/P1N2QBP/2P2PP1/R4RK1 w kq - 2 15",
  "predictions": [
    {"concept": "weak_back_rank", "confidence": 0.91},
    {"concept": "passed_pawn", "confidence": 0.63}
  ]
}
```

For remote inference without downloading models locally, call the production endpoint:

```bash
curl "https://pilipolio--chess-concept-extraction-extract-concepts.modal.run?fen=rnbqkbnr%2Fpppppppp%2F8%2F8%2F4P3%2F8%2FPPPP1PPP%2FRNBQKBNR+b+KQkq+e3+0+1&threshold=0.1"
```

Development mode (ephemeral deployment):
```bash
modal serve chess_sandbox/concept_extraction/endpoints.py

curl "https://pilipolio--chess-concept-extraction-extract-concepts-dev.modal.run?fen=..."
```

### Training the ML Model

The training pipeline extracts chess concepts from LC0 neural network activations using logistic regression classifiers (requires HF set-up)

**Local training:**
```bash
uv run python -m chess_sandbox.concept_extraction.model.train \
  --dataset-repo-id pilipolio/chess-positions-concepts \
  --lc0-model-repo-id lczerolens/maia-1500 \
  --layer-name block3/conv2/relu \
  --classifier-mode multi-label \
  --upload-to-hub --save-splits \
  --output-repo-id pilipolio/chess-positions-extractor \
  --n-jobs 4 --verbose 1
  --output-revision test_fixture

uv run python -m chess_sandbox.concept_extraction.model.evaluation evaluate \
  --classifier-model-repo-id pilipolio/chess-positions-extractor \
  --dataset-repo-id pilipolio/chess-positions-concepts \
  --dataset-filename test.jsonl \
  --sample-size 10
```

**Modal training (serverless):**

Requires Modal set-up with the `huggingface-read-write-secret` added to [secrets](https://modal.com/docs/guide/secrets):

```bash
modal run chess_sandbox/concept_extraction/model/modal_pipeline.py::train \
    --dataset-repo-id pilipolio/chess-positions-concepts \
    --lc0-model-repo-id lczerolens/maia-1500 \
    --classifier-mode multi-label \
    --upload-to-hub \
    --output-repo-id pilipolio/chess-positions-extractor
```

Detected concepts include tactical themes (pin, fork, skewer, sacrifice) and strategic themes (passed pawn, outpost, weak square, zugzwang). Models and datasets are versioned on [HuggingFace Hub](https://huggingface.co/pilipolio).

### Dataset Creation (Optional)

For creating new training datasets from annotated PGN files, see [docs/plans/concept-labelling-pipeline.md](docs/plans/concept-labelling-pipeline.md). This pipeline uses regex patterns and optional LLM validation to extract concepts from game annotations.


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

## Deployment

### Modal Serverless Endpoints

The extract-concepts endpoint is deployed as a serverless HTTP endpoint on Modal. See [docs/adrs/20251029-use-modal-for-serverless-endpoints.md](docs/adrs/20251029-use-modal-for-serverless-endpoints.md) for architectural rationale.

**Prerequisites:**
1. Create account at https://modal.com
2. Generate API token at https://modal.com/settings/tokens
3. Authenticate: `modal token set --token-id <ID> --token-secret <SECRET>`

**Development deployment:**
```bash
modal serve chess_sandbox/concept_extraction/endpoints.py
# Access at: https://pilipolio--chess-concept-extraction-extract-concepts-dev.modal.run
```

**Production deployment:**

Automated via GitHub Actions on [releases](.github/workflows/release.yml). The production endpoint is available at:
```
https://pilipolio--chess-concept-extraction-extract-concepts.modal.run
```

### Docker (Integration Testing)

```bash
docker build -t chess-sandbox .
docker run --rm chess-sandbox /app/.venv/bin/python -m pytest -m integration -v
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
