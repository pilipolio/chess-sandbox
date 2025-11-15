# Mirascope & Lilypad Evaluation Loop POC

This POC demonstrates how to use **Mirascope** and **Lilypad** to build an evaluation loop around the chess commentary system.

## ⚠️ Important Compatibility Note

**Current Status**: This POC uses Mirascope 2.0.0a1 (alpha/pre-release) which has a known compatibility issue with Python 3.13. The library imports successfully but cannot be run directly due to a typing error in the mirascope package itself.

**Recommendation**: This POC is provided as a **code reference and architecture demonstration**. To run it in production:
1. Wait for Mirascope 2.0 stable release with Python 3.13 support
2. Use Python 3.11 or 3.12 instead
3. Monitor https://github.com/Mirascope/mirascope for compatibility updates

The code structure, API usage patterns, and integration approach demonstrated here will remain valid once compatibility is resolved.

## Overview

The POC consists of two main components:

1. **MirascopeCommentator** (`chess_sandbox/commentary/mirascope_commentator.py`):
   - Uses Mirascope's `@llm.call` decorator for clean LLM integration
   - Uses Lilypad's `@lilypad.trace` decorator for automatic versioning and tracing
   - Maintains compatibility with the existing commentary system

2. **Lilypad Evaluation Loop** (`chess_sandbox/commentary/lilypad_evaluation.py`):
   - Runs systematic evaluations across different configurations
   - Automatically traces all LLM calls (both commentary generation and evaluation)
   - Versions different approaches for comparison
   - Stores results for analysis

## Key Benefits

### Mirascope Integration
- **Clean API**: `@llm.call` decorator transforms functions into LLM calls
- **Provider Flexibility**: Easy switching between OpenAI, Anthropic, etc.
- **Response Models**: Automatic parsing into Pydantic models
- **Type Safety**: Full type hints and IDE support

### Lilypad Integration
- **Automatic Versioning**: `@lilypad.trace(versioning="automatic")` tracks code changes
- **Complete Tracing**: Captures all inputs, outputs, prompts, costs, and latency
- **LLM Call Tracking**: `auto_llm=True` automatically traces all LLM API calls
- **Data Flywheel**: Builds a dataset of traced calls for continuous improvement

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Lilypad Evaluation Loop                   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Config 1: gpt-4o-mini + no engine                  │   │
│  │  Version: v1                                        │   │
│  │  Traces: [trace_id_1, trace_id_2, ...]            │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Config 2: gpt-4o-mini + engine (5 lines)          │   │
│  │  Version: v1                                        │   │
│  │  Traces: [trace_id_3, trace_id_4, ...]            │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Config 3: gpt-4o + engine, no tactical            │   │
│  │  Version: v1                                        │   │
│  │  Traces: [trace_id_5, trace_id_6, ...]            │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              Lilypad Platform (Cloud/Self-Hosted)           │
│                                                             │
│  • Stores all traces with full context                     │
│  • Enables version comparison                              │
│  • Supports annotation and labeling                        │
│  • Provides analytics and insights                         │
└─────────────────────────────────────────────────────────────┘
```

## Setup

### 1. Install Dependencies

Dependencies are already installed (done via `uv add --prerelease=allow`):
- `mirascope[openai]` (v2.0.0a1)
- `lilypad-sdk[openai]` (v0.10.4)

### 2. Configure Lilypad

#### Option A: Use Lilypad Cloud (Recommended for POC)

1. Create an account at https://lilypad.mirascope.com
2. Create a project and generate an API key
3. Set environment variables:

```bash
export LILYPAD_PROJECT_ID="your-project-id"
export LILYPAD_API_KEY="your-api-key"
export OPENAI_API_KEY="your-openai-key"
```

#### Option B: Self-Host Lilypad (Optional)

Follow instructions at: https://mirascope.com/docs/lilypad/getting-started/self-hosting

### 3. Prepare Ground Truth Data

Ensure you have ground truth data at `data/processed/chessdotcom.jsonl` with this format:

```json
{"fen": "...", "themes": ["theme1", "theme2", ...]}
{"fen": "...", "themes": ["theme1", "theme2", ...]}
```

## Usage

### Running the Evaluation Loop

```bash
# Run the full POC (limited to 3 positions for quick testing)
uv run python -m chess_sandbox.commentary.lilypad_evaluation

# Or run directly
uv run chess_sandbox/commentary/lilypad_evaluation.py
```

### Using the Mirascope Commentator Standalone

```python
from chess_sandbox.commentary.mirascope_commentator import MirascopeCommentator
import chess

# Configure
params = {
    "engine": {"depth": 20, "num_lines": 5},
    "llm": {"model": "gpt-4o-mini"},
    "include_tactical_patterns": True,
    "lilypad": {
        "project_id": "your-project-id",
        "api_key": "your-api-key",
    },
}

# Analyze a position
commentator = MirascopeCommentator.create(params)
board = chess.Board("8/8/2K5/p1p5/P1P5/1k6/8/8 w - - 0 58")
result = commentator.analyze(board)

print(f"Best Move: {result.best_move}")
print(f"Themes: {result.themes}")
```

## Key Features Demonstrated

### 1. Automatic Versioning

Every time you modify the `analyze_position_with_llm` or `judge_themes` functions, Lilypad automatically creates a new version. This allows you to:

- Track which code version produced which results
- Compare performance across code changes
- Roll back to previous versions if needed

### 2. Complete Tracing

All LLM calls are automatically traced with:
- Input prompts
- Output responses
- Model parameters
- Cost (tokens used)
- Latency
- Code version that made the call

### 3. Evaluation Loop

The evaluation loop demonstrates the "data flywheel" concept:

1. **Generate**: Run commentator on positions
2. **Evaluate**: Compare predictions with ground truth
3. **Analyze**: Review traces in Lilypad dashboard
4. **Annotate**: Label good/bad examples
5. **Optimize**: Modify prompts or configs
6. **Iterate**: Repeat with automatic versioning

### 4. Configuration Comparison

The POC evaluates three different configurations:

1. **gpt-4o-mini without engine**: Pure LLM analysis
2. **gpt-4o-mini with engine**: LLM + Stockfish analysis
3. **gpt-4o with engine, no tactical patterns**: Different model + feature ablation

Results are saved to `data/results/lilypad_eval_{config_name}.jsonl`

## Viewing Results

### In Lilypad Dashboard

After running the evaluation:

1. Go to https://lilypad.mirascope.com
2. Select your project
3. View traces organized by version and function
4. Compare performance across configurations
5. Annotate examples for future improvement

### Local Results

Results are saved as JSONL files in `data/results/`:

```bash
# View results for a specific configuration
cat data/results/lilypad_eval_gpt4o_mini_with_engine.jsonl | jq

# Calculate average score
cat data/results/lilypad_eval_*.jsonl | jq -s 'map(.score) | add / length'
```

## Next Steps

1. **Expand Evaluation Set**: Increase from 3 to more positions
2. **Add More Configurations**: Test different models, prompts, or engine settings
3. **Implement Annotations**: Use Lilypad UI to label high/low quality outputs
4. **Analyze Patterns**: Identify where configurations succeed/fail
5. **Iterate**: Use insights to improve prompts and code
6. **A/B Testing**: Compare new versions against baselines

## Troubleshooting

### No Lilypad Credentials

If you don't set `LILYPAD_PROJECT_ID` and `LILYPAD_API_KEY`, the POC will still run but won't send traces to Lilypad. You'll see:

```
Warning: LILYPAD_PROJECT_ID and LILYPAD_API_KEY not set.
Running without Lilypad tracing.
```

### Python 3.13 Compatibility Issue

The current version of Mirascope (2.0.0a1) has a typing compatibility issue with Python 3.13. You may see:

```
TypeError: can only concatenate list (not "tuple") to list
```

**Solutions**:
- Downgrade to Python 3.11 or 3.12
- Wait for Mirascope 2.0 stable release
- Use the POC as a code reference while waiting for compatibility fixes

### Missing Ground Truth Data

If `data/processed/chessdotcom.jsonl` doesn't exist, create sample data:

```bash
mkdir -p data/processed
cat > data/processed/chessdotcom.jsonl << 'EOF'
{"fen": "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", "themes": ["open game", "knight development", "center control"]}
{"fen": "8/8/2K5/p1p5/P1P5/1k6/8/8 w - - 0 58", "themes": ["opposition", "king and pawn endgame", "zugzwang"]}
EOF
```

## Technical Details

### Mirascope Decorator Chain

```python
@llm.call(provider="openai", response_model=ChessPositionExplanation)
@lilypad.trace(versioning="automatic")
def analyze_position_with_llm(analysis_text: str, tactical_context: str, model: str) -> str:
    return ANALYSIS_PROMPT.format(...)
```

This decorator chain:
1. `@lilypad.trace` wraps the function for versioning/tracing (outer)
2. `@llm.call` transforms the return value into an LLM API call (inner)
3. The function returns a prompt string
4. Mirascope sends it to OpenAI and parses the response
5. Lilypad captures everything automatically

### Version Detection

Lilypad's `versioning="automatic"` uses code hashing to detect changes:
- Changes to function body → new version
- Changes to decorators → new version
- Changes to imported dependencies → tracked in trace

## References

- [Mirascope Documentation](https://mirascope.com/docs)
- [Lilypad Documentation](https://mirascope.com/docs/lilypad)
- [Lilypad GitHub](https://github.com/Mirascope/lilypad)
- [Lilypad Cloud](https://lilypad.mirascope.com)
