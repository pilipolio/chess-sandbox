# Stockfish 8 Concept Extraction

This document describes how to extract chess concepts from positions using Stockfish 8's detailed static evaluation.

## Overview

Based on the research paper ["Concept-Guided LLM Agents for Human-AI Safety Coevolution in Driving"](https://arxiv.org/html/2410.20811v2) and the accompanying [concept-guided-chess-commentary repository](https://github.com/ml-postech/concept-guided-chess-commentary), we've implemented a Python interface to extract chess concepts using Stockfish 8.

## Concepts Extracted

Stockfish 8's `eval` command provides detailed evaluation breakdowns for the following concepts:

- **Material**: Raw material balance
- **Imbalance**: Material imbalance (bishop pair, etc.)
- **Pawns**: Pawn structure evaluation
- **Knights**: Knight positioning and activity
- **Bishops**: Bishop positioning and activity
- **Rooks**: Rook positioning and activity
- **Queens**: Queen positioning and activity
- **Mobility**: Piece mobility (available moves)
- **King safety**: King safety evaluation
- **Threats**: Tactical threats
- **Passed pawns**: Passed pawn advantage
- **Space**: Space advantage

Each concept provides scores for:
- White middlegame (MG)
- White endgame (EG)
- Black middlegame (MG)
- Black endgame (EG)
- Total middlegame advantage
- Total endgame advantage

## Installation

### 1. Compile Stockfish 8

Stockfish 8 is already compiled and available at:
```
data/engines/stockfish-8/src/stockfish
```

If you need to recompile:

```bash
cd data/engines
git clone https://github.com/official-stockfish/Stockfish.git stockfish-8
cd stockfish-8
git checkout sf_8
cd src
make -j build ARCH=x86-64
```

### 2. Configuration

The extractor follows the `EngineConfig` pattern from `chess_sandbox.engine.analyse`. You can configure the Stockfish 8 path in three ways:

**Option 1: Use default path (recommended)**
```python
from chess_sandbox.concept_extraction.stockfish_concepts import Stockfish8Config
config = Stockfish8Config()  # Uses data/engines/stockfish-8/src/stockfish
```

**Option 2: Set in environment/settings**
```bash
# In .env file
STOCKFISH_8_PATH=path/to/your/stockfish8
```

**Option 3: Specify directly**
```python
from pathlib import Path
config = Stockfish8Config(stockfish_8_path=Path("custom/path/to/stockfish8"))
```

### 3. Python Dependencies

The `chess` library is required and is already in the project dependencies.

## Usage

### Python API

```python
from chess_sandbox.concept_extraction.stockfish_concepts import (
    Stockfish8ConceptExtractor,
    Stockfish8Config,
)

# Initialize with default config (uses STOCKFISH_8_PATH from settings)
config = Stockfish8Config()
extractor = Stockfish8ConceptExtractor(config)

# Or specify a custom path
from pathlib import Path
custom_config = Stockfish8Config(stockfish_8_path=Path("path/to/stockfish8"))
extractor = Stockfish8ConceptExtractor(custom_config)

# Extract concepts from a position
fen = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
concepts = extractor.get_concepts(fen)

# Access individual concepts
print(f"Mobility advantage: {concepts.mobility.total_advantage:.2f}")
print(f"King safety advantage: {concepts.king_safety.total_advantage:.2f}")
print(f"Total evaluation: {concepts.total_eval:.2f}")

# Get all concepts as a dictionary
concept_dict = concepts.to_dict()
for concept_name, score in concept_dict.items():
    print(f"{concept_name}: {score:+.2f}")
```

### Batch Processing

```python
fens = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
]

all_concepts = extractor.get_concepts_batch(fens)
```

### Command Line Test

A simple test script is available:

```bash
python3 test_stockfish8.py
```

Or run the full demo (requires all dependencies):

```bash
uv run python examples/stockfish8_concepts_demo.py
```

## Example Output

For the Italian Game position:
```
FEN: r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3

Concept scores (total advantage):
  material       : +0.16
  imbalance      : +0.00
  pawns          : +0.00
  knights        : +0.00
  bishops        : -0.03
  rooks          : +0.04
  queens         : +0.00
  mobility       : +0.00
  king_safety    : +0.07
  threats        : +0.14
  passed_pawns   : +0.00
  space          : +0.00

Total evaluation: +0.24
```

## Technical Details

### Why Subprocess Instead of python-chess?

The `eval` command is a **non-standard UCI extension** specific to Stockfish. The python-chess library's engine API (`chess.engine.SimpleEngine`) only supports standard UCI commands like `go`, `stop`, and `setoption`.

While python-chess has internal `send_line()` methods in its `Protocol` class, these are:
- Not part of the public API
- Designed for asynchronous communication
- Require complex command queueing and response handling

Since the `eval` command is a simple request-response pattern that returns text output (not standard UCI info lines), using subprocess provides:
- ✅ Simpler, more direct communication
- ✅ Easier to parse the non-standard tabular output
- ✅ No dependencies on internal python-chess APIs
- ✅ Consistent with the codebase's `EngineConfig` pattern

### How It Works

1. The extractor communicates with Stockfish 8 via subprocess using the UCI protocol
2. For each position, it sends:
   ```
   uci
   position fen <fen-string>
   eval
   quit
   ```
3. Stockfish responds with a detailed evaluation table showing all concept scores
4. The output is parsed to extract middlegame and endgame scores for each concept
5. A total advantage is computed as the average of MG and EG scores

### Integration with Existing Code

The implementation follows the `EngineConfig` pattern from `chess_sandbox.engine.analyse`:
- Uses `Stockfish8Config` for configuration (similar to `EngineConfig`)
- Reads from `settings.STOCKFISH_8_PATH` (similar to `settings.STOCKFISH_PATH`)
- Provides sensible defaults with override capability
- Validates configuration before use

### Parsing the Eval Output

Stockfish 8's eval command returns a table like:

```
      Eval term |    White    |    Black    |    Total
                |   MG    EG  |   MG    EG  |   MG    EG
----------------+-------------+-------------+-------------
       Material |   ---   --- |   ---   --- |  0.19  0.12
       Mobility | -0.08  0.02 | -0.09  0.03 |  0.01 -0.00
    King safety |  0.83 -0.06 |  0.73 -0.10 |  0.10  0.04
----------------+-------------+-------------+-------------
          Total |   ---   --- |   ---   --- |  0.32  0.49

Total Evaluation: 0.24 (white side)
```

The parser:
- Splits each line by `|` to extract columns
- Parses numeric values (handling `---` as 0.0)
- Maps evaluation terms to Python attributes
- Extracts the final total evaluation

## Comparison with the Paper

The paper's approach:
1. Collected 200,000 positions from Lichess
2. Used Stockfish 8 to evaluate each position for all concepts
3. Sorted positions by concept scores
4. Labeled top and bottom 5% as positive/negative samples
5. Trained SVMs on LeelaChessZero representations to extract concept vectors

Our implementation provides the foundation (step 2) - extracting concept scores from Stockfish 8. This can be used to:
- Analyze positions for concept presence
- Label training data for ML models
- Generate concept-guided chess commentary
- Research position characteristics

## References

- Paper: [Concept-Guided Chess Commentary](https://arxiv.org/html/2410.20811v2)
- Original implementation: [GitHub Repository](https://github.com/ml-postech/concept-guided-chess-commentary)
- Stockfish 8: [GitHub Release](https://github.com/official-stockfish/Stockfish/releases/tag/sf_8)

## Future Enhancements

Potential improvements:
- Add caching for repeated positions
- Support parallel batch processing
- Implement move-based concept extraction (STS puzzles)
- Integration with existing concept extraction pipeline
- Support for custom concept definitions
