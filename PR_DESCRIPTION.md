# feat: Add Stockfish 8 concept extraction

## Summary

Implements Stockfish 8 concept extraction based on the approach from the research paper ["Concept-Guided Chess Commentary"](https://arxiv.org/html/2410.20811v2).

This adds a Python API to extract detailed chess concepts from positions using Stockfish 8's static evaluation command.

## Stockfish 8 Concepts

The implementation extracts **12 chess concepts** from any position:

### Positional Concepts
- **Material** - Raw material balance
- **Imbalance** - Material imbalance (bishop pair advantage, etc.)
- **Pawns** - Pawn structure evaluation
- **Space** - Space advantage

### Piece Activity
- **Knights** - Knight positioning and activity
- **Bishops** - Bishop positioning and activity
- **Rooks** - Rook positioning and activity
- **Queens** - Queen positioning and activity
- **Mobility** - Overall piece mobility (available moves)

### Tactical Concepts
- **King safety** - King safety evaluation
- **Threats** - Tactical threats in the position
- **Passed pawns** - Passed pawn advantage

Each concept provides:
- White middlegame (MG) score
- White endgame (EG) score
- Black middlegame (MG) score
- Black endgame (EG) score
- Total advantage (MG and EG)
- Overall evaluation

## Implementation Details

**Stockfish 8 Binary:**
- Compiled from source (tag `sf_8`)
- Located at `data/engines/stockfish-8/src/stockfish` (gitignored)

**Python Module:** `chess_sandbox/concept_extraction/stockfish_concepts.py`
- `Stockfish8ConceptExtractor` class
- Communicates with Stockfish via UCI protocol
- Parses detailed `eval` command output
- Returns structured `PositionConcepts` dataclass

## Usage Example

```python
from pathlib import Path
from chess_sandbox.concept_extraction.stockfish_concepts import Stockfish8ConceptExtractor

# Initialize
stockfish_path = Path("data/engines/stockfish-8/src/stockfish")
extractor = Stockfish8ConceptExtractor(stockfish_path)

# Extract concepts from a position
fen = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
concepts = extractor.get_concepts(fen)

# Access individual concepts
print(f"Mobility: {concepts.mobility.total_advantage:+.2f}")
print(f"King safety: {concepts.king_safety.total_advantage:+.2f}")
print(f"Total eval: {concepts.total_eval:+.2f}")

# Get all concepts as dictionary
concept_dict = concepts.to_dict()
```

## Example Output

Italian Game position:
```
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

## Files Added

- `chess_sandbox/concept_extraction/stockfish_concepts.py` - Core implementation (262 lines)
- `docs/stockfish8-concepts.md` - Comprehensive documentation (200 lines)
- `examples/stockfish8_concepts_demo.py` - Demo script with multiple positions (77 lines)
- `test_stockfish8.py` - Quick test script (29 lines)

**Total: 568 lines added**

## Use Cases

This implementation enables:
- **Position analysis** - Detailed concept-level evaluation
- **Training data labeling** - Label positions by concept strength (as in the paper)
- **Concept-guided commentary** - Generate explanations based on key concepts
- **Research** - Study position characteristics and concept relationships

## References

- Paper: [Concept-Guided Chess Commentary](https://arxiv.org/html/2410.20811v2)
- Original repo: [ml-postech/concept-guided-chess-commentary](https://github.com/ml-postech/concept-guided-chess-commentary)
- Stockfish 8: [GitHub Release](https://github.com/official-stockfish/Stockfish/releases/tag/sf_8) (Nov 2016)

## Testing

Run the quick test:
```bash
python3 test_stockfish8.py
```

Run the full demo (requires dependencies):
```bash
uv run python examples/stockfish8_concepts_demo.py
```

## Branch

- Base: `main`
- Head: `claude/run-stockfish-older-version-011CV2XF28GfvUTJTyJhG6ZX`
