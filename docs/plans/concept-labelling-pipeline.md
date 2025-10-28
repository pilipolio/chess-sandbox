# Chess Concept Labeling Pipeline - Implementation Plan

Building a dataset of chess positions and labels, based on https://huggingface.co/datasets/Waterhorse/chess_data/blob/main/chessclip_data/annotated_pgn/annotated_pgn_free.tar.gz and https://arxiv.org/pdf/2410.20811

**Status:** ✅ Implemented

## Implementation Summary

The pipeline has been implemented with a simplified architecture:
- **Single CLI command** for parsing + labeling (instead of separate commands)
- **Flat module structure** in `chess_sandbox/concept_labelling/`
- **Python-based patterns** (instead of YAML config)
- **Comprehensive doctests** in all modules
- **Integration test** covering the full pipeline

## Target Concepts (with occurrence counts)

**Tactical:**
- pin, fork, skewer, discovered attack: ~8,824 files
- sacrifice: ~3,329 files

**Strategic:**
- passed pawn: ~1,670 files
- outpost: ~764 files
- weak square / weakness: ~5,046 files
- initiative: (needs validation)
- zugzwang: ~171 files ✓

**Material:**
- sacrifice: ~3,329 files
- exchange: (overlap with general usage)

**King safety:**
- mating threat / mate: ~8,733 files
- exposed king / weakness: ~5,046 files

---

## Implemented Architecture

### Module Structure
```
chess_sandbox/concept_labelling/
├── __init__.py
├── models.py           # Data models (LabelledPosition)
├── patterns.py         # Concept regex patterns (Python constants)
├── parser.py           # PGN parsing & position extraction
├── labeller.py         # Concept detection logic
├── pipeline.py         # Main CLI (parse + label combined)
└── lichess_export.py   # Lichess study export
```

### Data Flow
```
data/raw/annotated_pgn_free/gameknot/*.pgn
  ↓ pipeline.py
data/processed/concept_labelling/positions_labeled.jsonl
  ↓ lichess_export.py
data/exports/lichess_study_sample.pgn
```

---

## Phase 1: PGN Parsing & Position Extraction (✅ Implemented)

### Libraries
- **python-chess** (v1.10+): PGN parsing, board state, FEN conversion
- **dataclasses**: Data validation for labeled positions

### Implementation

**Module:** `chess_sandbox/concept_labelling/parser.py`

```python
# Core data structure
@dataclass
class LabelledPosition:
    fen: str                    # Position in FEN notation
    move_number: int            # Full move number
    side_to_move: str          # 'white' or 'black'
    comment: str               # Associated annotation text
    game_id: str               # Source game identifier
    concepts: List[str]        # Detected concepts (filled in Phase 2)
```

**Key functions:**
- `parse_pgn_file(path: Path) -> List[Game]`
- `extract_positions(game: Game) -> List[LabelledPosition]`
- `associate_comments(position, node) -> str`

### Output Artifact
**Format:** JSONL (JSON Lines) - one position per line
**Path:** `data/processed/positions_raw.jsonl`

**Example entry:**
```json
{
  "fen": "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
  "move_number": 3,
  "side_to_move": "white",
  "comment": "Pin that knight to a bishop (and potentially his king).",
  "game_id": "gameknot_1160",
  "concepts": []
}
```

---

## Phase 2: Concept Labeling (✅ Implemented)

### Libraries
- **re**: Regex pattern matching

### Implementation

**Modules:**
- `chess_sandbox/concept_labelling/patterns.py` - Pattern definitions as Python constants
- `chess_sandbox/concept_labelling/labeller.py` - Concept detection logic

**Pattern example (patterns.py):**
```python
PIN_PATTERNS = [
    r"\bpin(?:s|ned|ning)?\b",
    r"pinned?\s+(?:to|against)",
]

CONCEPT_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "pin": [re.compile(p, re.IGNORECASE) for p in PIN_PATTERNS],
    "fork": [re.compile(p, re.IGNORECASE) for p in FORK_PATTERNS],
    # ... 12 concepts total
}
```

---

## Phase 3: Lichess Study Export (✅ Implemented)

### Libraries
- **random**: Stratified sampling (get examples of each concept)

### Implementation

**Module:** `chess_sandbox/concept_labelling/lichess_export.py`

**Key functions:**
- `sample_positions(positions, n_samples, strategy)` - Sample positions by concept
- `position_to_pgn(position)` - Convert to PGN format with tags
- `load_labeled_positions(jsonl_path)` - Load positions from JSONL

**PGN format with tags:**
```pgn
[Event "Concept: pin"]
[Site "gameknot_1160"]
[FEN "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"]

{ Pin that knight }
```

### CLI Command
```bash
uv run python -m chess_sandbox.concept_labelling.lichess_export \
  --input data/processed/concept_labelling/positions_labeled.jsonl \
  --output-pgn data/exports/lichess_study_sample.pgn \
  --n-samples 64 \
  --strategy balanced \
  --seed 42  # Optional: for reproducible sampling
```

**Note:** The exported PGN can be manually imported to Lichess via:
1. https://lichess.org/study → New Study → Import PGN

---

## Full Pipeline Execution (✅ Implemented)

**Quick test (5 games):**
```bash
# Parse + label in one command
uv run python -m chess_sandbox.concept_labelling.pipeline \
  --input-dir data/raw/annotated_pgn_free/gameknot \
  --output data/processed/concept_labelling/positions_labeled.jsonl \
  --stats data/processed/concept_labelling/concept_stats.json \
  --limit 5

# Export sample to Lichess PGN
uv run python -m chess_sandbox.concept_labelling.lichess_export \
  --input data/processed/concept_labelling/positions_labeled.jsonl \
  --output-pgn data/exports/lichess_study_sample.pgn \
  --n-samples 10 \
  --seed 42
```

**Full pipeline (all games):**
```bash
# Process all PGN files
uv run python -m chess_sandbox.concept_labelling.pipeline \
  --input-dir data/raw/annotated_pgn_free/gameknot \
  --output data/processed/concept_labelling/positions_labeled.jsonl \
  --stats data/processed/concept_labelling/concept_stats.json

# Export 64 samples for review
uv run python -m chess_sandbox.concept_labelling.lichess_export \
  --input data/processed/concept_labelling/positions_labeled.jsonl \
  --output-pgn data/exports/lichess_study_sample.pgn \
  --n-samples 64
```

---

## Directory Structure (✅ Implemented)
```
chess_sandbox/concept_labelling/    # Main package
├── __init__.py
├── models.py                      # LabelledPosition dataclass
├── patterns.py                    # Concept regex patterns
├── parser.py                      # PGN parsing
├── labeller.py                    # Concept detection
├── pipeline.py                    # Main CLI (parse + label)
└── lichess_export.py              # PGN export for Lichess

data/
├── raw/annotated_pgn_free/gameknot/  # Input PGNs (12,769 files)
├── processed/concept_labelling/
│   ├── positions_labeled.jsonl   # Output: labeled positions
│   └── concept_stats.json        # Concept distribution stats
└── exports/
    └── lichess_study_sample.pgn  # Lichess-importable PGN

tests/
└── test_integration.py           # Full pipeline integration test
```

---

## Next Steps (Post-Pipeline)
- Review Lichess study to validate label quality
- Refine regex patterns based on false positives/negatives
- Add temporal context (is concept a threat or actual position?)
- Build train/val/test splits
- Create data loader for neural network training

---

## Key Considerations

**Dataset strengths:**
- Natural language annotations (human-interpretable concepts)
- Good coverage of common tactical/strategic themes
- Free and already downloaded
- ~2,000+ games with rich annotations

**Challenges to address:**
- Annotation quality varies (casual players, not titled annotators)
- Concepts mentioned may refer to *threats* not actual positions
- Temporal alignment (comment may describe previous/future move)
- Sparse labels (not every position is annotated)

**Recommendation:** Start with clear, unambiguous tactical concepts (fork, pin) before expanding to abstract strategic concepts (initiative, compensation).
