# Chess Concept Labeling Pipeline - Implementation Plan

Building a dataset of chess positions and labels, based on https://huggingface.co/datasets/Waterhorse/chess_data/blob/main/chessclip_data/annotated_pgn/annotated_pgn_free.tar.gz and https://arxiv.org/pdf/2410.20811

**Status:** 🔄 Phase 2 Refinement In Progress

## Implementation Summary

The pipeline has been implemented with a two-phase architecture:

**Phase 1 (✅ Completed):** Broad Regex Labeling
- **Single CLI command** for parsing + labeling (instead of separate commands)
- **Flat module structure** in `chess_sandbox/concept_labelling/`
- **Python-based patterns** (instead of YAML config)
- **Comprehensive doctests** in all modules
- **Integration test** covering the full pipeline

**Phase 2 (🔄 In Progress):** LLM-Based Precision Refinement
- **Lightweight LLM** (gpt-4o-mini) to validate regex matches
- **Temporal context extraction** (actual vs threat vs hypothetical)
- **False positive filtering** (e.g., "material" ≠ "mate")
- **Pydantic structured output** for reliable parsing

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
├── models.py           # Data models (LabelledPosition, ConceptRefinement)
├── patterns.py         # Concept regex patterns (Python constants)
├── parser.py           # PGN parsing & position extraction
├── labeller.py         # Phase 1: Broad regex concept detection
├── refiner.py          # Phase 2: LLM precision validation (NEW)
├── pipeline.py         # Main CLI (parse + label + optional refine)
└── lichess_export.py   # Lichess study export
```

### Data Flow
```
data/raw/annotated_pgn_free/gameknot/*.pgn
  ↓ pipeline.py (Phase 1: Regex)
data/processed/concept_labelling/positions_labeled.jsonl
  ↓ pipeline.py --refine-with-llm (Phase 2: LLM validation)
data/processed/concept_labelling/positions_refined.jsonl
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
    fen: str                         # Position in FEN notation
    move_number: int                 # Full move number
    side_to_move: str               # 'white' or 'black'
    comment: str                    # Associated annotation text
    game_id: str                    # Source game identifier
    concepts_raw: List[str]         # Phase 1: Broad regex matches
    concepts_validated: List[str]   # Phase 2: LLM-validated concepts
    temporal_context: Dict[str, str] # Phase 2: 'actual'|'threat'|'hypothetical'|'past'
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

## Phase 2: Concept Labeling

### Phase 2a: Broad Regex Detection (✅ Implemented)

**Strategy:** Keep patterns broad for high recall, accept lower precision

**Libraries:**
- **re**: Regex pattern matching

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

**Observed Issues (from initial run on 91 positions):**
- 17/91 positions labeled (18.7% recall)
- False positives: "material" matched by `\bmate\b` pattern
- Temporal misalignment: 14/17 labels describe threats/hypotheticals, not actual position
- Only 2/17 labels clearly describe current position state

### Phase 2b: LLM Precision Refinement (🔄 In Progress)

**Strategy:** Use lightweight LLM to validate regex matches and extract temporal context

**Libraries:**
- **openai**: GPT-4o-mini API
- **pydantic**: Structured output parsing

**Module:** `chess_sandbox/concept_labelling/refiner.py`

**Architecture:** Following `chess_sandbox/commentator.py` pattern

```python
class ConceptRefinement(BaseModel):
    """LLM-validated concept labels with temporal context."""

    validated_concepts: list[str] = Field(
        description="Concepts that ACTUALLY exist in the current position"
    )
    temporal_context: dict[str, str] = Field(
        description="For each concept: 'actual', 'threat', 'past', or 'hypothetical'"
    )
    false_positives: list[str] = Field(
        description="Regex matches that are incorrect (e.g., 'material' as 'mate')"
    )
    reasoning: str = Field(
        description="Brief explanation of validation decisions"
    )

@dataclass
class Refiner:
    """Validates regex concept matches using LLM."""

    PROMPT = '''
    You are a chess expert validating concept labels from game annotations.

    POSITION: Move {move_number}, {side_to_move} to move
    FEN: {fen}
    COMMENT: "{comment}"
    REGEX DETECTED: {concepts_raw}

    For each detected concept:
    1. Is it a FALSE POSITIVE? (e.g., "material" wrongly matched as "mate")
    2. What is the TEMPORAL CONTEXT?
       - 'actual': Concept exists in the current position NOW
       - 'threat': Concept is threatened/possible in future moves
       - 'hypothetical': Discussing "if/could/would" scenarios
       - 'past': Referring to previous moves that already happened

    Only validate concepts clearly mentioned in the comment.
    '''

    llm_model: str
    client: OpenAI

    def refine(self, position: LabelledPosition) -> ConceptRefinement:
        prompt = self.PROMPT.format(
            move_number=position.move_number,
            side_to_move=position.side_to_move,
            fen=position.fen,
            comment=position.comment,
            concepts_raw=position.concepts_raw
        )

        response = self.client.responses.parse(
            model=self.llm_model,
            input=prompt,
            text_format=ConceptRefinement
        )

        # Extract parsed output (similar to commentator.py)
        ...
```

**Expected Improvements:**
- **Precision**: 50% → 90%+ (eliminate false positives)
- **Temporal accuracy**: Automatic classification without brittle rules
- **Training quality**: Filter to `temporal_context == 'actual'` for high-precision dataset
- **Cost**: ~$0.001/position × 100k positions = ~$100 (acceptable)

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
# Phase 1: Regex labeling (fast, broad recall)
uv run python -m chess_sandbox.concept_labelling.pipeline \
  --input-dir data/raw/annotated_pgn_free/gameknot \
  --output data/processed/concept_labelling/positions_labeled.jsonl \
  --stats data/processed/concept_labelling/concept_stats.json

# Phase 2: LLM refinement (slower, high precision)
uv run python -m chess_sandbox.concept_labelling.pipeline \
  --input-dir data/raw/annotated_pgn_free/gameknot \
  --output data/processed/concept_labelling/positions_refined.jsonl \
  --refine-with-llm \
  --llm-model gpt-4o-mini

# Export 64 samples for review (with temporal filtering)
uv run python -m chess_sandbox.concept_labelling.lichess_export \
  --input data/processed/concept_labelling/positions_refined.jsonl \
  --output-pgn data/exports/lichess_study_sample.pgn \
  --n-samples 64 \
  --filter-temporal actual  # Only positions with concepts in ACTUAL state
```

---

## Directory Structure (✅ Implemented)
```
chess_sandbox/concept_labelling/    # Main package
├── __init__.py
├── models.py                      # LabelledPosition + ConceptRefinement
├── patterns.py                    # Concept regex patterns
├── parser.py                      # PGN parsing
├── labeller.py                    # Phase 1: Regex concept detection
├── refiner.py                     # Phase 2: LLM validation (NEW)
├── pipeline.py                    # Main CLI (parse + label + refine)
└── lichess_export.py              # PGN export for Lichess

data/
├── raw/annotated_pgn_free/gameknot/  # Input PGNs (12,769 files)
├── processed/concept_labelling/
│   ├── positions_labeled.jsonl   # Phase 1: Regex output
│   ├── positions_refined.jsonl   # Phase 2: LLM validated (NEW)
│   ├── concept_stats.json        # Concept distribution stats
│   └── refinement_report.json    # Precision metrics (NEW)
└── exports/
    └── lichess_study_sample.pgn  # Lichess-importable PGN

tests/
├── test_integration.py           # Full pipeline integration test
└── test_refiner.py               # LLM refinement tests (NEW)
```

---

## Next Steps

**Immediate (Phase 2 Refinement):**
1. ✅ Analyze first-pass regex results (91 positions)
2. 🔄 Implement `Refiner` class with LLM validation
3. 🔄 Test refinement on 100 positions, measure precision improvement
4. 🔄 Run full refinement on all labeled positions
5. 🔄 Generate precision/recall metrics report

**Post-Refinement:**
- Review Lichess study of high-precision labels (temporal_context='actual')
- Build train/val/test splits with temporal filtering options
- Create data loader for neural network training
- Experiment with concept co-occurrence patterns

---

## Key Considerations

**Dataset strengths:**
- Natural language annotations (human-interpretable concepts)
- Good coverage of common tactical/strategic themes
- Free and already downloaded
- ~2,000+ games with rich annotations

**Challenges & Solutions:**

| Challenge | Solution |
|-----------|----------|
| Annotation quality varies | Use LLM to validate concept appropriateness |
| Concepts may refer to threats not actual positions | **Phase 2b: Temporal context extraction** |
| Temporal alignment issues (14/17 labels are mixed) | **LLM classification: actual/threat/hypothetical/past** |
| Sparse labels (18.7% coverage) | Keep broad regex, use LLM to improve precision not recall |
| False positives ("material" → "mate") | **LLM-based false positive filtering** |

**Recommendation:**
- Phase 1: Keep regex patterns **broad** (high recall, ~50% precision)
- Phase 2: Use **LLM refinement** (target 90%+ precision)
- Training: Filter to `temporal_context == 'actual'` for highest quality dataset
