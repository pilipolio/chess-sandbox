# Gameknot Crawler: Improvement Suggestions

## Executive Summary

The `gameknot_crawler` module scrapes annotated chess games from gameknot.com. Current implementation uses PyQt5/QWebEngine (heavy browser stack) but **chess diagrams are already in static HTML**. This document proposes practical improvements focused on simplicity, maintainability, and modern Python practices.

---

## Complete Data Flow Analysis

### Current Pipeline Overview

```
Phase 0: Prerequisites
  └─ saved_links.p (11,578 URLs) + extra_pages.p (pagination counts)

Phase 1: Web Scraping (run_all.py + save_rendered_webpage.py)
  └─ HTML files: saved_files/html/saved{i}.html

Phase 2: HTML Parsing (main.py html_parser)
  └─ Pickle objects: outputs/saved{i}.obj

Phase 3: Preprocessing (preprocess.py train|valid|test)
  └─ Parallel corpus: {split}.che-eng.{single|multi}.{che|en}

Phase 4: Data Conversion (data_converter.py)
  └─ Final pickle: train_single.pkl
```

### Detailed Phase Breakdown

#### Phase 0: Prerequisites (Pre-existing Data)
**Input Files:**
- `saved_files/saved_links.p` - 11,578 URLs to gameknot.com games (pickled list)
- `saved_files/extra_pages.p` - Pagination count per game (pickled list, 0 = 1 page, 2 = 3 pages)

**Note:** These files are provided in the repo for reproducibility since gameknot.com content changes over time.

---

#### Phase 1: Web Scraping
**Script:** `run_all.py`

**Purpose:** Orchestrate parallel HTML downloads

**Command:**
```bash
python run_all.py 0 11577 1  # start_idx, end_idx, max_processes
```

**Process Flow:**
1. Load `saved_links.p` and `extra_pages.p`
2. For each game index `i` from `start` to `end`:
   - Get pagination count from `extra_pages[i]`
   - Spawn subprocess for each page: `save_rendered_webpage.py -i {i} -num {page_num}`
3. Manage subprocess pool (throttle at `max_processes`)
4. Use `os.wait()` to control parallelism

**Output:**
- `saved_files/html/saved{i}.html` (main pages)
- `saved_files/html/saved{i}_{j}.html` (pagination pages where j > 0)

---

**Script:** `save_rendered_webpage.py`

**Current Implementation:**
```python
# Uses PyQt5 + QWebEngineView to render pages
# JavaScript is DISABLED: JavascriptEnabled = False
# Runs in headless mode (offscreen rendering)
```

**What it does:**
1. Construct URL: `url = all_links[i]` (add `&pg={page_num}` if paginated)
2. Load URL in QWebEngineView
3. Wait for `loadFinished` signal
4. Extract HTML via `page.toHtml()`
5. Save to `saved_files/html/saved{i}[_{page_num}].html`

**HTML Structure (Actual Content):**
```html
<div data-chess-diagram="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3|||last=e2e4"></div>
<td>Opening move comment text here</td>
```

**Key Finding:** Chess diagrams are in **static HTML attributes** (`data-chess-diagram`), not JavaScript-rendered. PyQt5 is overkill!

---

#### Phase 2: HTML Parsing
**Script:** `main.py html_parser`

**Purpose:** Extract structured chess data from HTML files

**Command:**
```bash
python main.py html_parser
```

**Process Flow:**
1. List all `.html` files in `saved_files/html/`
2. For each HTML file:
   - Parse with BeautifulSoup (`lxml` parser)
   - Find `<table class="dialog">` (game annotation table)
   - Extract every other `<tr>` row (skip alternating rows)
   - For each valid row:
     - **Column 0**: Extract move text + FEN from `data-chess-diagram` attribute
     - **Column 1**: Extract commentary text
   - Build list: `[[move, fen_with_metadata, comment], ...]`
3. Pickle list to `outputs/saved{i}.obj`

**Output Format (`.obj` files):**
```python
[
  ["1. e4", "rnbqkbnr/.../|||last=e2e4", "Opening with King's pawn"],
  ["1... e5", "rnbqkbnr/.../|||last=e7e5", "Symmetric response"],
  ...
]
```

**Error Handling:** Failed files logged to `error_files.txt`

---

#### Phase 3: Preprocessing
**Script:** `preprocess.py {train|valid|test}`

**Purpose:** Convert `.obj` files into parallel corpus format for NMT/LLM training

**Command:**
```bash
python preprocess.py train
python preprocess.py valid
python preprocess.py test
```

**Process Flow:**
1. Load appropriate split links: `{split}_links.p`
2. For each game in split:
   - Load corresponding `.obj` file(s) from `outputs/`
   - Track board state progression (startState → currentState)
   - For each move+comment:
     - Parse move notation (e.g., "Nf3" → "white _knight f3 <EOM>")
     - Extract previous FEN, current FEN, move sequence
     - Tokenize commentary with NLTK
     - Format as parallel corpus entry
3. Write to files:
   - **Single-move**: `{split}.che-eng.single.{che|en}`
   - **Multi-move**: `{split}.che-eng.multi.{che|en}`

**Output Format:**

`.che` file (Chess/Source):
```
{current_FEN} <EOC> {previous_FEN} <EOP> {parsed_moves} <EOMH> {raw_move}
```

`.en` file (English/Target):
```
Tokenized commentary text
```

**Move Parsing Example:**
- Input: `"23. Nh6+"`
- Output: `"white _knight h6 <EOM>"`
- Captures: `"white _queen X d5 <EOM>"` (X = capture marker)
- Special handling: checks (+), checkmates (#), castling, en passant

**Single vs Multi:**
- **Single**: Positions with exactly 1 move (used for fine-grained training)
- **Multi**: Positions with move sequences (for context-aware training)

---

#### Phase 4: Data Conversion
**Script:** `data_converter.py`

**Purpose:** Convert parallel corpus to structured pickle for concept extraction phase

**Command:**
```bash
python data_converter.py  # hardcoded to process train.che-eng.single.*
```

**Process Flow:**
1. Read `train.che-eng.single.che` and `train.che-eng.single.en`
2. For each line pair:
   - Parse `.che` line to extract:
     - `start_fen = previous_board + " 0 0"`
     - `end_fen = current_board + " 0 0"`
     - `move_sequence` (parsed moves)
   - Get comment from `.en` line
3. Build tuple: `(start_fen, end_fen, move_sequence, comment)`
4. Pickle list to `saved_files/train_single.pkl`

**Output Format:**
```python
[
  ("rnbqkbnr/... w KQkq - 0 0", "rnbqkbnr/... b KQkq e3 0 0", "white _pawn e4", "King's pawn opening"),
  ...
]
```

---

## Critical Issues Identified

### 1. PyQt5 is Unnecessary (HIGH PRIORITY) 🔴

**Problem:**
- Current implementation uses QWebEngineView (Chromium-based browser)
- JavaScript is **disabled** in the scraper
- Chess diagrams are **already in static HTML** as `data-chess-diagram` attributes
- PyQt5 adds ~200MB of dependencies and requires display server

**Evidence:**
```html
<!-- Actual HTML from saved14.html -->
<div data-chess-diagram="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3|||last=e2e4"></div>
```

**Impact:**
- Slow: Browser startup overhead for each page
- Heavy: PyQt5, PyQtWebEngine, Qt dependencies
- Complex: Requires X11/display setup on servers
- Fragile: Qt version compatibility issues

**Solution:** Replace with `requests` library (see recommendations below)

---

### 2. File Format Issues 🟡

**Problems:**
- **11,578+ separate pickle files** in `outputs/` directory
- No easy way to inspect data (binary format)
- Split information stored separately (train/valid/test links)
- Hard to version control or share dataset
- Memory issues when loading all games

**Impact:**
- Poor data portability
- Difficult to debug or validate
- Can't stream process large datasets
- No standard tooling support

---

### 3. Configuration Management 🟡

**Problems:**
- Hardcoded paths throughout: `"./saved_files/"`, `"./outputs/"`
- No environment variable support
- Magic constants scattered in code
- Different scripts have different assumptions

**Examples:**
```python
# In save_rendered_webpage.py
all_links = pickle.load(open("./saved_files/saved_links.p", "rb"))

# In main.py
self._data_path = "./saved_files/html/"
self._destination_path = "./outputs/"

# In preprocess.py
all_links = pickle.load(open("./saved_files/train_links.p", "rb"))
```

---

### 4. Code Organization 🟡

**Current Structure Issues:**
- Responsibilities scattered across 7+ files
- Duplicate utilities (HTML parsing, file I/O)
- No clear entry point
- Inconsistent naming conventions
- Mix of classes and procedural code

**Files:**
- `run_all.py` - Process orchestrator
- `save_rendered_webpage.py` - Scraping (Qt-based)
- `main.py` - HTML parsing + mode selection
- `preprocess.py` - Data formatting
- `data_converter.py` - Final conversion
- `utilities.py` - HTML helpers
- `boardUpdater.py` - Board state (partially used)
- `getGameLinks.py`, `trainTestSplit.py` - Data prep (already done)

---

### 5. Error Handling 🟡

**Problems:**
- No retry logic for failed HTTP requests
- No rate limiting (could trigger anti-scraping)
- Errors logged but pipeline continues
- No manifest of failed games
- Silent failures possible

**Example Issues:**
- Many HTML files are 404 errors (see saved100.html, saved630.html)
- No validation that scraping succeeded
- Failed parses written to `error_files.txt` but not tracked

---

### 6. Data Quality 🟢 (Acceptable)

**Current State:**
- Raw HTML parsing (no validation)
- Move parsing is string manipulation (fragile but works)
- No FEN validation
- Assumes gameknot.com structure is stable

**Decision:** Keep as-is. Validation adds complexity without much benefit for research code.

---

## Proposed Improvements

### 1. Replace PyQt5 with `requests` ✅

**Benefits:**
- 10-50x faster execution
- Remove ~200MB of dependencies
- Works on headless servers
- Simpler code (10 lines vs 95 lines)
- Easier to add retry logic and rate limiting

**Implementation:**

```python
# gameknot_crawler/scrape.py
import requests
import time
from pathlib import Path
from config import (
    SAVED_LINKS_PATH, EXTRA_PAGES_PATH, HTML_DIR,
    USER_AGENT, REQUEST_TIMEOUT, MAX_RETRIES, RATE_LIMIT_DELAY
)
import logging

logger = logging.getLogger(__name__)

def fetch_game_page(url: str, game_id: int, page_num: int = 0) -> bool:
    """Fetch a single game page and save HTML."""

    # Add pagination parameter if needed
    if page_num > 0:
        url = f"{url}&pg={page_num}"

    # Retry logic
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(
                url,
                headers={'User-Agent': USER_AGENT},
                timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()

            # Save HTML
            filename = f"saved{game_id}.html" if page_num == 0 else f"saved{game_id}_{page_num}.html"
            output_path = HTML_DIR / filename
            output_path.write_text(response.text, encoding='utf-8')

            logger.info(f"Saved {filename}")

            # Rate limiting
            time.sleep(RATE_LIMIT_DELAY)
            return True

        except requests.RequestException as e:
            logger.warning(f"Attempt {attempt + 1}/{MAX_RETRIES} failed for game {game_id} page {page_num}: {e}")
            time.sleep(2 ** attempt)  # Exponential backoff

    logger.error(f"Failed to fetch game {game_id} page {page_num} after {MAX_RETRIES} attempts")
    return False
```

**Migration Path:**
1. Keep old `save_rendered_webpage.py` as fallback
2. Test `requests` version on 100 games
3. Compare HTML output (should be identical)
4. Switch to `requests` by default

---

### 2. Consolidate to JSONL Format ✅

**Benefits:**
- Single file per split (easy to version/share)
- Human-readable (can inspect with text editor)
- Streaming support (process line-by-line)
- Standard tooling (jq, pandas, HuggingFace datasets)
- Split metadata embedded in each record

**Format:**
```jsonl
{"game_id": 14, "page": 0, "move_num": 1, "fen_before": "rnbq...", "fen_after": "rnbq...", "move": "e4", "move_parsed": "white _pawn e4", "comment": "King's pawn opening", "split": "train"}
{"game_id": 14, "page": 0, "move_num": 2, "fen_before": "rnbq...", "fen_after": "rnbq...", "move": "e5", "move_parsed": "black _pawn e5", "comment": "Symmetric response", "split": "train"}
```

**Implementation:**

```python
# gameknot_crawler/transform.py
import jsonlines
from pathlib import Path
from config import OUTPUT_DIR

def write_jsonl_dataset(games_data: list, split: str, output_dir: Path):
    """Write games data to JSONL format."""

    output_file = output_dir / f"gameknot_{split}.jsonl"

    with jsonlines.open(output_file, mode='w') as writer:
        for game in games_data:
            game_id = game['game_id']
            page_num = game.get('page', 0)

            for move_num, (fen_before, fen_after, move_parsed, comment) in enumerate(game['moves'], start=1):
                record = {
                    'game_id': game_id,
                    'page': page_num,
                    'move_num': move_num,
                    'fen_before': fen_before,
                    'fen_after': fen_after,
                    'move': move_parsed,  # Original notation
                    'comment': comment,
                    'split': split
                }
                writer.write(record)

    logger.info(f"Wrote {output_file}")
```

**Backward Compatibility:**
- Keep pickle output as option (`OUTPUT_FORMAT` in config)
- Provide conversion script: `convert_pkl_to_jsonl.py`

---

### 3. Centralized Configuration ✅

**Implementation:**

```python
# gameknot_crawler/config.py
"""Centralized configuration for gameknot_crawler."""
import os
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent
SAVED_FILES_DIR = Path(os.getenv("GK_SAVED_FILES", PROJECT_ROOT / "saved_files"))
HTML_DIR = Path(os.getenv("GK_HTML_DIR", SAVED_FILES_DIR / "html"))
OUTPUT_DIR = Path(os.getenv("GK_OUTPUT_DIR", PROJECT_ROOT / "outputs"))

# Ensure directories exist
HTML_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Input files
SAVED_LINKS_PATH = SAVED_FILES_DIR / "saved_links.p"
EXTRA_PAGES_PATH = SAVED_FILES_DIR / "extra_pages.p"
TRAIN_LINKS_PATH = SAVED_FILES_DIR / "train_links.p"
VALID_LINKS_PATH = SAVED_FILES_DIR / "valid_links.p"
TEST_LINKS_PATH = SAVED_FILES_DIR / "test_links.p"

# Scraping configuration
USER_AGENT = os.getenv(
    "GK_USER_AGENT",
    "Mozilla/5.0 (compatible; GKCrawler/1.0; +https://github.com/yourrepo)"
)
REQUEST_TIMEOUT = int(os.getenv("GK_TIMEOUT", "30"))  # seconds
MAX_RETRIES = int(os.getenv("GK_MAX_RETRIES", "3"))
RATE_LIMIT_DELAY = float(os.getenv("GK_RATE_LIMIT", "1.0"))  # seconds between requests

# Output configuration
OUTPUT_FORMAT = os.getenv("GK_OUTPUT_FORMAT", "jsonl")  # "jsonl" or "pickle"
INCLUDE_MULTI_MOVE = os.getenv("GK_INCLUDE_MULTI", "false").lower() == "true"

# Logging
LOG_LEVEL = os.getenv("GK_LOG_LEVEL", "INFO")
LOG_FILE = OUTPUT_DIR / "crawler.log"
```

**Usage:**
```python
# In any script
from config import HTML_DIR, SAVED_LINKS_PATH, REQUEST_TIMEOUT

html_path = HTML_DIR / f"saved{game_id}.html"
links = pickle.load(open(SAVED_LINKS_PATH, "rb"))
```

**Environment Variable Support:**
```bash
export GK_SAVED_FILES=/data/gameknot
export GK_RATE_LIMIT=2.0
export GK_OUTPUT_FORMAT=jsonl
python -m gameknot_crawler.cli scrape
```

---

### 4. Module Structure Refactoring ✅

**Proposed Structure:**

```
gameknot_crawler/
├── __init__.py
├── config.py              # Centralized configuration
├── scrape.py              # Scraping logic (requests-based)
├── parse.py               # HTML parsing (BeautifulSoup)
├── transform.py           # Data transformation to final format
├── cli.py                 # Command-line interface
│
├── utils/
│   ├── __init__.py
│   ├── html_utils.py      # BeautifulSoup helpers
│   ├── chess_utils.py     # Move parsing, FEN utilities
│   └── file_utils.py      # Pickle loading, file I/O
│
├── legacy/                # Old scripts (for reference)
│   ├── run_all.py
│   ├── save_rendered_webpage.py
│   ├── main.py
│   └── preprocess.py
│
└── saved_files/           # Data directory (not in git)
    ├── saved_links.p
    ├── extra_pages.p
    ├── train_links.p
    ├── valid_links.p
    ├── test_links.p
    └── html/
```

**Responsibilities:**

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `scrape.py` | Fetch HTML pages | `fetch_game_page()`, `scrape_games()` |
| `parse.py` | Extract from HTML | `parse_game_html()`, `extract_moves()` |
| `transform.py` | Format output | `write_jsonl_dataset()`, `format_parallel_corpus()` |
| `utils/chess_utils.py` | Chess operations | `parse_move_notation()`, `validate_fen()` |
| `utils/html_utils.py` | HTML helpers | `get_chess_diagram()`, `extract_comment()` |
| `cli.py` | Command interface | `scrape_cmd()`, `parse_cmd()`, `transform_cmd()` |

---

### 5. Improved Error Handling & Logging ✅

**Logging Setup:**

```python
# gameknot_crawler/utils/logging_utils.py
import logging
from config import LOG_FILE, LOG_LEVEL

def setup_logging(name: str) -> logging.Logger:
    """Setup logging with file and console handlers."""

    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)

    # File handler
    fh = logging.FileHandler(LOG_FILE)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(
        '%(levelname)s: %(message)s'
    ))

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger
```

**Error Tracking:**

```python
# gameknot_crawler/scrape.py
from collections import defaultdict

class ScrapeStats:
    """Track scraping statistics."""
    def __init__(self):
        self.success = 0
        self.failed = []
        self.errors = defaultdict(int)

    def record_success(self, game_id: int):
        self.success += 1

    def record_failure(self, game_id: int, error: str):
        self.failed.append(game_id)
        self.errors[error] += 1

    def save_manifest(self, output_path: Path):
        """Save failed games manifest."""
        with open(output_path, 'w') as f:
            f.write(f"Success: {self.success}\n")
            f.write(f"Failed: {len(self.failed)}\n\n")
            f.write("Error Summary:\n")
            for error, count in self.errors.items():
                f.write(f"  {error}: {count}\n")
            f.write("\nFailed Game IDs:\n")
            f.write("\n".join(map(str, self.failed)))
```

---

### 6. No Validation (By Design) ✅

**Rationale:**
- Keep implementation simple
- Trust source data (gameknot.com is established site)
- Validation adds dependencies (python-chess) and complexity
- Errors will be caught in downstream processing

**What to Skip:**
- ❌ FEN string validation
- ❌ Legal move checking
- ❌ Board state verification
- ❌ Comment text validation

**Minimal Checks Only:**
- ✅ Check if `data-chess-diagram` attribute exists
- ✅ Check if comment is not empty string
- ✅ Log warnings for missing data (don't crash)
- ✅ Record which games had parsing issues

**Implementation:**
```python
def parse_move_entry(row) -> Optional[dict]:
    """Parse a single move entry, return None if invalid."""
    try:
        diagram = row.find('div', {'data-chess-diagram': True})
        if not diagram:
            logger.warning("Missing chess diagram")
            return None

        fen = diagram['data-chess-diagram']
        if not fen:
            logger.warning("Empty FEN string")
            return None

        # Extract comment (don't validate content)
        comment_td = row.find_all('td')[1]
        comment = comment_td.get_text(strip=True)

        return {'fen': fen, 'comment': comment}

    except Exception as e:
        logger.warning(f"Parse error: {e}")
        return None
```

---

## Proposed CLI Interface

```python
# gameknot_crawler/cli.py
import argparse
from scrape import scrape_games
from parse import parse_all_html
from transform import transform_to_jsonl

def main():
    parser = argparse.ArgumentParser(description='GameKnot Chess Game Crawler')
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Scrape command
    scrape_parser = subparsers.add_parser('scrape', help='Scrape HTML pages')
    scrape_parser.add_argument('--start', type=int, default=0, help='Start game index')
    scrape_parser.add_argument('--end', type=int, default=-1, help='End game index (-1 = all)')
    scrape_parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')

    # Parse command
    parse_parser = subparsers.add_parser('parse', help='Parse HTML to intermediate format')
    parse_parser.add_argument('--input-dir', type=str, help='Override HTML input directory')

    # Transform command
    transform_parser = subparsers.add_parser('transform', help='Transform to final JSONL format')
    transform_parser.add_argument('--split', required=True, choices=['train', 'valid', 'test'])
    transform_parser.add_argument('--format', choices=['jsonl', 'pickle'], default='jsonl')

    # Full pipeline
    pipeline_parser = subparsers.add_parser('run-all', help='Run full pipeline')
    pipeline_parser.add_argument('--split', required=True, choices=['train', 'valid', 'test'])

    args = parser.parse_args()

    if args.command == 'scrape':
        scrape_games(args.start, args.end, args.workers)
    elif args.command == 'parse':
        parse_all_html(args.input_dir)
    elif args.command == 'transform':
        transform_to_jsonl(args.split, args.format)
    elif args.command == 'run-all':
        # Run full pipeline
        parse_all_html()
        transform_to_jsonl(args.split, 'jsonl')
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
```

**Usage Examples:**

```bash
# Scrape specific range with 4 workers
python -m gameknot_crawler.cli scrape --start 0 --end 100 --workers 4

# Parse all HTML files
python -m gameknot_crawler.cli parse

# Transform train split to JSONL
python -m gameknot_crawler.cli transform --split train --format jsonl

# Run full pipeline for test set
python -m gameknot_crawler.cli run-all --split test
```

---

## Implementation Roadmap

### Phase 1: Core Refactoring (2-3 hours)

**Goal:** Replace PyQt5 and centralize configuration

**Tasks:**
1. Create `gameknot_crawler/config.py`
2. Create `gameknot_crawler/scrape.py` with `requests`
3. Add basic logging setup
4. Test on 10 games, compare HTML output with original
5. Verify parse.py works with new HTML files

**Success Criteria:**
- ✅ HTML files identical to PyQt5 version
- ✅ 10x faster execution
- ✅ No Qt dependencies needed

---

### Phase 2: Data Format & Structure (2-3 hours)

**Goal:** JSONL output and module reorganization

**Tasks:**
1. Create `gameknot_crawler/parse.py` (refactor from main.py)
2. Create `gameknot_crawler/transform.py` with JSONL output
3. Create `utils/` directory with helper modules
4. Test full pipeline: scrape → parse → transform → JSONL
5. Validate JSONL format with sample loading

**Success Criteria:**
- ✅ JSONL files loadable with `jsonlines` and `pandas`
- ✅ Split metadata embedded in each record
- ✅ Single file per split (easier distribution)

---

### Phase 3: CLI & Polish (1-2 hours)

**Goal:** User-friendly interface and documentation

**Tasks:**
1. Create `gameknot_crawler/cli.py`
2. Add error tracking and manifests
3. Add retry logic with exponential backoff
4. Add rate limiting (configurable delay)
5. Write usage examples in README
6. Move old scripts to `legacy/` directory

**Success Criteria:**
- ✅ Single command to run full pipeline
- ✅ Failed games tracked in manifest
- ✅ Logs written to file + console
- ✅ Environment variable configuration works

---

## Testing Strategy

### Unit Tests (Optional but Recommended)

```python
# tests/test_parse.py
def test_parse_chess_diagram():
    html = '<div data-chess-diagram="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3|||last=e2e4"></div>'
    soup = BeautifulSoup(html, 'html.parser')
    fen = extract_fen(soup)
    assert fen.startswith("rnbqkbnr/pppppppp")

def test_parse_move_notation():
    assert parse_move_notation("Nf3", "white") == "white _knight f3"
    assert parse_move_notation("exd5", "black") == "black _pawn X d5"
```

### Integration Tests

```bash
# Test on small dataset
python -m gameknot_crawler.cli scrape --start 0 --end 10
python -m gameknot_crawler.cli parse
python -m gameknot_crawler.cli transform --split train

# Verify output
python -c "import jsonlines; print(len(list(jsonlines.open('outputs/gameknot_train.jsonl'))))"
```

---

## Migration Plan

### Step 1: Parallel Development
- Keep existing code in `legacy/` directory
- Build new modules alongside
- Test on subset of data (games 0-100)

### Step 2: Validation
- Compare outputs: old pickle vs new JSONL
- Verify identical FEN strings and comments
- Check move parsing consistency

### Step 3: Switchover
- Update documentation
- Set new scripts as default
- Keep legacy scripts for 1 release cycle

### Step 4: Cleanup
- Remove Qt dependencies from `requirements.txt`
- Archive legacy code
- Update CI/CD if applicable

---

## Dependencies Changes

### Current Requirements
```
PyQt5
PyQt5-tools
PyQtWebEngine
lxml
nltk
pyyaml
jsonlines
tqdm
beautifulsoup4
```

### Proposed Requirements
```
requests          # HTTP client (replaces PyQt5)
beautifulsoup4    # HTML parsing (keep)
lxml              # BS4 parser (keep)
nltk              # Tokenization (keep)
jsonlines         # JSONL format (new)
tqdm              # Progress bars (keep)
```

**Removed:**
- PyQt5 (~200MB)
- PyQt5-tools
- PyQtWebEngine

**Added:**
- jsonlines (lightweight)

**Net change:** -197MB dependencies, +1MB

---

## FAQ

### Q: Why not validate FEN strings with python-chess?
**A:** Adds complexity and dependency without much benefit. Gameknot.com data is trusted, and invalid entries will fail downstream anyway.

### Q: Should we keep pickle format as option?
**A:** Yes, for backward compatibility. Make JSONL the default, but allow pickle via `--format pickle` flag.

### Q: What about existing .obj files in outputs/?
**A:** Keep them. Provide conversion script: `python -m gameknot_crawler.tools.convert_obj_to_jsonl`

### Q: Will requests handle all edge cases PyQt5 did?
**A:** Yes. The current implementation disables JavaScript, so PyQt5 is just fetching static HTML. `requests` does the same thing faster.

### Q: How to handle gameknot.com rate limiting?
**A:** Use `RATE_LIMIT_DELAY` config (default 1 second between requests). Add exponential backoff on errors. Monitor for 429 status codes.

---

## Conclusion

The proposed improvements focus on **practical modernization** without over-engineering:

1. **Replace PyQt5 with requests** - 10x faster, simpler, lighter
2. **JSONL format** - Standard, inspectable, streamable
3. **Centralized config** - Environment variables, clear defaults
4. **Better structure** - Clear separation of scrape/parse/transform
5. **Improved logging** - Track errors, retry failures

**Estimated Effort:** 5-8 hours total
**Risk Level:** Low (incremental, testable changes)
**Backward Compatibility:** High (legacy scripts preserved)

Ready to implement! 🚀
