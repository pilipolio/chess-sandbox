# Modal Pipeline Execution for Chess Concept Labeling

## Overview

**Goal:** Run the chess concept labeling pipeline on Modal ephemeral apps, enabling serverless batch processing of PGN files with LLM-based concept validation.

**Scope:**
- Single Modal function wrapping existing pipeline logic
- Volume-based input/output for PGN files and JSONL results
- Secret management for OpenAI API key
- Reuses existing async parallelization (10 concurrent LLM calls via asyncio)
- Ephemeral execution model (no persistent deployments)

**Non-Goal:**
- Additional parallelization beyond existing asyncio (no Modal `.map()` or file-level parallelism)
- Cost estimation or monitoring
- Real-time API endpoints (use Modal CLI for batch processing)

## Architecture

### Simple Batch Processing Design

```
┌─────────────────┐
│  Local Machine  │
└────────┬────────┘
         │ 1. Upload PGN files
         ▼
┌─────────────────┐
│  Modal Volume   │
│  /pgn_inputs/   │
└────────┬────────┘
         │ 2. Mount volume
         ▼
┌─────────────────────────────────────┐
│       Modal Function                │
│  ┌───────────────────────────────┐  │
│  │ Parse PGN Directory           │  │
│  │   ↓                           │  │
│  │ Label Positions (Regex)       │  │
│  │   ↓                           │  │
│  │ Refine with LLM (Async)       │  │
│  │   • 10 concurrent API calls   │  │
│  │   • Semaphore-based limiting  │  │
│  │   ↓                           │  │
│  │ Write JSONL Output            │  │
│  └───────────────────────────────┘  │
└────────┬────────────────────────────┘
         │ 3. Write results
         ▼
┌─────────────────┐
│  Modal Volume   │
│  /outputs/      │
└────────┬────────┘
         │ 4. Download results
         ▼
┌─────────────────┐
│  Local Machine  │
└─────────────────┘
```

### Component Responsibilities

| Component | Responsibility |
|-----------|---------------|
| **Local Machine** | Upload input data, trigger Modal execution, download results |
| **Modal Volume** | Persistent storage for PGN inputs and JSONL outputs |
| **Modal Secret** | Secure injection of OPENAI_API_KEY into function environment |
| **Modal Function** | Stateless execution of pipeline logic (parse → label → refine → write) |
| **Existing Pipeline** | Core logic reused as-is (no modifications needed) |

## Key Documentation

- [Modal Ephemeral Apps](https://modal.com/docs/guide/apps#ephemeral-apps) - Running one-off batch jobs
- [Modal Volumes](https://modal.com/docs/guide/volumes) - Persistent file storage
- [Modal Secrets](https://modal.com/docs/guide/secrets) - Environment variable management
- [Modal Image Building](https://modal.com/docs/reference/modal.Image) - Container image configuration

## Code Reuse Strategy

### Design Considerations

The existing `pipeline.py::main()` function contains both business logic and CLI presentation (via Click). To avoid duplication between the CLI and Modal implementations, we have three architectural options:

### Option 1: Extract Core Logic (Strategic - Best Long-Term)

**Approach:** Create a new UI-agnostic `process_pipeline()` function containing all business logic. Both CLI and Modal call this function.

```python
# In pipeline.py
def process_pipeline(
    input_dir: Path,
    output: Path,
    limit: int | None = None,
    refine_with_llm: bool = False,
    llm_model: str = "gpt-5-nano",
    logger: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Core pipeline logic, UI-agnostic."""
    log = logger or (lambda msg: None)

    # Business logic: parse → label → refine → write
    # Returns structured statistics
```

```python
# CLI wrapper
@click.command()
def main(...) -> None:
    stats = process_pipeline(..., logger=click.echo)
    # Pretty-print stats for terminal
```

```python
# Modal wrapper
@app.function()
def modal_main(...) -> dict[str, Any]:
    stats = process_pipeline(..., logger=print)
    volume.commit()
    return stats
```

**Benefits:**
- ✅ Zero duplication of business logic
- ✅ Easy to test core logic without Click
- ✅ Both UIs can customize output/logging
- ✅ Returns structured data for programmatic use
- ✅ Clear separation of concerns

**Drawbacks:**
- ❌ Requires refactoring existing code
- ❌ More upfront effort
- ❌ Changes affect existing CLI (testing needed)

### Option 2: Wrap Click CLI from Modal (Tactical - Quick Validation)

**Approach:** Use Click's `CliRunner` to invoke the existing CLI from Modal. No changes to `pipeline.py`.

```python
# In modal_pipeline.py
@app.function(...)
def modal_main(...) -> dict[str, str]:
    from click.testing import CliRunner
    from chess_sandbox.concept_extraction.labelling.pipeline import main

    runner = CliRunner()
    result = runner.invoke(main, [
        '--input-dir', f"/data/{input_subdir}",
        '--output', f"/data/outputs/{output_filename}",
        '--limit', str(limit) if limit else '',
        '--refine-with-llm' if refine_with_llm else '',
        '--llm-model', llm_model,
    ])

    if result.exit_code != 0:
        raise RuntimeError(f"Pipeline failed: {result.output}")

    print(result.output)  # Forward CLI output to Modal logs
    volume.commit()

    return {"status": "success", "output": result.output}
```

**Benefits:**
- ✅ Zero code changes to existing pipeline
- ✅ Fast to implement and validate
- ✅ Click CLI remains authoritative
- ✅ Low risk (no refactoring)
- ✅ Can migrate to Option 1 later

**Drawbacks:**
- ❌ No structured return value (text output only)
- ❌ Harder to extract statistics programmatically
- ❌ Slightly less efficient (subprocess-like overhead)

### Option 3: Parameterize Logger in Existing main()

**Approach:** Make `click.echo` injectable in the existing `main()` function without full refactoring.

```python
def main(
    input_dir: Path,
    output: Path,
    ...,
    logger: Callable[[str], None] = click.echo,
) -> dict[str, Any]:
    logger(f"Parsing PGN files from: {input_dir}")
    # ... rest of logic
    return stats
```

**Benefits:**
- ✅ Moderate effort
- ✅ Returns structured data

**Drawbacks:**
- ❌ Still has Click decorator (must use CliRunner)
- ❌ Mixing concerns (business logic + CLI interface)
- ❌ Requires changes to existing code

### Chosen Approach: Option 2 (Tactical Validation)

**Decision:** Start with **Option 2** as a tactical approach to quickly validate Modal integration with zero changes to the existing pipeline code. This allows us to:

1. Test Modal volume mounting and secret management
2. Validate end-to-end execution in Modal environment
3. Confirm output correctness by comparing with CLI results
4. Gather real-world usage experience

**Migration Path:** Once validated, we can refactor to **Option 1** for better long-term maintainability if the Modal integration becomes a primary workflow.

## Implementation Steps

### Phase 1: Modal App Definition (Option 2 Implementation)

**File:** `chess_sandbox/concept_extraction.labelling/modal_pipeline.py`

**1.1 Build Modal Image**
```python
import modal

image = (
    modal.Image.debian_slim()
    .uv_sync(uv_project_dir="./", frozen=True)  # Install from pyproject.toml
    .add_local_python_source("chess_sandbox")   # Include our module
)
```

**Key Details:**
- Uses `uv_sync()` for reproducible dependency installation (consistent with existing `endpoints.py`)
- All dependencies from `pyproject.toml` automatically included:
  - `chess>=1.11.2` - PGN parsing
  - `click>=8.3.0` - Needed for CliRunner
  - `openai>=2.6.1` - LLM refinement
  - `httpx>=0.28.0` - HTTP client
  - `pydantic>=2.10.4` - Data models
- No system packages needed (pure Python implementation)

**1.2 Create Modal App**
```python
app = modal.App(name="chess-concept-pipeline", image=image)

volume = modal.Volume.from_name("chess-pgn-data", create_if_missing=True)
```

**1.3 Implement Pipeline Function (Wrapping Click CLI)**
```python
@app.function(
    timeout=3600,  # 1 hour for large batches
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("openai-secret")],
)
def process_pgn_batch(
    input_subdir: str,
    output_filename: str,
    limit: int | None = None,
    refine_with_llm: bool = False,
    llm_model: str = "gpt-5-nano",
) -> dict[str, str]:
    """Process PGN files from volume using Click CLI wrapper.

    Args:
        input_subdir: Subdirectory in /data containing PGN files
        output_filename: Output JSONL filename in /data/outputs/
        limit: Optional limit on number of PGN files to process
        refine_with_llm: Whether to use LLM for concept validation
        llm_model: LLM model to use (default: gpt-5-nano)

    Returns:
        Dictionary with status and CLI output text
    """
    from click.testing import CliRunner
    from chess_sandbox.concept_extraction.labelling.pipeline import main

    # Build CLI arguments
    input_dir = f"/data/{input_subdir}"
    output = f"/data/outputs/{output_filename}"

    args = [
        '--input-dir', input_dir,
        '--output', output,
    ]

    if limit is not None:
        args.extend(['--limit', str(limit)])

    if refine_with_llm:
        args.append('--refine-with-llm')

    args.extend(['--llm-model', llm_model])

    # Invoke Click CLI
    runner = CliRunner()
    result = runner.invoke(main, args)

    # Forward CLI output to Modal logs
    print(result.output)

    if result.exit_code != 0:
        error_msg = f"Pipeline failed with exit code {result.exit_code}"
        if result.exception:
            error_msg += f"\nException: {result.exception}"
        raise RuntimeError(error_msg)

    # Commit volume changes
    volume.commit()

    return {
        "status": "success",
        "exit_code": result.exit_code,
        "output": result.output,
    }
```

**1.4 Add Local Entrypoint**
```python
@app.local_entrypoint()
def main(
    input_subdir: str = "pgn_inputs/gameknot",
    output_filename: str = "labeled_positions.jsonl",
    limit: int | None = None,
    refine_with_llm: bool = False,
    llm_model: str = "gpt-5-nano",
):
    """Local entrypoint for running Modal pipeline."""
    result = process_pgn_batch.remote(
        input_subdir=input_subdir,
        output_filename=output_filename,
        limit=limit,
        refine_with_llm=refine_with_llm,
        llm_model=llm_model,
    )
    print(f"\nProcessing complete!")
    print(f"Status: {result['status']}")
    print(f"Output file: /data/outputs/{output_filename}")
```

### Modal Architecture: Functions vs Entrypoints

Understanding the distinction between `@app.function()` and `@app.local_entrypoint()` is crucial for Modal development:

#### Current Implementation: Local Entrypoint + Remote Function

```
┌─────────────────────────────────────┐
│      Your Local Machine             │
│                                     │
│  1. You run:                        │
│     modal run modal_pipeline.py::main │
│                                     │
│  2. main() executes locally         │
│     ├─ Prints "Starting..."         │
│     └─ Calls .remote()              │
│           ↓                         │
└───────────┼─────────────────────────┘
            │
            │ Network call
            ↓
┌─────────────────────────────────────┐
│      Modal Cloud                    │
│                                     │
│  3. process_pgn_batch() runs here   │
│     ├─ Mounts volume at /data       │
│     ├─ Injects OPENAI_API_KEY       │
│     ├─ Invokes pipeline.main()      │
│     ├─ Commits volume changes       │
│     └─ Returns result dict          │
│           ↓                         │
└───────────┼─────────────────────────┘
            │
            │ Returns result
            ↓
┌─────────────────────────────────────┐
│      Your Local Machine             │
│                                     │
│  4. main() continues locally        │
│     └─ Prints "Processing complete!"│
│                                     │
└─────────────────────────────────────┘
```

**Key Differences:**

| Aspect | `@app.function()` | `@app.local_entrypoint()` |
|--------|-------------------|---------------------------|
| **Execution Location** | Modal cloud (serverless containers) | Your local machine |
| **Purpose** | Heavy computation, processing | Orchestration, coordination |
| **Access** | Modal volumes, secrets, cloud resources | Local filesystem, calls remote functions |
| **Startup** | Cold start (container spin-up) | Immediate (local Python) |
| **Typical Use** | Process data, train models, call APIs | Parse arguments, call multiple functions, display results |

**Benefits of This Approach:**
- ✅ Clean CLI: `modal run modal_pipeline.py::main --input-subdir=...`
- ✅ Immediate feedback from local entrypoint (no cold start for orchestration)
- ✅ Can call multiple remote functions if needed
- ✅ Easy to add progress tracking and validation locally

#### Alternative: Remote Function Only (Simpler)

We could simplify by removing the local entrypoint:

```python
# Only one function (remote)
@app.function(
    timeout=3600,
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("openai-secret")],
)
def process_pgn_batch(...):
    """Process PGN files (runs in Modal cloud)."""
    # All logic here
    pass

# Usage (directly invoke remote function):
# modal run modal_pipeline.py::process_pgn_batch \
#   --input-subdir pgn_inputs/gameknot \
#   --output-filename output.jsonl
```

**Trade-offs:**

| Aspect | With Local Entrypoint | Without (Remote Only) |
|--------|----------------------|----------------------|
| **Complexity** | Two functions (orchestrator + worker) | One function (worker only) |
| **CLI Length** | Shorter command name (`::main`) | Longer command name (`::process_pgn_batch`) |
| **Flexibility** | Can call multiple functions | Single function only |
| **Startup Feedback** | Immediate local messages | Must wait for container startup |
| **Best For** | Multi-step workflows, user-facing tools | Simple single-function jobs |

**Decision:** Keep the local entrypoint for this implementation because:
1. Provides better user experience with immediate feedback
2. Cleaner CLI (`::main` vs `::process_pgn_batch`)
3. Establishes pattern for future multi-function workflows
4. Minimal complexity cost (just wraps `.remote()` call)

### Phase 2: Secret and Volume Setup

**2.1 Create OpenAI Secret**
```bash
# Create secret with OpenAI API key
modal secret create openai-secret OPENAI_API_KEY=sk-your-key-here
```

**Verification:**
```bash
# List secrets
modal secret list
```

**2.2 Create Volume**
```bash
# Create persistent volume for data
modal volume create chess-pgn-data
```

**2.3 Upload PGN Files**
```bash
# Upload local PGN directory to volume
modal volume put chess-pgn-data \
  ./data/raw/annotated_pgn_free/gameknot \
  /pgn_inputs/gameknot
```

**Verification:**
```bash
# List files in volume
modal volume ls chess-pgn-data /pgn_inputs/gameknot
```

### Phase 3: Testing

**3.1 Local Testing (Without Modal)**
```bash
# Test existing CLI pipeline first
uv run python -m chess_sandbox.concept_extraction.labelling.pipeline \
  --input-dir ./data/raw/annotated_pgn_free/gameknot \
  --output ./test_output.jsonl \
  --limit 5
```

**3.2 Modal Development Server**
```bash
# Start Modal dev server (runs locally, simulates Modal environment)
modal serve chess_sandbox/concept_extraction.labelling/modal_pipeline.py
```

**3.3 Small Batch Test**
```bash
# Test with 10 files, no LLM (fast validation)
modal run chess_sandbox/concept_extraction.labelling/modal_pipeline.py::main \
  --input-subdir pgn_inputs/gameknot \
  --output-filename test_10_files.jsonl \
  --limit 10
```

**3.4 LLM Integration Test**
```bash
# Test with 5 files and LLM refinement
modal run chess_sandbox/concept_extraction.labelling/modal_pipeline.py::main \
  --input-subdir pgn_inputs/gameknot \
  --output-filename test_llm_5_files.jsonl \
  --limit 5 \
  --refine-with-llm
```

**3.5 Download and Validate Results**
```bash
# Download output from volume
modal volume get chess-pgn-data \
  /outputs/test_llm_5_files.jsonl \
  ./test_modal_output.jsonl

# Validate JSONL format
cat test_modal_output.jsonl | jq '.'

# Compare with local CLI output (should be identical)
diff <(jq -S . local_output.jsonl) <(jq -S . test_modal_output.jsonl)
```

## Usage Examples

### Basic Usage (Regex Only)

```bash
# Process 100 PGN files with regex-based labeling
modal run chess_sandbox/concept_extraction.labelling/modal_pipeline.py::main \
  --input-subdir pgn_inputs/gameknot \
  --output-filename gameknot_100_regex.jsonl \
  --limit 100
```

### With LLM Refinement

```bash
# Process all files with LLM validation
modal run chess_sandbox/concept_extraction.labelling/modal_pipeline.py::main \
  --input-subdir pgn_inputs/gameknot \
  --output-filename gameknot_all_llm.jsonl \
  --refine-with-llm \
  --llm-model gpt-4o-mini
```

### Custom Model

```bash
# Use different LLM model
modal run chess_sandbox/concept_extraction.labelling/modal_pipeline.py::main \
  --input-subdir pgn_inputs/lichess \
  --output-filename lichess_gpt5nano.jsonl \
  --refine-with-llm \
  --llm-model gpt-5-nano
```

### Download Results

```bash
# Download output JSONL
modal volume get chess-pgn-data \
  /outputs/gameknot_all_llm.jsonl \
  ./data/processed/gameknot_labeled.jsonl

# Download entire output directory
modal volume get chess-pgn-data \
  /outputs \
  ./data/processed/
```

## File Structure

```
chess_sandbox/
├── concept_extraction.labelling/
│   ├── modal_pipeline.py       # NEW: Modal app definition
│   ├── pipeline.py              # EXISTING: Core logic (reused)
│   ├── parser.py                # EXISTING: PGN parsing
│   ├── labeller.py              # EXISTING: Regex detection
│   ├── refiner.py               # EXISTING: LLM validation
│   └── models.py                # EXISTING: Data models

docs/plans/
└── modal-pipeline-execution.md  # THIS DOCUMENT
```

## Deployment Instructions

### Prerequisites

1. **Modal Account**: Sign up at https://modal.com
2. **Create Token**: https://modal.com/settings/tokens
3. **Install Modal CLI**: `pip install modal`
4. **Authenticate**: `modal token set --token-id <ID> --token-secret <SECRET>`
5. **OpenAI API Key**: Get from https://platform.openai.com/api-keys

### One-Time Setup

```bash
# 1. Create Modal secret for OpenAI
modal secret create openai-secret OPENAI_API_KEY=sk-your-key-here

# 2. Create persistent volume
modal volume create chess-pgn-data

# 3. Upload PGN files
modal volume put chess-pgn-data \
  ./data/raw/annotated_pgn_free/gameknot \
  /pgn_inputs/gameknot
```

### Running the Pipeline

```bash
# Development testing (local simulation)
modal serve chess_sandbox/concept_extraction.labelling/modal_pipeline.py

# Production execution (remote Modal cloud)
modal run chess_sandbox/concept_extraction.labelling/modal_pipeline.py::main \
  --input-subdir pgn_inputs/gameknot \
  --output-filename output.jsonl \
  --refine-with-llm

# Download results
modal volume get chess-pgn-data \
  /outputs/output.jsonl \
  ./data/processed/output.jsonl
```

## Comparison with Local Execution

| Aspect | Local CLI | Modal Ephemeral App |
|--------|-----------|-------------------|
| **Command** | `uv run python -m chess_sandbox...` | `modal run modal_pipeline.py::main` |
| **Dependencies** | Local Python environment | Modal image (isolated) |
| **Secrets** | `.env` file | Modal Secrets |
| **Input** | Local filesystem | Modal Volume |
| **Output** | Local filesystem | Modal Volume |
| **Parallelization** | Asyncio (10 concurrent LLM calls) | Same asyncio pattern |
| **Cost** | Free (local compute) | Modal compute charges |
| **Scalability** | Limited by local machine | Auto-scales on Modal |
| **Use Case** | Development, testing | Large batch processing |

## Future Enhancements

1. **Progress Tracking**: Add logging for progress updates during execution
2. **Error Recovery**: Checkpoint intermediate results for fault tolerance
3. **Batch Splitting**: Split large volumes into multiple Modal function calls
4. **Output Streaming**: Stream JSONL output to cloud storage (S3/GCS)
5. **GPU Support**: Add GPU acceleration for local LLM inference (replace OpenAI)
6. **Monitoring**: Add custom metrics for tracking pipeline performance

## References

- Existing CLI: `chess_sandbox/concept_extraction.labelling/pipeline.py:main()`
- Parser logic: `chess_sandbox/concept_extraction.labelling/parser.py:parse_pgn_directory()`
- Labeller logic: `chess_sandbox/concept_extraction.labelling/labeller.py:label_positions()`
- Refiner logic: `chess_sandbox/concept_extraction.labelling/refiner.py:Refiner.refine()`
- Async parallelization: `chess_sandbox/concept_extraction.labelling/pipeline.py:refine_positions_parallel()`
- Modal API endpoint pattern: `chess_sandbox/endpoints.py`
- Modal ADR: `docs/adrs/20251029-use-modal-for-serverless-endpoints.md`
