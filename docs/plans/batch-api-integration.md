# OpenAI Batch API Integration Plan

## Executive Summary

Integrate OpenAI's Batch API into the concept labeling pipeline to reduce API costs by 50-92% when processing large datasets. The Batch API is ideal for our use case: validating 37,257 chess concepts across 35,676 positions from 12,769 PGN files, with a 24-hour processing window that fits our workflow.

## Overview

### What is the Batch API?

The OpenAI Batch API allows asynchronous processing of large request batches with:
- **50% cost reduction** compared to synchronous API
- **Separate rate limits** (much higher than sync API)
- **24-hour completion window** (often completes faster)
- **Same endpoints**: /v1/responses, /v1/chat/completions, /v1/embeddings

### Why Use Batch API for Concept Labeling?

Our current concept labeling workflow is a perfect fit:
- ✅ Large scale: 37,257 concept validations needed
- ✅ Non-time-sensitive: 24-hour turnaround is acceptable
- ✅ Predictable format: Structured inputs/outputs with Pydantic
- ✅ Cost-sensitive: Processing full dataset repeatedly for development/evaluation
- ✅ Already async-friendly: Using httpx, respx, and JSONL workflows

## Cost Analysis

### Current Dataset (GameKnot Annotated Games)

From initial processing:
```
Total positions extracted: 350,019
Positions with concepts:    35,676
Total concept validations:  37,257

Top concepts:
  mating_threat:       11,725
  pin:                  8,260
  fork:                 4,770
  sacrifice:            4,421
  passed_pawn:          2,347
  weak_square:          2,096
  initiative:           1,383
  discovered_attack:    1,020
  outpost:                810
  skewer:                 425
```

### Token Estimation Per Validation

Based on `refiner.py` structure:
- **Input tokens**: ~350 per request
  - System prompt + concept definition: ~250 tokens (cacheable)
  - Position context (FEN, move, concept): ~100 tokens
- **Output tokens**: ~100 per request
  - Structured validation with reasoning

### Total Tokens for Full Dataset

- **Input**: 37,257 × 350 = **13,039,950 tokens** (~13M)
- **Output**: 37,257 × 100 = **3,725,700 tokens** (~3.7M)

### Cost Comparison: GPT-5 mini (default model)

| Approach | Input Cost | Output Cost | Total Cost | Savings |
|----------|-----------|-------------|------------|---------|
| **Regular API** | $3.25 | $7.40 | **$10.65** | - |
| **Batch API** | $1.63 | $3.70 | **$5.33** | **50%** |
| **Regular + Caching** | $1.16 | $7.40 | **$8.56** | 20% |
| **Batch + Caching** | $0.58 | $3.70 | **$4.28** | **60%** |

### Cost Comparison: GPT-5 nano (faster, cheaper alternative)

| Approach | Input Cost | Output Cost | Total Cost | Savings |
|----------|-----------|-------------|------------|---------|
| **Regular API** | $0.65 | $1.48 | **$2.13** | - |
| **Batch API** | $0.33 | $0.74 | **$1.07** | **50%** |
| **Regular + Caching** | $0.23 | $1.48 | **$1.71** | 20% |
| **Batch + Caching** | $0.12 | $0.74 | **$0.86** | **60%** |

### Recommended Configuration

**Batch API + Prompt Caching + GPT-5 nano = $0.86** (vs $10.65 regular) = **92% cost reduction**

This optimized configuration provides:
- 50% savings from Batch API
- 15-20% additional savings from prompt caching
- 5x cost reduction from using nano instead of mini
- Still effective for classification/validation tasks

## Current Architecture

### Existing Workflow (Synchronous)

```
pipeline.py (lines 82-96):
┌─────────────────────────────────────────┐
│ positions_to_refine = [...]             │
│                                         │
│ with progressbar(positions) as bar:    │
│     for position in bar:               │
│         position.concepts =            │
│             refiner.refine(position)   │ ← Sequential API calls
│                                         │
└─────────────────────────────────────────┘

refiner.py:
┌─────────────────────────────────────────┐
│ def refine(position: LabelledPosition): │
│     refined_concepts = []              │
│     for concept in position.concepts:  │
│         response = client.responses.   │ ← Sync API call
│             parse(...)                 │
│         if response.is_present:        │
│             refined_concepts.append()  │
│     return refined_concepts            │
└─────────────────────────────────────────┘
```

### Issues with Current Approach

1. **Sequential processing**: One API call at a time
2. **Rate limited**: Subject to regular API rate limits
3. **Expensive**: Full price for all requests
4. **Blocking**: Progress bar waits for each API call
5. **Not resumable**: If interrupted, must restart from beginning

## Target Architecture

### Batch API Workflow

```
pipeline.py:
┌──────────────────────────────────────────┐
│ positions_to_refine = [...]              │
│                                          │
│ batch_refiner = BatchRefiner()           │
│ batch_id = batch_refiner.submit_batch(  │ ← Submit all at once
│     positions_to_refine                  │
│ )                                        │
│                                          │
│ while not batch_complete:                │
│     status = batch_refiner.poll_status() │ ← Poll periodically
│     display_progress(status)             │
│     sleep(60)  # Check every minute      │
│                                          │
│ results = batch_refiner.                 │ ← Download results
│     process_results(batch_id)            │
└──────────────────────────────────────────┘

batch_refiner.py:
┌──────────────────────────────────────────┐
│ 1. prepare_batch_input()                 │
│    ├─ For each position & concept:       │
│    ├─ Create batch request with unique   │
│    │   custom_id: "pos_{i}_concept_{j}"  │
│    └─ Write to .jsonl file               │
│                                          │
│ 2. submit_batch()                        │
│    ├─ Upload .jsonl file                 │
│    ├─ Create batch job                   │
│    └─ Return batch_id                    │
│                                          │
│ 3. poll_status()                         │
│    ├─ Check batch status                 │
│    └─ Return progress metrics            │
│                                          │
│ 4. process_results()                     │
│    ├─ Download output file               │
│    ├─ Parse JSONL results                │
│    ├─ Map custom_id → position/concept   │
│    └─ Update positions with validations  │
└──────────────────────────────────────────┘
```

## Implementation Plan

### Phase 1: Core Batch Processing

#### 1.1 Create BatchRefiner Class

**File**: `chess_sandbox/concept_labelling/batch_refiner.py`

```python
from pathlib import Path
from openai import OpenAI
from .models import LabelledPosition, Concept
import json

class BatchRefiner:
    """Validates chess concepts using OpenAI Batch API for cost-effective processing."""

    def __init__(self, model: str = "gpt-5-nano", client: OpenAI | None = None):
        self.client = client or OpenAI()
        self.model = model
        self.batch_dir = Path("data/batches")
        self.batch_dir.mkdir(exist_ok=True)

    def prepare_batch_input(
        self,
        positions: list[LabelledPosition]
    ) -> Path:
        """Generate JSONL batch input file with all validation requests."""

    def submit_batch(self, batch_file: Path) -> str:
        """Upload batch file and submit job. Returns batch_id."""

    def poll_status(self, batch_id: str) -> dict:
        """Check batch job status and return progress metrics."""

    def download_results(self, batch_id: str) -> Path:
        """Download completed batch output file."""

    def process_results(
        self,
        results_file: Path,
        original_positions: list[LabelledPosition]
    ) -> list[LabelledPosition]:
        """Parse batch results and update positions with validated concepts."""
```

**Key Implementation Details**:

1. **prepare_batch_input()**:
   - Iterate through positions and concepts
   - Create custom_id: `f"pos_{pos_idx}_concept_{concept_idx}"`
   - Build request body matching current `refiner.py` prompt structure
   - Write to JSONL: one request per line
   - Store mapping file for custom_id → (position_idx, concept_idx)

2. **submit_batch()**:
   - Upload file: `client.files.create(file=batch_file, purpose="batch")`
   - Create batch: `client.batches.create(input_file_id=..., endpoint="/v1/responses", completion_window="24h")`
   - Save batch_id to metadata file for resumability
   - Return batch_id

3. **poll_status()**:
   - Call: `client.batches.retrieve(batch_id)`
   - Extract: status, request_counts (total, completed, failed)
   - Calculate: percent_complete = completed / total
   - Return structured dict for display

4. **process_results()**:
   - Download: `client.files.content(output_file_id)`
   - Parse JSONL output
   - Use custom_id to map results back to positions
   - Apply same validation logic as current `refiner.py`
   - Handle failed requests (write to error log)

#### 1.2 Update Pipeline Integration

**File**: `chess_sandbox/concept_labelling/pipeline.py`

**Changes**:
```python
# OLD (lines 82-96):
if refine_with_llm:
    with click.progressbar(positions_to_refine, label="Refining") as bar:
        for position in bar:
            try:
                position.concepts = refiner.refine(position)
            except Exception as e:
                click.echo(f"Warning: Failed to refine {position.move}: {e}")

# NEW:
if refine_with_llm:
    from .batch_refiner import BatchRefiner

    batch_refiner = BatchRefiner(model=model)

    # Submit batch
    click.echo("Preparing batch validation requests...")
    batch_file = batch_refiner.prepare_batch_input(positions_to_refine)
    click.echo(f"Submitting batch with {len(positions_to_refine)} positions...")
    batch_id = batch_refiner.submit_batch(batch_file)

    # Poll for completion
    click.echo(f"Batch submitted: {batch_id}")
    click.echo("Processing (this may take up to 24 hours)...")

    while True:
        status = batch_refiner.poll_status(batch_id)

        if status["status"] == "completed":
            click.echo("Batch completed! Downloading results...")
            break
        elif status["status"] in ["failed", "expired", "cancelled"]:
            raise RuntimeError(f"Batch {status['status']}: {status.get('errors')}")

        # Show progress
        pct = status["request_counts"]["completed"] / status["request_counts"]["total"] * 100
        click.echo(f"Progress: {pct:.1f}% ({status['request_counts']['completed']}/{status['request_counts']['total']})")

        time.sleep(60)  # Check every minute

    # Download and process results
    results_file = batch_refiner.download_results(batch_id)
    positions_to_refine = batch_refiner.process_results(results_file, positions_to_refine)
```

**Notes**:
- Remove `--refine-with-llm` flag logic (batch is now default when refining)
- Keep model parameter for configurability
- Add `--resume-batch <batch_id>` flag for resuming interrupted jobs

#### 1.3 Preserve Refiner for Testing/Development

**File**: `chess_sandbox/concept_labelling/refiner.py`

Keep the existing synchronous `Refiner` class for:
- Small-scale testing (< 100 positions)
- Interactive development
- Unit tests
- Fallback if batch API has issues

Add helper script for testing:
```python
# scripts/test_refiner.py
"""Test concept validation on small sample before submitting batch."""
```

### Phase 2: Error Handling & Resumability

#### 2.1 Handle Partial Failures

Batch API can have individual request failures within a successful batch:

```python
def process_results(self, results_file: Path, positions: list[LabelledPosition]):
    """Process results and handle failures."""

    results = self._parse_jsonl(results_file)
    failed_requests = []

    for result in results:
        if result.get("error"):
            # Log failed request
            failed_requests.append({
                "custom_id": result["custom_id"],
                "error": result["error"]
            })
            continue

        # Process successful result
        custom_id = result["custom_id"]
        pos_idx, concept_idx = self._parse_custom_id(custom_id)
        # ... update position

    # Write failed requests to error log
    if failed_requests:
        error_file = self.batch_dir / f"errors_{batch_id}.json"
        error_file.write_text(json.dumps(failed_requests, indent=2))
        click.echo(f"Warning: {len(failed_requests)} requests failed. See {error_file}")

    return positions
```

#### 2.2 Implement Resumability

Store batch metadata for resuming interrupted jobs:

```python
def submit_batch(self, batch_file: Path) -> str:
    """Submit batch and save metadata for resumability."""

    # Upload and submit
    file_obj = self.client.files.create(...)
    batch = self.client.batches.create(...)

    # Save metadata
    metadata = {
        "batch_id": batch.id,
        "batch_file": str(batch_file),
        "submitted_at": datetime.now().isoformat(),
        "status": batch.status,
        "num_positions": len(positions)
    }

    metadata_file = self.batch_dir / f"batch_{batch.id}.json"
    metadata_file.write_text(json.dumps(metadata, indent=2))

    return batch.id
```

Add CLI support:
```python
@click.option("--resume-batch", help="Resume a previously submitted batch job")
def main(resume_batch: str | None = None):
    if resume_batch:
        batch_refiner = BatchRefiner()
        # Poll and process existing batch
        ...
```

### Phase 3: Optimization

#### 3.1 Enable Prompt Caching

The system prompt and concept definitions are repeated across all requests. Enable caching:

```python
def prepare_batch_input(self, positions: list[LabelledPosition]) -> Path:
    """Generate batch input with prompt caching enabled."""

    # Mark cacheable content in system prompt
    system_prompt = {
        "role": "system",
        "content": [
            {
                "type": "cache_control",
                "cache_type": "ephemeral"
            },
            {
                "type": "text",
                "text": self._get_system_prompt()  # Cacheable
            }
        ]
    }

    # Build request...
```

**Expected savings**: 15-20% additional cost reduction

#### 3.2 Consider GPT-5 nano

For concept validation (classification task), GPT-5 nano may be sufficient:

- 5x cheaper than GPT-5 mini
- Optimized for classification and summarization
- Should handle binary validation (is concept present?) well

**Testing approach**:
1. Run small batch (100 positions) with nano
2. Compare validation accuracy vs mini
3. If accuracy is acceptable (>95% agreement), switch default to nano

### Phase 4: Testing

#### 4.1 Unit Tests with respx

Mock Batch API endpoints:

```python
# tests/test_batch_refiner.py
import respx
from httpx import Response

@respx.mock
def test_submit_batch():
    # Mock file upload
    respx.post("https://api.openai.com/v1/files").mock(
        return_value=Response(200, json={"id": "file-123"})
    )

    # Mock batch creation
    respx.post("https://api.openai.com/v1/batches").mock(
        return_value=Response(200, json={
            "id": "batch-456",
            "status": "validating",
            ...
        })
    )

    batch_refiner = BatchRefiner()
    batch_id = batch_refiner.submit_batch(test_file)

    assert batch_id == "batch-456"
```

#### 4.2 Integration Tests (Small Scale)

Test with real API on small dataset:

```python
# tests/integration/test_batch_refiner_integration.py
@pytest.mark.integration
def test_batch_refiner_end_to_end():
    """Test actual Batch API with 10 positions."""

    positions = load_test_positions(limit=10)
    batch_refiner = BatchRefiner(model="gpt-5-nano")

    batch_file = batch_refiner.prepare_batch_input(positions)
    batch_id = batch_refiner.submit_batch(batch_file)

    # Wait for completion (with timeout)
    result_positions = batch_refiner.wait_and_process(batch_id, timeout=3600)

    # Validate results
    assert len(result_positions) == 10
    for pos in result_positions:
        assert len(pos.concepts) >= 0  # May have filtered out false positives
```

#### 4.3 Validation Testing

Ensure batch results match synchronous results:

```python
def test_batch_matches_sync():
    """Verify batch processing produces same results as sync."""

    positions = load_test_positions(limit=20)

    # Sync refining
    sync_refiner = Refiner()
    sync_results = [sync_refiner.refine(p) for p in positions]

    # Batch refining
    batch_refiner = BatchRefiner()
    batch_results = batch_refiner.submit_and_process(positions)

    # Compare
    for sync_pos, batch_pos in zip(sync_results, batch_results):
        assert sync_pos.concepts == batch_pos.concepts
```

## Migration Path

### Step 1: Development & Testing (Week 1)
- [ ] Implement `BatchRefiner` class
- [ ] Write unit tests with respx mocks
- [ ] Test with small real batch (10-100 positions)

### Step 2: Validation (Week 1-2)
- [ ] Compare batch vs sync results on sample dataset
- [ ] Test GPT-5 nano vs mini accuracy
- [ ] Validate error handling with intentionally malformed requests

### Step 3: Integration (Week 2)
- [ ] Update `pipeline.py` to use `BatchRefiner`
- [ ] Add `--resume-batch` flag
- [ ] Update documentation

### Step 4: Full-Scale Testing (Week 2-3)
- [ ] Process full GameKnot dataset (37,257 validations)
- [ ] Monitor for errors and edge cases
- [ ] Validate output quality

### Step 5: Cleanup (Week 3)
- [ ] Remove old synchronous loop from `pipeline.py`
- [ ] Keep `Refiner` for testing/development
- [ ] Update CLAUDE.md with new workflow

## Monitoring & Observability

### Batch Job Tracking

Store batch metadata for monitoring:

```json
{
  "batch_id": "batch_abc123",
  "submitted_at": "2025-10-30T10:00:00Z",
  "completed_at": "2025-10-30T14:23:15Z",
  "status": "completed",
  "num_positions": 35676,
  "num_requests": 37257,
  "request_counts": {
    "total": 37257,
    "completed": 37180,
    "failed": 77
  },
  "cost_estimate": {
    "model": "gpt-5-nano",
    "input_tokens": 13039950,
    "output_tokens": 3725700,
    "total_cost_usd": 0.86
  }
}
```

### Error Tracking

Log failed requests for investigation:

```json
{
  "batch_id": "batch_abc123",
  "failed_requests": [
    {
      "custom_id": "pos_1234_concept_2",
      "error": {
        "code": "invalid_request",
        "message": "..."
      }
    }
  ]
}
```

## Future Enhancements

### 1. Parallel Batch Processing

For very large datasets, split into multiple concurrent batches:
- Each batch limited to 50,000 requests
- Submit multiple batches in parallel
- Aggregate results

### 2. Incremental Processing

For ongoing data collection:
- Track which positions have been validated
- Only submit new positions in batch
- Append to existing dataset

### 3. Batch Evaluation Pipeline

Apply same pattern to `evaluation.py`:
- Batch all commentary generation
- Batch all judging tasks
- Significant cost savings for experiments

### 4. Embeddings for Similarity Search

Use Batch API for embeddings:
- Generate embeddings for all positions
- Build similarity search index
- Find similar tactical patterns

## References

- [OpenAI Batch API Documentation](https://platform.openai.com/docs/guides/batch)
- [OpenAI Pricing](https://openai.com/api/pricing/)
- [Prompt Caching Documentation](https://platform.openai.com/docs/guides/prompt-caching)
- ADR: [Use OpenAI SDK Directly](../adrs/20251030-use-openai-sdk-directly.md)

## Success Metrics

- ✅ **Cost reduction**: Achieve 50%+ savings vs synchronous API
- ✅ **Quality**: Maintain >95% agreement with synchronous validation results
- ✅ **Scale**: Successfully process full GameKnot dataset (37,257 validations)
- ✅ **Reliability**: <1% failed requests in batch processing
- ✅ **Resumability**: Support interruption and resumption of batch jobs
