# Enhanced HuggingFace Model Cards for Concept Extraction

## Overview

This plan outlines improvements to the model card implementation for chess concept extraction models uploaded to HuggingFace Hub. Currently, model cards are generated using a static markdown template with basic YAML frontmatter. This plan proposes leveraging HuggingFace's full model card capabilities, including the Python API, structured evaluation results (EvalResult), proper dataset linking, and comprehensive metadata formatting.

## Current State

### Existing Implementation

The model card creation is implemented in `chess_sandbox/concept_extraction/model/model_artefact.py:229-323` via the `ModelTrainingOutput._create_model_card()` method.

**Current features:**
- Basic YAML frontmatter (library_name, tags, model_type, language, license)
- Model description with concept list and layer information
- Performance metrics in markdown tables (exact match, per-concept accuracy/F1)
- Usage example code
- Training details (samples, split, seed)
- Additional metadata in separate `metadata.json` file

### Limitations

1. **No dataset linking**: Training datasets not referenced in YAML frontmatter, preventing automatic "Datasets used to train" display on Hub
2. **No structured evaluation results**: Metrics are in markdown tables, not in HuggingFace's `model-index` format compatible with Papers with Code
3. **Missing EvalResult usage**: Python API not leveraged for programmatic model card creation
4. **Incomplete metrics**: Precision/recall calculated but not included in model card (TODO on line 299)
5. **No base model reference**: LC0 model used for feature extraction not linked
6. **Manual YAML management**: Template uses string formatting instead of `ModelCardData` class
7. **Split provenance**: Some metadata in model card, some in separate JSON file

## Proposed Enhancements

### 1. Migrate to HuggingFace Model Card Python API

**Use `huggingface_hub` classes:**
- `ModelCard`: Main class for programmatic card creation
- `ModelCardData`: Manages YAML frontmatter metadata
- `EvalResult`: Structures evaluation metrics for model-index

**Benefits:**
- Type-safe metadata management
- Automatic YAML formatting and validation
- Consistent with HuggingFace best practices
- Easier to maintain and extend

### 2. Add Structured Evaluation Results (model-index)

**Create EvalResult objects for:**
- Overall metrics (exact match, hamming loss)
- Per-concept metrics (accuracy, F1, precision, recall)
- Baseline comparison metrics

**Format specification:**
```yaml
model-index:
- name: chess-concept-extraction-v1
  results:
  - task:
      type: multi-label-classification
      name: Chess Position Concept Extraction
    dataset:
      type: pilipolio/chess-positions-concepts
      name: Chess Positions with Concepts
      revision: main
    metrics:
    - type: exact_match
      value: 0.85
      name: Exact Match
    - type: hamming_loss
      value: 0.03
      name: Hamming Loss
    # Per-concept metrics as separate results...
```

### 3. Enhanced YAML Frontmatter

**Add missing fields:**
- `datasets`: Array of dataset repo IDs (e.g., `["pilipolio/chess-positions-concepts"]`)
- `base_model`: LC0 model used for feature extraction (e.g., `lc0-network-weights`)
- `pipeline_tag`: Classification task type
- `metrics`: Array of metric names for display
- `tags`: Enhanced tags (multi-label, chess, position-analysis)

### 4. Comprehensive Metrics Display

**Include in model card:**
- All calculated metrics (add precision/recall to current tables)
- Baseline comparison with improvement ratios
- Per-concept performance breakdown
- Support distribution for each concept

### 5. Better Provenance Tracking

**Consolidate metadata in model card:**
- Move relevant provenance from `metadata.json` into card
- Dataset version/revision used for training
- Base model version
- Hyperparameters
- Training environment details

## Implementation Phases

### Phase 1: Refactor Model Card Creation

**Update `_create_model_card()` method:**

```python
from huggingface_hub import ModelCard, ModelCardData, EvalResult

def _create_model_card(self) -> str:
    # Create evaluation results
    eval_results = self._create_eval_results()

    # Create model card data with metadata
    card_data = ModelCardData(
        language="en",
        license="apache-2.0",
        library_name="scikit-learn",
        tags=["chess", "multi-label-classification", "concept-extraction"],
        datasets=["pilipolio/chess-positions-concepts"],
        base_model="lc0-network-weights",  # If applicable
        metrics=["precision", "recall"],
        model_index=eval_results
    )

    # Create card with template
    card = ModelCard.from_template(
        card_data=card_data,
        template_path="model_card_template.md",
        # Template variables
        concepts=self.concepts,
        layer_name=self.layer_name,
        # ... other template vars
    )

    return str(card)
```

### Phase 2: Create EvalResult Objects

**Add helper method to generate structured results:**

```python
def _create_eval_results(self) -> list[dict]:
    """Create model-index structure with EvalResult objects."""
    results = []
    stats = self.training_stats

    # Overall model performance
    overall_results = [
        EvalResult(
            task_type="multi-label-classification",
            dataset_type="pilipolio/chess-positions-concepts",
            dataset_name="Chess Positions with Concepts",
            dataset_revision=self.dataset_revision,  # From metadata
            metric_type="exact_match",
            metric_value=stats["overall"]["exact_match"],
            metric_name="Exact Match"
        ),
        EvalResult(
            task_type="multi-label-classification",
            dataset_type="pilipolio/chess-positions-concepts",
            dataset_name="Chess Positions with Concepts",
            dataset_revision=self.dataset_revision,
            metric_type="hamming_loss",
            metric_value=stats["overall"]["hamming_loss"],
            metric_name="Hamming Loss"
        )
    ]

    return [{"results": [r.to_dict() for r in overall_results]}]
```

### Phase 3: Enhance Model Card Template

**Update template to include:**
- Precision/recall in per-concept performance table
- Baseline comparison section
- Training hyperparameters section
- Dataset version information
- Link to source dataset card

**Example enhanced performance section:**

```markdown
## Performance

### Overall Metrics

| Metric | Value |
|--------|-------|
| Exact Match | {{ exact_match }} |
| Hamming Loss | {{ hamming_loss }} |

### Per-Concept Performance

| Concept | Accuracy | F1 | Precision | Recall | Support |
|---------|----------|----|-----------| -------|---------|
{% for concept in concepts %}
| {{ concept }} | {{ metrics[concept].accuracy }} | {{ metrics[concept].f1 }} | {{ metrics[concept].precision }} | {{ metrics[concept].recall }} | {{ metrics[concept].support }} |
{% endfor %}

### Baseline Comparison

Compared to a dummy classifier (most_frequent strategy):
- Exact match improvement: {{ baseline_improvement.exact_match }}x
- Hamming loss reduction: {{ baseline_improvement.hamming_loss }}x
```

### Phase 4: Update Metadata Management

**Consolidate provenance:**
- Keep `metadata.json` for internal use
- Include key provenance in model card YAML and markdown
- Ensure dataset revision is tracked and referenced

**Add to card data:**
```python
card_data = ModelCardData(
    # ... existing fields ...
    dataset_info={
        "pilipolio/chess-positions-concepts": {
            "revision": self.dataset_revision,
            "split": "train",
            "samples": self.training_stats["samples"]
        }
    }
)
```

## Code Changes Required

### Files to Modify

1. **`chess_sandbox/concept_extraction/model/model_artefact.py`**
   - Update `_create_model_card()` method
   - Add `_create_eval_results()` helper
   - Add `_format_metrics_table()` helper for enhanced tables
   - Update `MODEL_CARD_TEMPLATE` constant

2. **`chess_sandbox/concept_extraction/model/train.py`** (potentially)
   - Ensure dataset revision is captured during training
   - Pass additional metadata to ModelTrainingOutput

### Testing Strategy

1. **Unit tests for EvalResult creation:**
   - Verify correct formatting of overall metrics
   - Verify correct formatting of per-concept metrics
   - Test with different numbers of concepts

2. **Integration test for model card generation:**
   - Create test training output
   - Generate model card
   - Validate YAML frontmatter
   - Verify model-index structure
   - Check markdown content rendering

3. **Manual validation:**
   - Upload test model to HuggingFace Hub
   - Verify "Datasets used to train" section appears
   - Check that metrics display correctly
   - Confirm Papers with Code integration works

## Benefits

1. **Better discoverability**: Proper dataset linking and metrics enable Hub's search and filtering
2. **Papers with Code integration**: Structured model-index enables automatic leaderboard inclusion
3. **Improved reproducibility**: Complete provenance in model card makes experiments reproducible
4. **Professional presentation**: Consistent with HuggingFace best practices and community standards
5. **Easier maintenance**: Python API is more maintainable than string templates
6. **Type safety**: ModelCardData provides validation and type checking
7. **Future-proof**: Following HF conventions means automatic support for new Hub features

## Migration Path

### Backward Compatibility

- Existing models keep their current cards
- New models get enhanced cards automatically
- Optional: Script to regenerate cards for existing models

### Rollout

1. Implement changes in development branch
2. Test with new model training
3. Validate on HuggingFace Hub preview
4. Merge to main
5. Update documentation

## References

- [HuggingFace Model Cards Guide](https://huggingface.co/docs/hub/model-cards)
- [Model Cards Python API](https://huggingface.co/docs/huggingface_hub/main/en/guides/model-cards)
- [Papers with Code Model Index Spec](https://github.com/paperswithcode/model-index)
- [ADR: Use HuggingFace Hub for Versioning](../adrs/20251101-use-huggingface-hub-for-versioning.md)
- [Concept Extraction HF DAG Plan](./concept-extraction-hf-dag.md)

## Open Questions

1. Should we include LC0 model version as base_model if we're using it for feature extraction?
2. What pipeline_tag best describes our task? (tabular-classification, other, or custom?)
3. Should we create separate EvalResult entries for each concept or group them?
4. Do we want to include chess position visualizations in the model card?
5. Should baseline metrics be in model-index or just in markdown?
