---
base_model: lczerolens/maia-1500
datasets:
- pilipolio/chess-positions-concepts
language: en
library_name: scikit-learn
license: mit
model_name: Chess Concept Probe
pipeline_tag: tabular-classification
tags:
- chess
- concept-detection
- interpretability
- lc0
- multi-label-classification
model_index:
- name: chess-concept-extraction
  results:
  - task:
      type: tabular-classification
      name: Chess Position Concept Extraction
    dataset:
      type: pilipolio/chess-positions-concepts
      name: Chess Positions with Concepts
      revision: test_fixture
    metrics:
    - type: exact_match
      value: 0.0
      name: Exact Match
    - type: hamming_loss
      value: 0.3333333333333333
      name: Hamming Loss
    - type: precision
      value: 0.0
      name: Precision (Micro)
    - type: recall
      value: 0.0
      name: Recall (Micro)
    - type: f1
      value: 0.0
      name: F1 (Micro)
    - type: precision
      value: 0.0
      name: Precision (Macro)
    - type: recall
      value: 0.0
      name: Recall (Macro)
    - type: f1
      value: 0.0
      name: F1 (Macro)
model-index:
- name: chess-concept-extraction
  results:
  - task:
      type: tabular-classification
      name: Chess Position Concept Extraction
    dataset:
      type: pilipolio/chess-positions-concepts
      name: Chess Positions with Concepts
      revision: test_fixture
    metrics:
    - type: exact_match
      value: 0.0
      name: Exact Match
    - type: hamming_loss
      value: 0.3333333333333333
      name: Hamming Loss
    - type: precision
      value: 0.0
      name: Precision (Micro)
    - type: recall
      value: 0.0
      name: Recall (Micro)
    - type: f1
      value: 0.0
      name: F1 (Micro)
    - type: precision
      value: 0.0
      name: Precision (Macro)
    - type: recall
      value: 0.0
      name: Recall (Macro)
    - type: f1
      value: 0.0
      name: F1 (Macro)
---
# Chess Concept Probe

Trained multi-label classifier for detecting chess concepts from LC0 layer activations.

## Model Description

Detects 6 chess concepts: `discovered_attack`, `fork`, `mating_threat`, `passed_pawn`, `pin`, `sacrifice`

**Layer:** `block3/conv2/relu` | **Mode:** multi-label | **Trained:** 2025-11-04

## Performance

- Exact Match: **0.0%**
- Hamming Loss: **0.3333**


Detailed metrics available in model-index below.

## Usage

```python
from chess_sandbox.concept_extraction.model.model_artefact import ModelTrainingOutput

# Load from HF Hub
output = ModelTrainingOutput.from_hub("pilipolio/chess-sandbox-concept-probes")
probe = output.probe

# Extract features and predict
from chess_sandbox.concept_extraction.model.features import extract_features
features = extract_features(
    fen="rnbqkb1r/pp1ppppp/5n2/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    model_path="path/to/maia-1500.onnx",
    layer_name="block3/conv2/relu"
)
concepts = probe.predict(features)
```

## Training Details

- Training: 14 samples | Test: 4 samples | Split: 20.0% | Seed: 42
