---
library_name: sklearn
tags:
- chess
- concept-detection
- interpretability
- lc0
- multi-label-classification
model_type: concept-probe
language:
- en
license: mit
---

# Chess Concept Probe

Trained multi-label classifier for detecting chess concepts from LC0 layer activations.

## Model Description

Detects 6 chess concepts from internal activations of Leela Chess Zero (LC0) models:

`discovered_attack`, `fork`, `mating_threat`, `passed_pawn`, `pin`, `sacrifice`

**Layer:** `block3/conv2/relu`
**Mode:** multi-label
**Training Date:** 2025-11-04

## Performance

- **Exact Match: 0.0%**
- 

### Per-Concept Performance

| Concept | Accuracy | F1 Score | Support |
|---------|----------|----------|---------|
| discovered_attack | 0.750 | 0.000 | 1 |
| fork | 0.750 | 0.000 | 1 |
| passed_pawn | 0.500 | 0.000 | 2 |
| pin | 0.750 | 0.000 | 1 |

## Usage

```python
from chess_sandbox.concept_extraction.model.model_artefact import ModelTrainingOutput

# Load training output from HF Hub
output = ModelTrainingOutput.from_hub("pilipolio/chess-sandbox-concept-probes")
probe = output.probe

# Extract features
from chess_sandbox.concept_extraction.model.features import extract_features
features = extract_features(
    fen="rnbqkb1r/pp1ppppp/5n2/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    model_path="path/to/maia-1500.onnx",
    layer_name="block3/conv2/relu"
)

# Predict concepts
concepts = probe.predict(features)
```

## Training Details

- **Training samples:** 14
- **Test samples:** 4
- **Test split:** 20.0%
- **Random seed:** 42
