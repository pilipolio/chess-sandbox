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

# Chess Concept Probe - v1

Trained multi-label classifier for detecting chess concepts from LC0 layer activations.

## Model Description

Detects 11 chess concepts from internal activations of Leela Chess Zero (LC0) models:

`discovered_attack`, `exposed_king`, `fork`, `initiative`, `mating_threat`, `outpost`, `passed_pawn`, `pin`, `sacrifice`, `skewer`, `weak_square`

**Layer:** `block3/conv2/relu`
**Mode:** multi-label
**Training Date:** 2025-11-02

## Performance

- **Exact Match: 23.8%**
- Hamming Loss: 11.3%

### Per-Concept Performance

| Concept | Accuracy | F1 Score | Support |
|---------|----------|----------|---------|
| discovered_attack | 0.968 | 0.000 | 2 |
| fork | 0.873 | 0.333 | 9 |
| initiative | 0.905 | 0.000 | 5 |
| mating_threat | 0.683 | 0.231 | 13 |
| outpost | 0.968 | 0.000 | 2 |
| passed_pawn | 0.921 | 0.444 | 6 |
| pin | 0.635 | 0.410 | 19 |
| sacrifice | 0.889 | 0.364 | 8 |
| skewer | 0.952 | 0.000 | 3 |
| weak_square | 0.968 | 0.500 | 3 |

## Usage

```python
from chess_sandbox.concept_extraction.model.inference import ConceptProbe

# Load probe
probe = ConceptProbe.load("path/to/v1")

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

- **Training samples:** 250
- **Test samples:** 63
- **Test split:** 20.0%
- **Random seed:** 42

## Dependencies

- scikit-learn >= unknown
- Python >= 3.13.9
- lczerolens >= 0.3.3
- torch >= 2.9.0
