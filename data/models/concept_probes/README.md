# Concept Probe Model Registry

This directory contains trained concept probe models for detecting chess concepts from LC0 layer activations.

## Model Naming Convention

Models should follow this naming pattern:
```
probe_v{version}_{date}_{layer}.pkl
```

Example: `probe_v1_2025_10_30_block3.pkl`

## Model Format

Each `.pkl` file contains:
- `classifier`: Trained sklearn OneVsRestClassifier
- `concept_list`: List of concept names in prediction order
- `layer_name`: Layer used for feature extraction (e.g., "block3/conv2/relu")
- `training_metrics`: Performance metrics from training
- `training_date`: ISO timestamp of when model was trained
- `model_version`: Version identifier

## Usage

```python
from chess_sandbox.concept_extraction.model.inference import ConceptProbe

# Load trained probe
probe = ConceptProbe.load("data/models/concept_probes/probe_v1.pkl")

# Extract features from a position
from chess_sandbox.concept_extraction.model.features import extract_features
features = extract_features(fen, "data/models/maia-1500.pt", probe.layer_name)

# Predict concepts
concepts = probe.predict(features)
```

## Training a New Probe

```bash
python -m chess_sandbox.concept_extraction.model.train \
    --data-path data/processed/async_100_positions_llm_refined_concepts.jsonl \
    --model-path data/models/maia-1500.pt \
    --layer-name block3/conv2/relu \
    --output data/models/concept_probes/probe_v2.pkl
```

## Model History

| Version | Date | Layer | Concepts | Exact Match | Notes |
|---------|------|-------|----------|-------------|-------|
| - | - | - | - | - | Training results will be logged here |
