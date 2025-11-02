"""
Inference wrapper for trained concept probes.

Provides loading and prediction for sklearn classifiers trained to detect
chess concepts from LC0 layer activations.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from ..labelling.labeller import Concept


@dataclass
class ConceptProbe:
    """
    Trained concept classifier with metadata.

    Loads and applies sklearn classifiers to predict chess concepts
    from activation vectors. Supports both multi-class (one concept
    per position) and multi-label (multiple concepts per position).

    Example:
        >>> probe = ConceptProbe.load("models/concept_probes/probe_v1.pkl")  # doctest: +SKIP
        >>> features = np.random.rand(4096)  # From feature extraction  # doctest: +SKIP
        >>> concepts = probe.predict(features)  # doctest: +SKIP
        >>> len(concepts) >= 0  # doctest: +SKIP
        True
    """

    classifier: Any
    concept_list: list[str]
    layer_name: str
    training_metrics: dict[str, Any]
    training_date: str
    model_version: str
    label_encoder: Any = None

    @classmethod
    def load(cls, path: str | Path) -> "ConceptProbe":
        """
        Load trained probe from HF snapshot format (directory) or legacy .pkl file.

        Args:
            path: Path to probe directory

        Returns:
            Loaded ConceptProbe instance

        Example:
            >>> # probe = ConceptProbe.load("models/concept_probes/probe_v1")
            >>> # probe.concept_list
            >>> # ['pin', 'fork', 'skewer', ...]
            True
        """
        path = Path(path)

        # Load from HF snapshot format (directory)
        if not path.is_dir():
            msg = f"Path must be directory (HF format): {path}"
            raise ValueError(msg)

        # Load model components
        classifier = joblib.load(path / "probe_model.joblib")

        encoder_path = path / "label_encoder.joblib"
        label_encoder = joblib.load(encoder_path) if encoder_path.exists() else None

        # Load metadata
        config = json.loads((path / "config.json").read_text())
        metadata = json.loads((path / "probe_metadata.json").read_text())

        return cls(
            classifier=classifier,
            concept_list=config["concepts"],
            layer_name=config["layer_name"],
            training_metrics=metadata["performance"],
            training_date=metadata["model_config"]["training_date"],
            model_version=config["version"],
            label_encoder=label_encoder,
        )

    def save(self, path: str | Path) -> None:
        """
        Save probe in HuggingFace snapshot format (directory with multiple files).

        Creates a directory with:
        - probe_model.joblib: Sklearn classifier
        - label_encoder.joblib: Label encoder (if exists)
        - config.json: Quick-load configuration
        - probe_metadata.json: Full training metadata
        - README.md: Model card

        Args:
            path: Path to save probe directory (e.g., "models/probe_v1")

        Example:
            >>> # probe.save("models/concept_probes/probe_v2")
            >>> # Creates: models/concept_probes/probe_v2/ with multiple files
            True
        """
        path = Path(path)

        # Create directory
        path.mkdir(parents=True, exist_ok=True)

        # Save model components using joblib (sklearn best practice)
        joblib.dump(self.classifier, path / "probe_model.joblib")

        if self.label_encoder is not None:
            joblib.dump(self.label_encoder, path / "label_encoder.joblib")

        # Save metadata and config
        (path / "probe_metadata.json").write_text(json.dumps(self._create_metadata(), indent=2))
        (path / "config.json").write_text(json.dumps(self._create_config(), indent=2))
        (path / "README.md").write_text(self._create_model_card())

        print(f"Saved probe to {path}/ (HF snapshot format)")

    def _create_metadata(self) -> dict[str, Any]:
        """Create structured metadata for HF Hub."""
        metrics = self.training_metrics
        mode = metrics.get("mode", "multi-label")

        return {
            "model_config": {
                "layer_name": self.layer_name,
                "mode": mode,
                "version": self.model_version,
                "n_concepts": len(self.concept_list),
                "concepts": self.concept_list,
                "training_date": self.training_date,
            },
            "training_info": {
                "data_path": metrics.get("data_path", "unknown"),
                "n_train": metrics.get("training_samples", 0),
                "n_test": metrics.get("test_samples", 0),
                "test_split": metrics.get("test_split", 0.2),
                "random_seed": metrics.get("random_seed", 42),
                "training_params": {
                    "verbose": metrics.get("verbose", False),
                    "n_jobs": metrics.get("n_jobs", -1),
                },
            },
            "performance": {
                "baseline": metrics.get("baseline", {}),
                "probe": metrics.get("probe", {}),
            },
            "dependencies": {
                "sklearn": self._get_package_version("sklearn"),
                "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "lczerolens": self._get_package_version("lczerolens"),
                "torch": self._get_package_version("torch"),
            },
        }

    def _create_config(self) -> dict[str, Any]:
        """Create minimal quick-load config."""
        return {
            "model_type": "concept-probe",
            "library": "sklearn",
            "layer_name": self.layer_name,
            "mode": self.training_metrics.get("mode", "multi-label"),
            "version": self.model_version,
            "n_concepts": len(self.concept_list),
            "concepts": self.concept_list,
            "feature_dim": 4096,  # Standard for block3/conv2/relu
        }

    def _create_model_card(self) -> str:
        """Generate model card with YAML frontmatter."""
        metadata = self._create_metadata()
        mode = metadata["model_config"]["mode"]
        metrics = metadata["performance"]["probe"]

        # Calculate key metrics
        if mode == "multi-label":
            key_metric = f"Exact Match: {metrics.get('exact_match', 0):.1%}"
            secondary_metric = f"Hamming Loss: {metrics.get('hamming_loss', 0):.1%}"
        else:
            key_metric = f"Accuracy: {metrics.get('accuracy', 0):.1%}"
            secondary_metric = f"F1 (macro): {metrics.get('f1_macro', 0):.1%}"

        card = f"""---
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

# Chess Concept Probe - {self.model_version}

Trained {mode} classifier for detecting chess concepts from LC0 layer activations.

## Model Description

Detects {len(self.concept_list)} chess concepts from internal activations of Leela Chess Zero (LC0) models:

{', '.join(f'`{c}`' for c in self.concept_list)}

**Layer:** `{self.layer_name}`
**Mode:** {mode}
**Training Date:** {self.training_date[:10]}

## Performance

- **{key_metric}**
- {secondary_metric}

### Per-Concept Performance

| Concept | Accuracy | F1 Score | Support |
|---------|----------|----------|---------|
"""

        # Add per-concept metrics
        per_concept = metrics.get("per_concept", {})
        for concept in self.concept_list:
            if concept in per_concept:
                c_metrics = per_concept[concept]
                card += (
                    f"| {concept} | {c_metrics['accuracy']:.3f} | {c_metrics['f1']:.3f} | {c_metrics['support']} |\n"
                )

        card += f"""
## Usage

```python
from chess_sandbox.concept_extraction.model.inference import ConceptProbe

# Load probe
probe = ConceptProbe.load("path/to/{self.model_version}")

# Extract features
from chess_sandbox.concept_extraction.model.features import extract_features
features = extract_features(
    fen="rnbqkb1r/pp1ppppp/5n2/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    model_path="path/to/maia-1500.pt",
    layer_name="{self.layer_name}"
)

# Predict concepts
concepts = probe.predict(features)
```

## Training Details

- **Training samples:** {metadata['training_info']['n_train']:,}
- **Test samples:** {metadata['training_info']['n_test']:,}
- **Test split:** {metadata['training_info']['test_split']:.1%}
- **Random seed:** {metadata['training_info']['random_seed']}

## Dependencies

- scikit-learn >= {metadata['dependencies']['sklearn']}
- Python >= {metadata['dependencies']['python']}
- lczerolens >= {metadata['dependencies']['lczerolens']}
- torch >= {metadata['dependencies']['torch']}
"""
        return card

    @staticmethod
    def _get_package_version(package: str) -> str:
        """Get package version."""
        try:
            import importlib.metadata

            return importlib.metadata.version(package)
        except Exception:
            return "unknown"

    @classmethod
    def from_hub(
        cls,
        repo_id: str,
        *,
        revision: str | None = None,
        cache_dir: Path | str | None = None,
        force_download: bool = False,
        token: str | None = None,
    ) -> "ConceptProbe":
        """
        Load probe from HuggingFace Hub.

        Args:
            repo_id: Repository ID (e.g., "chess-concept-probes/probe-v1")
            revision: Git revision (tag, branch, commit). Defaults to "main"
            cache_dir: Custom cache directory (defaults to ~/.cache/huggingface/)
            force_download: Force re-download even if cached
            token: HF authentication token for private repos

        Returns:
            Loaded ConceptProbe instance

        Example:
            >>> probe = ConceptProbe.from_hub("chess-concept-probes/probe-v1")  # doctest: +SKIP
            >>> probe.concept_list  # doctest: +SKIP
            ['fork', 'pin', ...]
        """
        from huggingface_hub import snapshot_download

        from chess_sandbox.config import settings

        # Use settings for defaults
        cache_dir = cache_dir or settings.HF_CACHE_DIR
        token = token or settings.HF_TOKEN or None

        # Download snapshot (uses HF cache automatically)
        local_dir = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            token=token,
        )

        # Load from downloaded snapshot
        return cls.load(local_dir)

    def predict(self, features: np.ndarray, threshold: float = 0.5) -> list[Concept]:
        """
        Predict concepts from activation features.

        Args:
            features: Flattened activation vector (1D or 2D with batch size 1)
            threshold: Probability threshold for positive prediction

        Returns:
            List of detected concepts with validated_by="probe"

        Example:
            >>> # probe = ConceptProbe.load("models/probe.pkl")
            >>> # features = np.random.rand(4096)
            >>> # concepts = probe.predict(features)
            >>> # all(c.validated_by == "probe" for c in concepts)
            True
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)

        if hasattr(self.classifier, "predict_proba"):
            probas: Any = self.classifier.predict_proba(features)[0]
            predictions: Any = probas >= threshold
        else:
            predictions = self.classifier.predict(features)[0]

        concepts: list[Concept] = []
        for i, concept_name in enumerate(self.concept_list):
            if predictions[i]:
                concepts.append(
                    Concept(
                        name=concept_name,
                        validated_by="probe",
                        temporal="actual",
                    )
                )

        return concepts

    def predict_batch(self, features: np.ndarray, threshold: float = 0.5) -> list[list[Concept]]:
        """
        Predict concepts for multiple positions.

        Args:
            features: Array of shape (n_positions, n_features)
            threshold: Probability threshold for positive prediction

        Returns:
            List of concept lists, one per position

        Example:
            >>> # probe = ConceptProbe.load("models/probe.pkl")
            >>> # features = np.random.rand(10, 4096)
            >>> # batch_concepts = probe.predict_batch(features)
            >>> # len(batch_concepts) == 10
            True
        """
        if hasattr(self.classifier, "predict_proba"):
            probas: Any = self.classifier.predict_proba(features)
            predictions: Any = probas >= threshold
        else:
            predictions = self.classifier.predict(features)

        batch_concepts: list[list[Concept]] = []
        for pred in predictions:
            concepts: list[Concept] = []
            for i, concept_name in enumerate(self.concept_list):
                if pred[i]:
                    concepts.append(
                        Concept(
                            name=concept_name,
                            validated_by="probe",
                            temporal="actual",
                        )
                    )
            batch_concepts.append(concepts)

        return batch_concepts

    def predict_with_confidence(self, features: np.ndarray) -> list[tuple[str, float]]:
        """
        Predict concepts with confidence scores.

        Args:
            features: Flattened activation vector

        Returns:
            List of (concept_name, probability) tuples for all concepts

        Example:
            >>> # probe = ConceptProbe.load("models/probe.pkl")
            >>> # features = np.random.rand(4096)
            >>> # scores = probe.predict_with_confidence(features)
            >>> # all(0 <= score <= 1 for _, score in scores)
            True
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)

        if hasattr(self.classifier, "predict_proba"):
            probas = self.classifier.predict_proba(features)[0]
        else:
            predictions = self.classifier.predict(features)[0]
            probas = predictions.astype(float)

        return [(name, float(prob)) for name, prob in zip(self.concept_list, probas, strict=True)]

    def __repr__(self) -> str:
        return (
            f"ConceptProbe(concepts={len(self.concept_list)}, "
            f"layer={self.layer_name}, "
            f"version={self.model_version})"
        )
