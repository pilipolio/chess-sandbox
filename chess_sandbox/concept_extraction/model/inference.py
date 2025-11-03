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

import click
import joblib
import numpy as np
from huggingface_hub import hf_hub_download, snapshot_download

from ...config import settings
from ..labelling.labeller import Concept, LabelledPosition


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

        if not path.is_dir():
            msg = f"Path must be directory (HF format): {path}"
            raise ValueError(msg)

        classifier = joblib.load(path / "probe_model.joblib")
        encoder_path = path / "label_encoder.joblib"
        label_encoder = joblib.load(encoder_path) if encoder_path.exists() else None
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
    model_path="path/to/maia-1500.onnx",
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


@dataclass
class ConceptExtractor:
    """
    High-level wrapper for extracting chess concepts from FEN positions.

    Combines LC0 model loading, feature extraction, and concept prediction
    into a single convenient interface. Downloads all dependencies from
    HuggingFace Hub for zero-setup usage.

    Example:
        >>> extractor = ConceptExtractor.from_hf(  # doctest: +SKIP
        ...     probe_repo_id="pilipolio/chess-sandbox-concept-probes"
        ... )
        >>> concepts = extractor.extract_concepts(  # doctest: +SKIP
        ...     "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        ... )
    """

    probe: ConceptProbe
    model: Any
    layer_name: str

    @classmethod
    def from_hf(
        cls,
        probe_repo_id: str,
        *,
        model_repo_id: str = "lczerolens/maia-1500",
        model_filename: str = "model.onnx",
        revision: str | None = None,
        cache_dir: Path | str | None = None,
        force_download: bool = False,
        token: str | None = None,
    ) -> "ConceptExtractor":
        """
        Load extractor from HuggingFace Hub (downloads both model and probe).

        Downloads the Leela Chess Zero model and concept probe from HuggingFace Hub,
        with automatic caching to avoid redundant downloads.

        Args:
            probe_repo_id: Probe repository (e.g., "pilipolio/chess-sandbox-concept-probes")
            model_repo_id: LC0 model repository (default: "lczerolens/maia-1500")
            model_filename: Model file to download (default: "model.onnx")
            revision: Git revision for probe (tag, branch, commit). Defaults to "main"
            cache_dir: Custom cache directory (defaults to ~/.cache/huggingface/)
            force_download: Force re-download even if cached
            token: HF authentication token for private repos

        Returns:
            Loaded ConceptExtractor instance with model and probe ready

        Example:
            >>> extractor = ConceptExtractor.from_hf(  # doctest: +SKIP
            ...     probe_repo_id="pilipolio/chess-sandbox-concept-probes"
            ... )
            >>> extractor.layer_name  # doctest: +SKIP
            'block3/conv2/relu'
        """
        from .features import LczeroModel

        cache_dir = cache_dir or settings.HF_CACHE_DIR
        token = token or settings.HF_TOKEN or None

        print(f"Downloading probe from {probe_repo_id}...")
        probe = ConceptProbe.from_hub(
            repo_id=probe_repo_id,
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            token=token,
        )
        print(f"Loaded probe: {probe}")

        print(f"\nDownloading LC0 model from {model_repo_id}/{model_filename}...")
        model_path = hf_hub_download(
            repo_id=model_repo_id,
            filename=model_filename,
            cache_dir=cache_dir,
            force_download=force_download,
            token=token,
        )
        print(f"Model downloaded to: {model_path}")

        print("Loading LC0 model...")
        model = LczeroModel.from_path(model_path)
        print("Model loaded successfully!")

        return cls(
            probe=probe,
            model=model,
            layer_name=probe.layer_name,
        )

    def extract_concepts(self, fen: str, threshold: float = 0.5) -> list[Concept]:
        """
        Extract concepts from a single FEN position.

        Args:
            fen: FEN notation of chess position
            threshold: Probability threshold for concept detection (default: 0.5)

        Returns:
            List of detected concepts with validated_by="probe"

        Example:
            >>> # extractor = ConceptExtractor.from_hf(...)
            >>> # concepts = extractor.extract_concepts(
            >>> #     "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
            >>> # )
            >>> # len(concepts) >= 0
            True
        """
        from .features import extract_features_batch

        features_batch = extract_features_batch(
            fens=[fen],
            layer_name=self.layer_name,
            model=self.model,
            batch_size=1,
        )
        features = features_batch[0]

        return self.probe.predict(features, threshold=threshold)

    def extract_concepts_with_confidence(self, fen: str) -> list[tuple[str, float]]:
        """
        Extract concepts with confidence scores from a FEN position.

        Args:
            fen: FEN notation of chess position

        Returns:
            List of (concept_name, probability) tuples for all concepts

        Example:
            >>> # extractor = ConceptExtractor.from_hf(...)
            >>> # scores = extractor.extract_concepts_with_confidence(
            >>> #     "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
            >>> # )
            >>> # all(0 <= score <= 1 for _, score in scores)
            True
        """
        from .features import extract_features_batch

        features_batch = extract_features_batch(
            fens=[fen],
            layer_name=self.layer_name,
            model=self.model,
            batch_size=1,
        )
        features = features_batch[0]

        return self.probe.predict_with_confidence(features)

    def __repr__(self) -> str:
        return f"ConceptExtractor(probe={self.probe}, " f"layer={self.layer_name})"


@click.group()
def cli() -> None:
    """Concept probe inference CLI."""
    pass


@cli.command()
@click.argument("fen", type=str)
@click.option(
    "--probe-path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to trained probe directory",
)
@click.option(
    "--model-path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to LC0 model file (e.g., maia-1500.onnx)",
)
def predict(
    fen: str,
    probe_path: Path,
    model_path: Path,
) -> None:
    """
    Predict concepts for a single FEN position with confidence scores.

    Example:
        python -m chess_sandbox.concept_extraction.model.inference predict \\
            --probe-path data/models/concept_probes/probe_v1 \\
            --model-path tests/fixtures/maia-1500.onnx \\
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    """
    from .features import extract_features

    print("Loading probe...")
    probe = ConceptProbe.load(probe_path)
    print(f"Loaded probe: {probe}")

    print(f"\nExtracting features for FEN: {fen}")
    features = extract_features(
        fen=fen,
        model_path=model_path,
        layer_name=probe.layer_name,
    )

    print("\nMaking predictions...")
    predictions_with_confidence = probe.predict_with_confidence(features)

    # Sort by confidence descending
    predictions_with_confidence.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{'=' * 70}")
    print("PREDICTIONS")
    print(f"{'=' * 70}\n")
    print(f"FEN: {fen}\n")

    if predictions_with_confidence:
        print("Predicted Concepts (with confidence):")
        for concept, confidence in predictions_with_confidence:
            if confidence >= 0.1:
                print(f"  {concept}: {confidence:.2%}")
    else:
        print("No concepts predicted")

    print()


@cli.command()
@click.option(
    "--probe-path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to trained probe directory",
)
@click.option(
    "--data-path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to JSONL file with labeled positions",
)
@click.option(
    "--model-path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to LC0 model file (e.g., maia-1500.onnx)",
)
@click.option(
    "--batch-size",
    default=32,
    type=int,
    help="Batch size for activation extraction",
)
def batch_predict(
    probe_path: Path,
    data_path: Path,
    model_path: Path,
    batch_size: int,
) -> None:
    """
    Predict concepts for multiple positions from JSONL file with confidence scores.

    Example:
        python -m chess_sandbox.concept_extraction.model.inference batch-predict \\
            --probe-path data/models/concept_probes/probe_v1 \\
            --data-path data/processed/test_labeled_positions.jsonl \\
            --model-path tests/fixtures/maia-1500.onnx
    """
    from .features import LczeroModel, extract_features_batch

    print("Loading probe...")
    probe = ConceptProbe.load(probe_path)
    print(f"Loaded probe: {probe}")

    print("\nLoading LC0 model...")
    model = LczeroModel.from_path(str(model_path))
    print("Model loaded successfully!")

    print("\nLoading positions from JSONL...")
    positions: list[LabelledPosition] = []
    with data_path.open() as f:
        for line in f:
            positions.append(LabelledPosition.from_dict(json.loads(line)))

    print(f"Loaded {len(positions)} positions")

    print(f"\nExtracting activations for {len(positions)} positions...")
    fens = [p.fen for p in positions]
    activations = extract_features_batch(
        fens,
        probe.layer_name,
        model=model,
        batch_size=batch_size,
    )
    print(f"Activation matrix shape: {activations.shape}")

    print("\nMaking predictions...")

    print(f"\n{'=' * 70}")
    print(f"BATCH PREDICTIONS ({len(positions)} positions)")
    print(f"{'=' * 70}\n")

    for i, pos in enumerate(positions):
        features = activations[i : i + 1]
        predictions_with_confidence = probe.predict_with_confidence(features)
        predictions_with_confidence.sort(key=lambda x: x[1], reverse=True)

        print(f"Position {i + 1}:")
        print(f"FEN: {pos.fen}")

        # Show ground truth if available
        if pos.concepts:
            validated_concepts = [c.name for c in pos.concepts if c.validated_by is not None]
            if validated_concepts:
                print(f"Ground Truth: {', '.join(validated_concepts)}")

        # Show predictions with confidence
        high_confidence = [(c, conf) for c, conf in predictions_with_confidence if conf >= 0.5]
        if high_confidence:
            pred_str = ", ".join(f"{c} ({conf:.2%})" for c, conf in high_confidence)
            print(f"Predictions:  {pred_str}")
        else:
            print("Predictions:  (none)")

        print()


@cli.command()
@click.option(
    "--probe-path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to trained probe directory",
)
@click.option(
    "--data-path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to JSONL file with labeled positions",
)
@click.option(
    "--model-path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to LC0 model file (e.g., maia-1500.onnx)",
)
@click.option(
    "--sample-size",
    default=10,
    type=int,
    help="Number of sample predictions to display",
)
@click.option(
    "--batch-size",
    default=32,
    type=int,
    help="Batch size for activation extraction",
)
@click.option(
    "--random-seed",
    default=42,
    type=int,
    help="Random seed for sample selection",
)
def evaluate(
    probe_path: Path,
    data_path: Path,
    model_path: Path,
    sample_size: int,
    batch_size: int,
    random_seed: int,
) -> None:
    """
    Evaluate trained concept probe on test data and display sample predictions.

    Example:
        python -m chess_sandbox.concept_extraction.model.inference evaluate \\
            --probe-path models/concept_probes/probe_v1 \\
            --data-path data/positions.jsonl \\
            --model-path models/maia-1500.onnx \\
            --sample-size 10
    """
    from .features import LczeroModel, extract_features_batch

    print("Loading probe...")
    probe = ConceptProbe.load(probe_path)
    print(f"Loaded probe: {probe}")
    print(f"Concepts: {', '.join(probe.concept_list)}")

    print("\nLoading LC0 model...")
    model = LczeroModel.from_path(str(model_path))
    print("Model loaded successfully!")

    print("\nLoading test data...")
    positions: list[LabelledPosition] = []
    with data_path.open() as f:
        for line in f:
            positions.append(LabelledPosition.from_dict(json.loads(line)))

    positions_with_concepts = [p for p in positions if p.concepts]
    print(f"Loaded {len(positions)} positions, kept {len(positions_with_concepts)} with concepts")

    # Filter positions with validated concepts
    filtered_positions: list[LabelledPosition] = []
    for pos in positions_with_concepts:
        validated_concepts = [c.name for c in pos.concepts if c.validated_by is not None]
        if validated_concepts:
            filtered_positions.append(pos)

    print(f"Filtered to {len(filtered_positions)} positions with validated concepts")

    if len(filtered_positions) == 0:
        print("No positions with validated concepts found!")
        return

    # Sample random positions
    n_samples = min(sample_size, len(filtered_positions))
    rng = np.random.RandomState(random_seed)
    sample_indices = rng.choice(len(filtered_positions), size=n_samples, replace=False)
    sample_positions = [filtered_positions[i] for i in sample_indices]

    print(f"\nExtracting activations for {n_samples} samples...")
    fens = [p.fen for p in sample_positions]
    activations = extract_features_batch(
        fens,
        probe.layer_name,
        model=model,
        batch_size=batch_size,
    )
    print(f"Activation matrix shape: {activations.shape}")

    print("\nMaking predictions...")
    predictions_batch = probe.predict_batch(activations)

    print(f"\n{'=' * 70}")
    print(f"SAMPLE PREDICTIONS ({n_samples} examples)")
    print(f"{'=' * 70}\n")

    for pos, predicted_concepts in zip(sample_positions, predictions_batch, strict=True):
        print(f"FEN: {pos.fen}")

        # Get ground truth
        ground_truth_concepts = [c.name for c in pos.concepts if c.validated_by is not None]
        predicted_concept_names = [c.name for c in predicted_concepts]

        # Check if prediction matches
        match_marker = "✓" if set(ground_truth_concepts) == set(predicted_concept_names) else "✗"

        gt_str = ", ".join(ground_truth_concepts) if ground_truth_concepts else "(none)"
        pred_str = ", ".join(predicted_concept_names) if predicted_concept_names else "(none)"
        print(f"Ground Truth: {gt_str}")
        print(f"Prediction:   {pred_str} {match_marker}")
        print()

    # Calculate overall accuracy
    correct = 0
    for pos, predicted_concepts in zip(sample_positions, predictions_batch, strict=True):
        ground_truth_concepts = set(c.name for c in pos.concepts if c.validated_by is not None)
        predicted_concept_names = set(c.name for c in predicted_concepts)
        if ground_truth_concepts == predicted_concept_names:
            correct += 1

    accuracy = correct / n_samples
    print(f"{'=' * 70}")
    print(f"Sample Exact Match Rate: {accuracy:.1%} ({correct}/{n_samples})")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    cli()
