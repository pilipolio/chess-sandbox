"""
Inference wrapper for trained concept probes.

Provides loading and prediction for sklearn classifiers trained to detect
chess concepts from LC0 layer activations.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click
import numpy as np
from huggingface_hub import hf_hub_download
from pydantic import BaseModel

from ...config import settings
from ..labelling.labeller import Concept, LabelledPosition
from .features import extract_features_batch


class ConceptResponse(BaseModel):
    """Pydantic model for concept in API responses."""

    name: str
    validated_by: str | None = None
    temporal: str | None = None
    reasoning: str | None = None

    @classmethod
    def from_concept(cls, concept: Concept) -> "ConceptResponse":
        """Convert dataclass Concept to Pydantic model."""
        return cls(
            name=concept.name,
            validated_by=concept.validated_by,
            temporal=concept.temporal,
            reasoning=concept.reasoning,
        )


class ConceptPredictionResponse(BaseModel):
    """Response model for single position concept prediction."""

    fen: str
    concepts: list[ConceptResponse]
    threshold: float

    @classmethod
    def from_concepts(cls, fen: str, concepts: list[Concept], threshold: float) -> "ConceptPredictionResponse":
        """Create response from list of Concept dataclasses."""
        return cls(
            fen=fen,
            concepts=[ConceptResponse.from_concept(c) for c in concepts],
            threshold=threshold,
        )


class ConceptConfidenceResponse(BaseModel):
    """Response model for concept predictions with confidence scores."""

    fen: str
    predictions: list[tuple[str, float]]

    @classmethod
    def from_predictions(cls, fen: str, predictions: list[tuple[str, float]]) -> "ConceptConfidenceResponse":
        """Create response from predictions with confidence scores."""
        return cls(
            fen=fen,
            predictions=predictions,
        )


@dataclass
class ConceptProbe:
    """
    Trained concept classifier for inference.

    Applies sklearn classifiers to predict chess concepts from activation
    vectors. Supports both multi-class (one concept per position) and
    multi-label (multiple concepts per position).

    Example:
        >>> from chess_sandbox.concept_extraction.model.train import ModelTrainingOutput  # doctest: +SKIP
        >>> output = ModelTrainingOutput.load("models/training_output")  # doctest: +SKIP
        >>> probe = output.probe  # doctest: +SKIP
        >>> features = np.random.rand(4096)  # From feature extraction  # doctest: +SKIP
        >>> concepts = probe.predict(features)  # doctest: +SKIP
        >>> len(concepts) >= 0  # doctest: +SKIP
        True
    """

    classifier: Any
    concept_list: list[str]
    layer_name: str
    label_encoder: Any = None

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
        return f"ConceptProbe(concepts={len(self.concept_list)}, " f"layer={self.layer_name})"


@dataclass
class ConceptExtractor:
    """
    High-level wrapper for extracting chess concepts from FEN positions.

    Combines LC0 model loading, feature extraction, and concept prediction
    into a single convenient interface. Downloads all dependencies from
    HuggingFace Hub for zero-setup usage.

    Example:
        >>> extractor = ConceptExtractor.from_hf(  # doctest: +SKIP
        ...     probe_repo_id="pilipolio/chess-positions-extractor"
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
            probe_repo_id: Probe repository (e.g., "pilipolio/chess-positions-extractor")
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
            ...     probe_repo_id="pilipolio/chess-positions-extractor"
            ... )
            >>> extractor.layer_name  # doctest: +SKIP
            'block3/conv2/relu'
        """
        from .features import LczeroModel
        from .model_artefact import ModelTrainingOutput

        cache_dir = cache_dir or settings.HF_CACHE_DIR
        token = token or settings.HF_TOKEN or None

        print(f"Downloading training output from {probe_repo_id}...")
        training_output = ModelTrainingOutput.from_hf(
            repo_id=probe_repo_id,
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            token=token,
        )
        probe = training_output.probe
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

        features_batch = extract_features_batch(
            model=self.model,
            fens=[fen],
            layer_name=self.layer_name,
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


def hf_hub_options(f):
    """Decorator to add common HuggingFace Hub options to CLI commands."""
    f = click.option(
        "--token",
        type=str,
        help="HuggingFace authentication token for private repos",
    )(f)
    f = click.option(
        "--force-download",
        is_flag=True,
        help="Force re-download even if cached",
    )(f)
    f = click.option(
        "--cache-dir",
        type=click.Path(path_type=Path),
        help="Custom cache directory (defaults to ~/.cache/huggingface/)",
    )(f)
    f = click.option(
        "--lc0-filename",
        default="model.onnx",
        type=str,
        help="LC0 model filename to download (default: model.onnx)",
    )(f)
    f = click.option(
        "--lc0-repo-id",
        default="lczerolens/maia-1500",
        type=str,
        help="LC0 model repository ID (default: lczerolens/maia-1500)",
    )(f)
    f = click.option(
        "--revision",
        type=str,
        help="Git revision for model (tag, branch, commit). Defaults to 'main'",
    )(f)
    f = click.option(
        "--model-repo-id",
        required=True,
        type=str,
        default="pilipolio/chess-positions-extractor",
        help="HuggingFace repository ID",
    )(f)
    return f


@click.group()
def cli() -> None:
    """Concept probe inference CLI."""
    pass


@cli.command()
@click.argument("fen", type=str)
@hf_hub_options
@click.option(
    "--min-confidence",
    default=0.1,
    type=float,
    help="Minimum confidence to display (default: 0.1)",
)
def predict(
    fen: str,
    model_repo_id: str,
    lc0_repo_id: str,
    lc0_filename: str,
    min_confidence: float,
    revision: str | None,
    cache_dir: Path | None,
    force_download: bool,
    token: str | None,
) -> None:
    """
    Predict concepts for a single FEN position with confidence scores.

    Example:
        uv run python -m chess_sandbox.concept_extraction.model.inference predict \\
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

        # With custom threshold:
        uv run python -m chess_sandbox.concept_extraction.model.inference predict \\
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" \\
            --min-confidence 0.3
    """
    print("Loading ConceptExtractor from HuggingFace Hub...")
    extractor = ConceptExtractor.from_hf(
        probe_repo_id=model_repo_id,
        model_repo_id=lc0_repo_id,
        model_filename=lc0_filename,
        revision=revision,
        cache_dir=cache_dir,
        force_download=force_download,
        token=token,
    )

    print(f"\nExtracting concepts for FEN: {fen}")
    predictions_with_confidence = extractor.extract_concepts_with_confidence(fen)

    # Sort by confidence descending
    predictions_with_confidence.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{'=' * 70}")
    print("PREDICTIONS")
    print(f"{'=' * 70}\n")
    print(f"FEN: {fen}\n")

    # Filter by minimum confidence
    filtered_predictions = [(c, conf) for c, conf in predictions_with_confidence if conf >= min_confidence]

    if filtered_predictions:
        print(f"Predicted Concepts (confidence ≥ {min_confidence:.0%}):")
        for concept, confidence in filtered_predictions:
            print(f"  {concept}: {confidence:.2%}")
    else:
        print(f"No concepts predicted above {min_confidence:.0%} confidence")

    print()


@cli.command()
@hf_hub_options
@click.option(
    "--data-path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to JSONL file with labeled positions",
)
@click.option(
    "--batch-size",
    default=32,
    type=int,
    help="Batch size for activation extraction",
)
def batch_predict(
    model_repo_id: str,
    revision: str | None,
    lc0_repo_id: str,
    lc0_filename: str,
    cache_dir: Path | None,
    force_download: bool,
    token: str | None,
    data_path: Path,
    batch_size: int,
) -> None:
    """
    Predict concepts for multiple positions from JSONL file with confidence scores.

    Example:
        python -m chess_sandbox.concept_extraction.model.inference batch-predict \\
            --model-repo-id pilipolio/chess-positions-extractor \\
            --data-path data/processed/test_labeled_positions.jsonl
    """
    print("Loading ConceptExtractor from HuggingFace Hub...")
    extractor = ConceptExtractor.from_hf(
        probe_repo_id=model_repo_id,
        model_repo_id=lc0_repo_id,
        model_filename=lc0_filename,
        revision=revision,
        cache_dir=cache_dir,
        force_download=force_download,
        token=token,
    )

    print("\nLoading positions from JSONL...")
    positions: list[LabelledPosition] = []
    with data_path.open() as f:
        for line in f:
            positions.append(LabelledPosition.from_dict(json.loads(line)))

    print(f"Loaded {len(positions)} positions")

    print(f"\nExtracting activations for {len(positions)} positions...")
    from .features import extract_features_batch

    fens = [p.fen for p in positions]
    activations = extract_features_batch(
        fens,
        extractor.layer_name,
        model=extractor.model,
        batch_size=batch_size,
    )
    print(f"Activation matrix shape: {activations.shape}")

    print("\nMaking predictions...")

    print(f"\n{'=' * 70}")
    print(f"BATCH PREDICTIONS ({len(positions)} positions)")
    print(f"{'=' * 70}\n")

    for i, pos in enumerate(positions):
        features = activations[i : i + 1]
        predictions_with_confidence = extractor.probe.predict_with_confidence(features)
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


if __name__ == "__main__":
    cli()
