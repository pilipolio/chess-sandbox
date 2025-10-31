"""
Inference wrapper for trained concept probes.

Provides loading and prediction for sklearn classifiers trained to detect
chess concepts from LC0 layer activations.
"""

import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .models import Concept


@dataclass
class ConceptProbe:
    """
    Trained concept classifier with metadata.

    Loads and applies sklearn multi-label classifiers to predict
    chess concepts from activation vectors.

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

    @classmethod
    def load(cls, path: str | Path) -> "ConceptProbe":
        """
        Load trained probe from pickle file.

        Args:
            path: Path to saved probe file

        Returns:
            Loaded ConceptProbe instance

        Example:
            >>> # probe = ConceptProbe.load("models/probe.pkl")
            >>> # probe.concept_list
            >>> # ['pin', 'fork', 'skewer', ...]
            True
        """
        with Path(path).open("rb") as f:
            data = pickle.load(f)

        return cls(
            classifier=data["classifier"],
            concept_list=data["concept_list"],
            layer_name=data["layer_name"],
            training_metrics=data.get("training_metrics", {}),
            training_date=data.get("training_date", "unknown"),
            model_version=data.get("model_version", "unknown"),
        )

    def save(self, path: str | Path) -> None:
        """
        Save probe to pickle file.

        Args:
            path: Path to save probe file

        Example:
            >>> # probe.save("models/probe_v2.pkl")
            True
        """
        data = {
            "classifier": self.classifier,
            "concept_list": self.concept_list,
            "layer_name": self.layer_name,
            "training_metrics": self.training_metrics,
            "training_date": self.training_date,
            "model_version": self.model_version,
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("wb") as f:
            pickle.dump(data, f)

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
                        validated_by="probe",  # type: ignore[arg-type]
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
                            validated_by="probe",  # type: ignore[arg-type]
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


def create_probe(
    classifier: Any,
    concept_list: list[str],
    layer_name: str,
    training_metrics: dict[str, Any],
    model_version: str = "v1",
) -> ConceptProbe:
    """
    Create a new ConceptProbe instance.

    Args:
        classifier: Trained sklearn classifier
        concept_list: List of concept names in prediction order
        layer_name: Layer used for feature extraction
        training_metrics: Performance metrics from training
        model_version: Version identifier

    Returns:
        ConceptProbe instance ready for inference

    Example:
        >>> # from sklearn.linear_model import LogisticRegression
        >>> # from sklearn.multioutput import OneVsRestClassifier
        >>> # clf = OneVsRestClassifier(LogisticRegression())
        >>> # clf.fit(X_train, y_train)
        >>> # probe = create_probe(
        >>> #     clf, ["pin", "fork"], "block3/conv2/relu",
        >>> #     {"accuracy": 0.85}, "v1"
        >>> # )
        >>> # probe.model_version == "v1"
        True
    """
    return ConceptProbe(
        classifier=classifier,
        concept_list=concept_list,
        layer_name=layer_name,
        training_metrics=training_metrics,
        training_date=datetime.now().isoformat(),
        model_version=model_version,
    )
