"""
HuggingFace Hub integration for concept probes.

Handles model artifact serialization, model cards, and Hub uploads.
"""

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
from huggingface_hub import HfApi, snapshot_download

from ...config import settings
from .inference import ConceptProbe

# Model card template for HuggingFace Hub
MODEL_CARD_TEMPLATE = """---
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

Trained {mode} classifier for detecting chess concepts from LC0 layer activations.

## Model Description

Detects {n_concepts} chess concepts from internal activations of Leela Chess Zero (LC0) models:

{concept_list}

**Layer:** `{layer_name}`
**Mode:** {mode}
**Training Date:** {training_date}

## Performance

- **{key_metric}**
- {secondary_metric}

### Per-Concept Performance

| Concept | Accuracy | F1 Score | Support |
|---------|----------|----------|---------|
{per_concept_table}

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
    layer_name="{layer_name}"
)

# Predict concepts
concepts = probe.predict(features)
```

## Training Details

- **Training samples:** {n_train:,}
- **Test samples:** {n_test:,}
- **Test split:** {test_split:.1%}
- **Random seed:** {random_seed}
"""


@dataclass
class ModelTrainingOutput:
    """
    Training output containing probe, metrics, and provenance.

    Wraps a trained ConceptProbe with training statistics and source provenance
    for versioning and reproducibility. Handles save/load/HF Hub operations.

    Attributes:
        probe: Trained ConceptProbe for inference
        training_stats: Training metrics dict with baseline/probe metrics
        source_provenance: Source dataset and model information (optional)
        training_date: ISO format datetime string
    """

    probe: ConceptProbe
    training_stats: dict[str, Any]
    source_provenance: dict[str, Any] | None
    training_date: str

    def save(self, path: str | Path) -> None:
        """
        Save training output in HuggingFace snapshot format.

        Creates directory with:
        - model.joblib: Sklearn classifier
        - encoder.joblib: Label encoder (if exists)
        - metadata.json: Training stats, provenance, and dependencies
        - README.md: Model card

        Args:
            path: Path to save directory (e.g., "models/probe_v1")
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save model components
        joblib.dump(self.probe.classifier, path / "model.joblib")
        if self.probe.label_encoder is not None:
            joblib.dump(self.probe.label_encoder, path / "encoder.joblib")

        # Save metadata
        (path / "metadata.json").write_text(json.dumps(self._create_metadata(), indent=2))
        (path / "README.md").write_text(self._create_model_card())

        print(f"Saved training output to {path}/ (HF snapshot format)")

    @classmethod
    def load(cls, path: str | Path) -> "ModelTrainingOutput":
        """
        Load training output from HF snapshot format directory.

        Args:
            path: Path to training output directory

        Returns:
            Loaded ModelTrainingOutput instance
        """
        path = Path(path)

        if not path.is_dir():
            msg = f"Path must be directory (HF format): {path}"
            raise ValueError(msg)

        # Load model components
        classifier = joblib.load(path / "model.joblib")
        encoder_path = path / "encoder.joblib"
        label_encoder = joblib.load(encoder_path) if encoder_path.exists() else None

        # Load metadata
        metadata = json.loads((path / "metadata.json").read_text())

        # Extract probe-specific fields
        model_config = metadata["model_config"]
        probe = ConceptProbe(
            classifier=classifier,
            concept_list=model_config["concepts"],
            layer_name=model_config["layer_name"],
            label_encoder=label_encoder,
        )

        return cls(
            probe=probe,
            training_stats=metadata["performance"],
            source_provenance=metadata.get("provenance"),
            training_date=model_config["training_date"],
        )

    @classmethod
    def from_hf(
        cls,
        repo_id: str,
        *,
        revision: str | None = None,
        cache_dir: Path | str | None = None,
        force_download: bool = False,
        token: str | None = None,
    ) -> "ModelTrainingOutput":
        """
        Load training output from HuggingFace Hub.

        Args:
            repo_id: Repository ID (e.g., "pilipolio/chess-sandbox-concept-probes")
            revision: Git revision (tag, branch, commit). Defaults to "main"
            cache_dir: Custom cache directory (defaults to ~/.cache/huggingface/)
            force_download: Force re-download even if cached
            token: HF authentication token for private repos

        Returns:
            Loaded ModelTrainingOutput instance
        """
        cache_dir = cache_dir or settings.HF_CACHE_DIR

        print(
            f"Loading training output from {repo_id} with cache dir {cache_dir} and force download {force_download}..."
        )
        token = token or settings.HF_TOKEN or None

        local_dir = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            token=token,
        )

        return cls.load(local_dir)

    def upload_to_hf(
        self,
        local_dir: Path | str,
        repo_id: str,
        revision: str | None = None,
        token: str | None = None,
        commit_message: str | None = None,
    ) -> str:
        """
        Upload probe directory to HuggingFace Hub.

        The probe must already be saved to local_dir using save().

        Args:
            local_dir: Path to saved probe directory
            repo_id: Full HuggingFace repository ID (e.g., "pilipolio/chess-sandbox-concept-probes")
            revision: Optional revision/tag name (included in commit message if provided)
            token: HF token (defaults to settings.HF_TOKEN)
            commit_message: Custom commit message (auto-generated if not provided)

        Returns:
            Commit URL from HuggingFace Hub
        """
        local_dir = Path(local_dir)

        # TODO: write down locally if not exists
        if not local_dir.is_dir():
            msg = f"Probe directory does not exist: {local_dir}"
            raise ValueError(msg)

        token = token or settings.HF_TOKEN or None

        api = HfApi(token=token)
        return api.upload_folder(
            folder_path=local_dir,
            repo_id=repo_id,
            repo_type="model",
            revision=revision,  # TODO: currently failing when not main...
        )

    def _create_metadata(self) -> dict[str, Any]:
        """Create structured metadata for HF Hub."""
        mode = self.training_stats.get("mode", "multi-label")

        metadata: dict[str, Any] = {
            "model_config": {
                "layer_name": self.probe.layer_name,
                "mode": mode,
                "n_concepts": len(self.probe.concept_list),
                "concepts": self.probe.concept_list,
                "training_date": self.training_date,
            },
            "training_info": {
                "n_train": self.training_stats.get("training_samples", 0),
                "n_test": self.training_stats.get("test_samples", 0),
                "test_split": self.training_stats.get("test_split", 0.2),
                "random_seed": self.training_stats.get("random_seed", 42),
                "training_params": {
                    "verbose": self.training_stats.get("verbose", False),
                    "n_jobs": self.training_stats.get("n_jobs", -1),
                },
            },
            "performance": self.training_stats,
            "dependencies": {
                "sklearn": self._get_package_version("sklearn"),
                "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "lczerolens": self._get_package_version("lczerolens"),
                "torch": self._get_package_version("torch"),
            },
        }

        if self.source_provenance:
            metadata["provenance"] = self.source_provenance

        return metadata

    def _create_model_card(self) -> str:
        """Generate model card with YAML frontmatter."""
        mode = self.training_stats.get("mode", "multi-label")
        probe_metrics = self.training_stats.get("probe", {})

        # TODO: add precision, recall
        key_metric = f"Exact Match: {probe_metrics.get('exact_match', 0):.1%}"

        # Build per-concept table
        per_concept_rows = []
        per_concept = probe_metrics.get("per_concept", {})
        for concept in self.probe.concept_list:
            if concept in per_concept:
                c_metrics = per_concept[concept]
                per_concept_rows.append(
                    f"| {concept} | {c_metrics['accuracy']:.3f} | {c_metrics['f1']:.3f} | {c_metrics['support']} |"
                )
        per_concept_table = "\n".join(per_concept_rows)

        # Format concept list
        concept_list = ", ".join(f"`{c}`" for c in self.probe.concept_list)

        return MODEL_CARD_TEMPLATE.format(
            mode=mode,
            n_concepts=len(self.probe.concept_list),
            concept_list=concept_list,
            layer_name=self.probe.layer_name,
            training_date=self.training_date[:10],
            key_metric=key_metric,
            secondary_metric="",
            per_concept_table=per_concept_table,
            n_train=self.training_stats.get("training_samples", 0),
            n_test=self.training_stats.get("test_samples", 0),
            test_split=self.training_stats.get("test_split", 0.2),
            random_seed=self.training_stats.get("random_seed", 42),
        )

    @staticmethod
    def _get_package_version(package: str) -> str:
        """Get package version."""
        try:
            import importlib.metadata

            return importlib.metadata.version(package)
        except Exception:
            return "unknown"


def generate_probe_name(model_repo_id: str, layer_name: str, mode: str, dataset_repo_id: str) -> str:
    """
    Generate a semantic probe name from training parameters.

    Format: {model_name}_{layer_safe}_{mode}_{dataset_hash}

    Args:
        model_repo_id: HuggingFace model repo ID (e.g., "pilipolio/maia-1500")
        layer_name: Layer name (e.g., "block3/conv2/relu")
        mode: Training mode ("multi-label" or "multi-class")
        dataset_repo_id: HuggingFace dataset repo ID

    Returns:
        Generated probe name (e.g., "maia1500_block3_conv2_relu_multilabel_a3f8c2d")
    """
    model_name = model_repo_id.split("/")[-1].replace("-", "").replace("_", "")
    layer_safe = layer_name.replace("/", "_").replace("-", "_")
    mode_safe = mode.replace("-", "")
    dataset_hash = hashlib.sha256(dataset_repo_id.encode()).hexdigest()[:8]
    return f"{model_name}_{layer_safe}_{mode_safe}_{dataset_hash}"
