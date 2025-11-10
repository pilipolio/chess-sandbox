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
from huggingface_hub import EvalResult, HfApi, ModelCard, ModelCardData, snapshot_download

from ...config import settings
from .inference import ConceptProbe


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

    def _create_eval_results(self) -> list[EvalResult]:
        """Create EvalResult objects from training_stats for model card."""
        if not self.source_provenance or "source_dataset" not in self.source_provenance:
            return []

        dataset_info = self.source_provenance["source_dataset"]
        dataset_type = dataset_info.get("repo_id", "unknown")
        dataset_revision = dataset_info.get("revision") or "main"

        probe_metrics = self.training_stats.get("probe", {})
        if not probe_metrics:
            return []

        results = []

        # Core metrics (always included)
        metric_mappings = [
            ("exact_match", "subset_accuracy"),  # HF standard name for subset accuracy
            ("micro_precision", "micro_precision"),
            ("micro_recall", "micro_recall"),
            ("macro_precision", "macro_precision"),
            ("macro_recall", "macro_recall"),
        ]

        for metric_name, probe_key in metric_mappings:
            if probe_key in probe_metrics:
                results.append(
                    EvalResult(
                        task_type="tabular-classification",
                        dataset_type=dataset_type,
                        dataset_name="Chess Positions with Concepts",
                        dataset_revision=dataset_revision,
                        metric_type=metric_name,
                        metric_value=probe_metrics[probe_key],
                    )
                )

        # Optional AUC metrics (only if available)
        for auc_metric in ["micro_auc", "macro_auc"]:
            if auc_metric in probe_metrics:
                results.append(
                    EvalResult(
                        task_type="tabular-classification",
                        dataset_type=dataset_type,
                        dataset_name="Chess Positions with Concepts",
                        dataset_revision=dataset_revision,
                        metric_type=auc_metric,
                        metric_value=probe_metrics[auc_metric],
                    )
                )

        return results

    def _create_model_card(self) -> str:
        """Generate model card using HuggingFace default template."""
        # Extract provenance info
        datasets = []
        base_model = None
        dataset_repo_id = None
        github_repo = None

        if self.source_provenance:
            if "source_dataset" in self.source_provenance:
                dataset_repo_id = self.source_provenance["source_dataset"].get("repo_id", "")
                datasets.append(dataset_repo_id)
            if "source_model" in self.source_provenance:
                base_model = self.source_provenance["source_model"].get("repo_id")
            if "training_code" in self.source_provenance:
                training_code = self.source_provenance["training_code"]
                repo_name = training_code.get("repo", "")
                if repo_name:
                    # Convert repo name to GitHub URL
                    github_repo = f"https://github.com/{repo_name}"

        card_data = ModelCardData(
            language="en",
            license="mit",
            library_name="scikit-learn",
            tags=["chess", "concept-detection", "interpretability", "lc0", "multi-label-classification"],
            datasets=datasets if datasets else None,
            base_model=base_model,
            pipeline_tag="tabular-classification",
            model_name="Chess Concept Probe",
            eval_results=self._create_eval_results(),
        )

        mode = self.training_stats.get("mode", "multi-label")
        concept_list = ", ".join(self.probe.concept_list)
        model_description = (
            f"Trained {mode} classifier for detecting {len(self.probe.concept_list)} chess concepts "
            f"({concept_list}) from LC0 layer activations ({self.probe.layer_name})."
        )

        # Add git commit info if available
        if self.source_provenance and "training_code" in self.source_provenance:
            training_code = self.source_provenance["training_code"]
            if training_code.get("commit"):
                repo = training_code.get("repo", "chess-sandbox")
                commit = training_code["commit"]
                commit_url = f"https://github.com/{repo}/commit/{commit}"
                model_description += f"\n\nTrained from [{repo}@{commit}]({commit_url})."

        # Use default HuggingFace template
        card = ModelCard.from_template(
            card_data,
            model_id="chess-concept-probe",
            model_description=model_description,
            developers="chess-sandbox",
            model_type="Multi-label tabular classifier (scikit-learn LogisticRegression)",
            direct_use=(
                f"Extract chess concepts from positions. See {github_repo or 'repository'} for usage examples."
            ),
            out_of_scope_use="Not suitable for non-chess domains or positions outside training distribution.",
            get_started_code=f"See {github_repo or 'repository'} README for complete examples.",
            testing_metrics=(
                "Multi-label metrics: precision, recall, AUC, subset accuracy. See evaluation results above."
            ),
            results=(
                "See evaluation results in model card metadata above and metadata.json "
                "for detailed per-concept breakdown."
            ),
            model_specs=(
                f"sklearn LogisticRegression (C={self.training_stats.get('classifier_c', 1.0)}) "
                "with OneVsRestClassifier wrapper for multi-label classification."
            ),
            model_card_contact=f"{github_repo}/issues" if github_repo else "[More Information Needed]",
            repo=github_repo if github_repo else None,
            training_data=(
                f"Dataset: [{dataset_repo_id}](https://huggingface.co/datasets/{dataset_repo_id})"
                if dataset_repo_id
                else None
            ),
            testing_data=(
                f"Same as training data. See [{dataset_repo_id}](https://huggingface.co/datasets/{dataset_repo_id})"
                if dataset_repo_id
                else None
            ),
            compute_infrastructure="CPU-based training (8 cores typical)",
        )

        return str(card)

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
