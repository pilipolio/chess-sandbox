"""
Board encoding and activation extraction for concept probing.

Converts chess positions to LC0's 112×8×8 tensor format and extracts
internal layer activations from LC0 models using PyTorch hooks.

Adapted from prototype implementations with minimal dependencies.
"""

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lczerolens import (  # type: ignore[import-untyped]
    LczeroBoard,
    LczeroModel,
)
from tqdm import tqdm

# Suppress PyTorch 2.9 deprecation warning from onnx2torch library
warnings.filterwarnings("ignore", category=UserWarning, module="onnx2torch")


class ActivationExtractor:
    """
    Extract activations from LC0 models using PyTorch forward hooks.

    Example:
        >>> True  # doctest: +SKIP
        True
    """

    def __init__(self, model: LczeroModel, layer_names: list[str], device: str | None = None):
        """
        Initialize activation extractor.

        Args:
            model: Loaded LC0 model (from lczerolens)
            layer_names: List of layer names to hook (e.g., ["block3/conv2/relu"])
            device: Device to use ("cuda", "cpu", or None for auto)
        """
        self.model = model
        self.layer_names = layer_names
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.activations: dict[str, torch.Tensor] = {}
        self.hooks: list[Any] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward hooks on target layers."""
        for name in self.layer_names:
            try:
                layer = self._get_layer_by_name(name)
                hook = layer.register_forward_hook(self._make_hook(name))
                self.hooks.append(hook)
            except Exception as e:
                raise ValueError(
                    f"Could not register hook for layer '{name}', available layers: {self.model.module.layer_names}"
                ) from e

    def _get_layer_by_name(self, name: str) -> Any:
        """Get layer module by hierarchical name."""
        module = self.model.module if hasattr(self.model, "module") else self.model

        try:
            return module.get_submodule(name)  # type: ignore[arg-type]
        except AttributeError:
            alt_name = name.replace("/", ".") if "/" in name else name.replace(".", "/")
            return module.get_submodule(alt_name)  # type: ignore[arg-type]

    def _make_hook(self, name: str) -> Any:
        """Create a hook function for capturing activations."""

        def hook(module: Any, input_: Any, output: torch.Tensor) -> None:
            self.activations[name] = output.detach()

        return hook

    def extract(self, board: LczeroBoard) -> dict[str, torch.Tensor]:
        """
        Extract activations for a single board position.

        Args:
            board: Encoded chess position

        Returns:
            Dict mapping layer names to activation tensors
        """
        tensor = board.to_input_tensor().unsqueeze(0)
        tensor = tensor.to(self.device)

        self.activations.clear()

        with torch.no_grad():
            module = self.model.module if hasattr(self.model, "module") else self.model
            _ = module(tensor)  # type: ignore[arg-type]

        return self.activations.copy()

    def extract_batch(self, boards: list[LczeroBoard]) -> dict[str, torch.Tensor]:
        """
        Extract activations for multiple boards (batched).

        Args:
            boards: List of encoded chess positions

        Returns:
            Dict mapping layer names to batched activation tensors
        """
        tensors = [b.to_input_tensor() for b in boards]
        tensor = torch.stack(tensors)
        tensor = tensor.to(self.device)

        self.activations.clear()

        with torch.no_grad():
            module = self.model.module if hasattr(self.model, "module") else self.model
            _ = module(tensor)  # type: ignore[arg-type]

        return self.activations.copy()

    def cleanup(self) -> None:
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def __enter__(self) -> "ActivationExtractor":
        """Context manager support."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Auto-cleanup when exiting context."""
        self.cleanup()


def extract_features_batch(
    fens: list[str],
    layer_name: str,
    model: LczeroModel,
    batch_size: int = 512,
) -> np.ndarray:
    """
    Extract flattened activation vectors from multiple chess positions.

    Args:
        fens: List of chess positions in FEN notation
        layer_name: Layer to extract from
        model: Pre-loaded LC0 model (preferred for efficiency)
        model_path: Path to LC0 model file (loaded if model not provided)
        batch_size: Number of positions to process at once

    Returns:
        Array of shape (n_positions, n_features)

    Note:
        Provide either model or model_path. Using a pre-loaded model avoids
        repeated loading overhead when processing multiple batches.

    Example:
        >>> fens = ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"] * 10  # doctest: +SKIP
        >>> features = extract_features_batch(  # doctest: +SKIP
        ...     fens, "block3/conv2/relu", model_path="models/maia-1500.onnx"
        ... )
        >>> features.shape  # doctest: +SKIP
        (10, 4096)
    """

    print(f"Extracting activations for {len(fens)} positions in batch of size {batch_size}...")

    results: list[np.ndarray] = []

    with ActivationExtractor(model, [layer_name]) as extractor:
        for i in tqdm(range(0, len(fens), batch_size)):
            batch_fens = fens[i : i + batch_size]
            boards = [LczeroBoard(fen) for fen in batch_fens]
            activations = extractor.extract_batch(boards)
            batch_activations = activations[layer_name].cpu().numpy()

            for j in range(len(batch_fens)):
                flattened = batch_activations[j].reshape(-1)
                results.append(flattened)

    return np.array(results)


def compute_square_saliency(
    activations: np.ndarray,
    aggregation: str = "mean",
) -> np.ndarray:
    """
    Compute per-square saliency from spatial activation maps.

    Takes activation maps from convolutional layers with spatial structure
    (channels, height, width) and computes a saliency score for each square
    by aggregating across channels.

    Args:
        activations: Activation tensor of shape (channels, 8, 8) or (batch, channels, 8, 8)
        aggregation: Method to aggregate across channels ("mean", "max", "l2")

    Returns:
        Saliency map of shape (8, 8) or (batch, 8, 8) with scores for each square

    Example:
        >>> activations = np.random.rand(64, 8, 8)  # doctest: +SKIP
        >>> saliency = compute_square_saliency(activations)  # doctest: +SKIP
        >>> saliency.shape  # doctest: +SKIP
        (8, 8)
    """
    if activations.ndim == 3:
        channels, height, width = activations.shape
        if height != 8 or width != 8:
            raise ValueError(f"Expected spatial dimensions of 8x8, got {height}x{width}")

        if aggregation == "mean":
            return np.mean(np.abs(activations), axis=0)
        elif aggregation == "max":
            return np.max(np.abs(activations), axis=0)
        elif aggregation == "l2":
            return np.sqrt(np.sum(activations**2, axis=0))
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")

    elif activations.ndim == 4:
        batch, channels, height, width = activations.shape
        if height != 8 or width != 8:
            raise ValueError(f"Expected spatial dimensions of 8x8, got {height}x{width}")

        if aggregation == "mean":
            return np.mean(np.abs(activations), axis=1)
        elif aggregation == "max":
            return np.max(np.abs(activations), axis=1)
        elif aggregation == "l2":
            return np.sqrt(np.sum(activations**2, axis=1))
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")

    else:
        raise ValueError(f"Expected 3D or 4D activations, got shape {activations.shape}")


def extract_features_and_saliency_batch(
    fens: list[str],
    layer_name: str,
    model: LczeroModel,
    batch_size: int = 512,
    saliency_aggregation: str = "mean",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract both flattened features and per-square saliency maps.

    Args:
        fens: List of chess positions in FEN notation
        layer_name: Layer to extract from (should be a spatial convolutional layer)
        model: Pre-loaded LC0 model
        batch_size: Number of positions to process at once
        saliency_aggregation: Method to aggregate saliency across channels

    Returns:
        Tuple of (features, saliency_maps) where:
        - features: Array of shape (n_positions, n_features)
        - saliency_maps: Array of shape (n_positions, 8, 8)

    Example:
        >>> fens = ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"] * 10  # doctest: +SKIP
        >>> features, saliency = extract_features_and_saliency_batch(  # doctest: +SKIP
        ...     fens, "block3/conv2/relu", model
        ... )
        >>> features.shape, saliency.shape  # doctest: +SKIP
        ((10, 4096), (10, 8, 8))
    """
    print(f"Extracting activations and saliency for {len(fens)} positions in batch of size {batch_size}...")

    features_list: list[np.ndarray] = []
    saliency_list: list[np.ndarray] = []

    with ActivationExtractor(model, [layer_name]) as extractor:
        for i in tqdm(range(0, len(fens), batch_size)):
            batch_fens = fens[i : i + batch_size]
            boards = [LczeroBoard(fen) for fen in batch_fens]
            activations = extractor.extract_batch(boards)
            batch_activations = activations[layer_name].cpu().numpy()

            for j in range(len(batch_fens)):
                flattened = batch_activations[j].reshape(-1)
                features_list.append(flattened)

                spatial_activations = batch_activations[j]
                saliency = compute_square_saliency(spatial_activations, aggregation=saliency_aggregation)
                saliency_list.append(saliency)

    return np.array(features_list), np.array(saliency_list)


def list_available_layers(model_path: str | Path) -> list[str]:
    """
    List all available layer names in a model.

    Args:
        model_path: Path to LC0 model file

    Returns:
        List of layer names that can be hooked
    """
    model = LczeroModel.from_path(str(model_path))
    module: Any = model.module if hasattr(model, "module") else model

    layers: list[str] = []
    for name, _ in module.named_modules():
        if name:
            layers.append(name)

    return sorted(layers)
