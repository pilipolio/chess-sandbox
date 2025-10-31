"""
Board encoding and activation extraction for concept probing.

Converts chess positions to LC0's 112×8×8 tensor format and extracts
internal layer activations from LC0 models using PyTorch hooks.

Adapted from prototype implementations with minimal dependencies.
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from lczerolens import LczeroBoard  # type: ignore[import-untyped]
from lczerolens import LczeroModel as _LczeroModel
from tqdm import tqdm  # type: ignore[import-untyped]


class LczeroModel:
    """
    Wrapper around lczerolens.LczeroModel for PyTorch 2.6+ compatibility.

    PyTorch 2.6+ changed torch.load() to use weights_only=True by default for security.
    LC0 models contain FX graph modules that aren't allowed under this restriction.
    This wrapper temporarily patches torch.load to use weights_only=False.

    TODO: Remove once lczerolens adds weights_only parameter support.
    See: https://github.com/Xmaster6y/lczerolens/issues/XXX
    """

    @classmethod
    def from_path(cls, model_path: str) -> Any:
        """
        Load LC0 model with PyTorch 2.6+ compatibility.

        Args:
            model_path: Path to LC0 model file (.pt or .onnx)

        Returns:
            Loaded LczeroModel instance

        Note:
            Temporarily uses weights_only=False for compatibility with FX graph modules.
            Only use with trusted model files.
        """
        original_load = torch.load

        def patched_load(*args: Any, **kwargs: Any) -> Any:
            kwargs["weights_only"] = False
            return original_load(*args, **kwargs)

        try:
            torch.load = patched_load  # type: ignore[assignment]
            return _LczeroModel.from_path(model_path)
        finally:
            torch.load = original_load  # type: ignore[assignment]


class ActivationExtractor:
    """
    Extract activations from LC0 models using PyTorch forward hooks.

    Example:
        >>> True  # doctest: +SKIP
        True
    """

    def __init__(self, model: Any, layer_names: list[str], device: str | None = None):
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
                print(f"Warning: Could not register hook for '{name}': {e}")

    def _get_layer_by_name(self, name: str) -> Any:
        """Get layer module by hierarchical name."""
        module = self.model.module if hasattr(self.model, "module") else self.model

        try:
            return module.get_submodule(name)
        except AttributeError:
            alt_name = name.replace("/", ".") if "/" in name else name.replace(".", "/")
            return module.get_submodule(alt_name)

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
            _ = module(tensor)

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
            _ = module(tensor)

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


def extract_features(
    fen: str,
    model_path: str | Path,
    layer_name: str,
) -> np.ndarray:
    """
    Extract flattened activation vector from a chess position.

    Args:
        fen: Chess position in FEN notation
        model_path: Path to LC0 model file
        layer_name: Layer to extract from (e.g., "block3/conv2/relu")

    Returns:
        Flattened activation vector (e.g., shape (4096,) for 64×8×8)

    Example:
        >>> features = extract_features(  # doctest: +SKIP
        ...     "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        ...     "models/maia-1500.pt",
        ...     "block3/conv2/relu"
        ... )
        >>> features.shape  # doctest: +SKIP
        (4096,)
    """
    model = LczeroModel.from_path(str(model_path))
    with ActivationExtractor(model, [layer_name]) as extractor:
        board = LczeroBoard(fen)
        activations = extractor.extract(board)

    activation = activations[layer_name].cpu().numpy()
    return activation.reshape(-1)


def extract_features_batch(
    fens: list[str],
    model_path: str | Path,
    layer_name: str,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Extract flattened activation vectors from multiple chess positions.

    Args:
        fens: List of chess positions in FEN notation
        model_path: Path to LC0 model file
        layer_name: Layer to extract from
        batch_size: Number of positions to process at once
        show_progress: Whether to print progress

    Returns:
        Array of shape (n_positions, n_features)

    Example:
        >>> fens = ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"] * 10  # doctest: +SKIP
        >>> features = extract_features_batch(fens, "models/maia-1500.pt", "block3/conv2/relu")  # doctest: +SKIP
        >>> features.shape  # doctest: +SKIP
        (10, 4096)
    """
    print(f"Extracting activations for {len(fens)} positions...")

    results: list[np.ndarray] = []
    model = LczeroModel.from_path(str(model_path))

    with ActivationExtractor(model, [layer_name]) as extractor:
        for i in tqdm(range(0, len(fens), batch_size)):
            batch_fens = fens[i : i + batch_size]
            boards = [LczeroBoard(fen) for fen in batch_fens]
            activations = extractor.extract_batch(boards)
            batch_activations: Any = activations[layer_name].cpu().numpy()

            for j in range(len(batch_fens)):
                flattened: np.ndarray = batch_activations[j].reshape(-1)
                results.append(flattened)

    return np.array(results)


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
