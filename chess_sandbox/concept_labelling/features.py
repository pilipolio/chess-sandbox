"""
Board encoding and activation extraction for concept probing.

Converts chess positions to LC0's 112×8×8 tensor format and extracts
internal layer activations from LC0 models using PyTorch hooks.

Adapted from prototype implementations with minimal dependencies.
"""

import hashlib
import struct
from pathlib import Path
from typing import Any, NamedTuple

import chess
import diskcache  # type: ignore[import-untyped]
import numpy as np
import torch
from lczerolens import LczeroModel as _LczeroModel  # type: ignore[import-untyped]


class LczeroModel:
    """Wrapper around lczerolens.LczeroModel that handles PyTorch 2.6+ compatibility."""

    @classmethod
    def from_path(cls, model_path: str) -> Any:
        """
        Load LC0 model with weights_only=False for PyTorch 2.6+ compatibility.

        PyTorch 2.6+ changed torch.load to use weights_only=True by default for security,
        but LC0 models contain FX graph modules and other classes that aren't allowed.
        We trust these model files, so we temporarily use the legacy loading behavior.
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


_CACHE_DIR = Path(".cache/activations")
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_cache: Any = diskcache.Cache(str(_CACHE_DIR))

_FLAT_PLANES = [np.ones((8, 8), dtype=np.uint8) * i for i in range(256)]


class _Lc0BoardData(NamedTuple):
    plane_bytes: bytes
    repetition: bool
    transposition_key: int
    us_ooo: int
    us_oo: int
    them_ooo: int
    them_oo: int
    side_to_move: int
    rule50_count: int


class Lc0BoardEncoder:
    """
    Encoder for converting chess positions to Leela Chess Zero's 112-plane format.

    Maintains position history needed for LC0 encoding.

    Example:
        >>> board = Lc0BoardEncoder("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
        >>> encoding = board.to_input_array()
        >>> encoding.shape  # doctest: +SKIP
        (112, 8, 8)
    """

    _plane_bytes_struct = struct.Struct(">Q")

    def __init__(self, fen: str | None = None):
        """
        Initialize the board encoder.

        Args:
            fen: FEN string for initial position. If None, starts from standard position.
        """
        self._board = chess.Board() if fen is None else chess.Board(fen)
        self._history_stack: list[_Lc0BoardData] = []
        self._transposition_counter: dict[int, int] = {}
        self._push_history()

    @property
    def turn(self) -> bool:
        """Current player's turn (True=White, False=Black)"""
        return self._board.turn

    @property
    def fen(self) -> str:
        """Current position as FEN string"""
        return self._board.fen()

    def push(self, move: chess.Move) -> None:
        """Make a move on the board."""
        self._board.push(move)
        self._push_history()

    def push_uci(self, uci: str) -> None:
        """Make a move from UCI string (e.g., 'e2e4', 'e7e8q')."""
        self._board.push(chess.Move.from_uci(uci))
        self._push_history()

    def push_san(self, san: str) -> None:
        """Make a move from SAN string (e.g., 'e4', 'Nf3', 'O-O')."""
        self._board.push_san(san)
        self._push_history()

    def _get_plane_bytes(self) -> Any:
        """Extract piece positions as packed bytes."""
        pack = self._plane_bytes_struct.pack
        pieces_mask = self._board.pieces_mask

        for color in (True, False):
            for piece_type in range(1, 7):
                piece_mask = pieces_mask(piece_type, color)
                yield pack(piece_mask)

    def _push_history(self) -> None:
        """Update history stack after a move."""
        transposition_key: int = self._board._transposition_key()  # type: ignore[attr-defined]
        self._transposition_counter[transposition_key] = self._transposition_counter.get(transposition_key, 0) + 1
        repetitions = self._transposition_counter[transposition_key] - 1

        side_to_move = 0 if self._board.turn else 1
        rule50_count = self._board.halfmove_clock

        if not side_to_move:
            castling_rights = self._board.castling_rights
            us_ooo = (castling_rights >> chess.A1) & 1
            us_oo = (castling_rights >> chess.H1) & 1
            them_ooo = (castling_rights >> chess.A8) & 1
            them_oo = (castling_rights >> chess.H8) & 1
        else:
            castling_rights = self._board.castling_rights
            us_ooo = (castling_rights >> chess.A8) & 1
            us_oo = (castling_rights >> chess.H8) & 1
            them_ooo = (castling_rights >> chess.A1) & 1
            them_oo = (castling_rights >> chess.H1) & 1

        plane_bytes = b"".join(self._get_plane_bytes())
        repetition = repetitions >= 1

        board_data = _Lc0BoardData(
            plane_bytes=plane_bytes,
            repetition=repetition,
            transposition_key=transposition_key,
            us_ooo=us_ooo,
            us_oo=us_oo,
            them_ooo=them_ooo,
            them_oo=them_oo,
            side_to_move=side_to_move,
            rule50_count=rule50_count,
        )

        self._history_stack.append(board_data)

    def to_input_array(self) -> np.ndarray:
        """
        Convert current board position to LC0's 112×8×8 input encoding.

        Returns:
            Array of shape (112, 8, 8) with dtype uint8.
            Planes 0-103: 8 positions × 13 planes (pieces + repetition)
            Planes 104-111: Metadata (castling, side to move, etc.)
        """
        planes_list: list[Any] = []
        current_data = self._history_stack[-1]
        planes_count = 0

        for data in self._history_stack[-1:-9:-1]:
            plane_bytes = data.plane_bytes

            if not current_data.side_to_move:
                planes: Any = np.unpackbits(np.frombuffer(plane_bytes, dtype=np.uint8))[::-1].reshape(12, 8, 8)[::-1]
            else:
                planes = (
                    np.unpackbits(np.frombuffer(plane_bytes, dtype=np.uint8))[::-1]
                    .reshape(12, 8, 8)[::-1]
                    .reshape(2, 6, 8, 8)[::-1, :, ::-1]
                    .reshape(12, 8, 8)
                )

            planes_list.append(planes)
            planes_list.append([_FLAT_PLANES[int(data.repetition)]])
            planes_count += 13

        empty_planes_count = 104 - planes_count
        if empty_planes_count > 0:
            empty_planes: list[Any] = [_FLAT_PLANES[0] for _ in range(empty_planes_count)]
            planes_list.append(empty_planes)

        metadata_planes: list[Any] = [
            _FLAT_PLANES[current_data.us_ooo],
            _FLAT_PLANES[current_data.us_oo],
            _FLAT_PLANES[current_data.them_ooo],
            _FLAT_PLANES[current_data.them_oo],
            _FLAT_PLANES[current_data.side_to_move],
            _FLAT_PLANES[current_data.rule50_count],
            _FLAT_PLANES[0],
            _FLAT_PLANES[1],
        ]
        planes_list.append(metadata_planes)

        return np.concatenate(planes_list)

    def __repr__(self) -> str:
        return f"Lc0BoardEncoder('{self.fen}')"


class ActivationExtractor:
    """
    Extract activations from LC0 models using PyTorch forward hooks.

    Example:
        >>> # model = LczeroModel.from_path("model.pt")  # doctest: +SKIP
        >>> # extractor = ActivationExtractor(model, ["block3/conv2/relu"])  # doctest: +SKIP
        >>> # board = Lc0BoardEncoder()  # doctest: +SKIP
        >>> # activations = extractor.extract(board)  # doctest: +SKIP
        >>> # activations["block3/conv2/relu"].shape  # doctest: +SKIP
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

    def extract(self, board: Lc0BoardEncoder) -> dict[str, torch.Tensor]:
        """
        Extract activations for a single board position.

        Args:
            board: Encoded chess position

        Returns:
            Dict mapping layer names to activation tensors
        """
        encoding = board.to_input_array()
        tensor = torch.from_numpy(encoding.astype(np.float32)).unsqueeze(0)  # type: ignore[call-arg]
        tensor = tensor.to(self.device)

        self.activations.clear()

        with torch.no_grad():
            module = self.model.module if hasattr(self.model, "module") else self.model
            _ = module(tensor)

        return self.activations.copy()

    def extract_batch(self, boards: list[Lc0BoardEncoder]) -> dict[str, torch.Tensor]:
        """
        Extract activations for multiple boards (batched).

        Args:
            boards: List of encoded chess positions

        Returns:
            Dict mapping layer names to batched activation tensors
        """
        encodings = np.stack([b.to_input_array() for b in boards])
        tensor = torch.from_numpy(encodings.astype(np.float32))  # type: ignore[call-arg]
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


def _cache_key(fen: str, model_path: str, layer_name: str) -> str:
    """Generate cache key for activation."""
    key_str = f"{fen}|{model_path}|{layer_name}"
    return hashlib.sha256(key_str.encode()).hexdigest()


def extract_features(
    fen: str,
    model_path: str | Path,
    layer_name: str,
    use_cache: bool = True,
) -> np.ndarray:
    """
    Extract flattened activation vector from a chess position.

    Args:
        fen: Chess position in FEN notation
        model_path: Path to LC0 model file
        layer_name: Layer to extract from (e.g., "block3/conv2/relu")
        use_cache: Whether to use cached activations

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
    model_path_str = str(model_path)
    cache_key = _cache_key(fen, model_path_str, layer_name)

    if use_cache and cache_key in _cache:
        return _cache[cache_key]

    model = LczeroModel.from_path(model_path_str)
    with ActivationExtractor(model, [layer_name]) as extractor:
        board = Lc0BoardEncoder(fen)
        activations = extractor.extract(board)

    activation = activations[layer_name].cpu().numpy()
    flattened = activation.reshape(-1)

    if use_cache:
        _cache[cache_key] = flattened

    return flattened


def extract_features_batch(
    fens: list[str],
    model_path: str | Path,
    layer_name: str,
    batch_size: int = 32,
    use_cache: bool = True,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Extract flattened activation vectors from multiple chess positions.

    Args:
        fens: List of chess positions in FEN notation
        model_path: Path to LC0 model file
        layer_name: Layer to extract from
        batch_size: Number of positions to process at once
        use_cache: Whether to use cached activations
        show_progress: Whether to print progress

    Returns:
        Array of shape (n_positions, n_features)

    Example:
        >>> fens = ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"] * 10  # doctest: +SKIP
        >>> features = extract_features_batch(fens, "models/maia-1500.pt", "block3/conv2/relu")  # doctest: +SKIP
        >>> features.shape  # doctest: +SKIP
        (10, 4096)
    """
    model_path_str = str(model_path)
    results: list[np.ndarray | None] = []
    cache_misses: list[str] = []
    cache_miss_indices: list[int] = []

    for idx, fen in enumerate(fens):
        cache_key = _cache_key(fen, model_path_str, layer_name)
        if use_cache and cache_key in _cache:
            results.append(_cache[cache_key])
        else:
            results.append(None)
            cache_misses.append(fen)
            cache_miss_indices.append(idx)

    if cache_misses:
        if show_progress:
            print(f"Extracting activations for {len(cache_misses)}/{len(fens)} positions...")

        model = LczeroModel.from_path(model_path_str)
        with ActivationExtractor(model, [layer_name]) as extractor:
            for i in range(0, len(cache_misses), batch_size):
                batch_fens = cache_misses[i : i + batch_size]
                batch_indices = cache_miss_indices[i : i + batch_size]
                boards = [Lc0BoardEncoder(fen) for fen in batch_fens]
                activations = extractor.extract_batch(boards)
                batch_activations: Any = activations[layer_name].cpu().numpy()

                for j, (fen, idx) in enumerate(zip(batch_fens, batch_indices, strict=True)):
                    flattened: np.ndarray = batch_activations[j].reshape(-1)
                    results[idx] = flattened

                    if use_cache:
                        cache_key = _cache_key(fen, model_path_str, layer_name)
                        _cache[cache_key] = flattened

                if show_progress:
                    print(f"  Processed {min(i + batch_size, len(cache_misses))}/{len(cache_misses)}")

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
