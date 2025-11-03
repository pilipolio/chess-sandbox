"""Modal endpoints for concept extraction inference."""

import modal

from chess_sandbox.concept_extraction.model.inference import (
    ConceptConfidenceResponse,
    ConceptExtractor,
    ConceptPredictionResponse,
)
from chess_sandbox.config import settings

image = (
    modal.Image.debian_slim()
    .env(
        {
            "HF_CACHE_DIR": "/root/.cache/huggingface",
            "HF_CONCEPT_PROBE_REPO": "pilipolio/chess-positions-extractor",
        }
    )
    .uv_sync(uv_project_dir="./", frozen=True)
    .add_local_python_source("chess_sandbox")
)

app = modal.App(name="chess-concept-extraction", image=image)


_extractor: ConceptExtractor | None = None


def get_extractor() -> ConceptExtractor:
    """
    Get or initialize the ConceptExtractor instance.

    Lazy loads the extractor on first call, downloading models from HuggingFace Hub.
    Subsequent calls return the cached instance. Uses configured default repo from settings.

    Returns:
        Initialized ConceptExtractor instance
    """
    global _extractor
    if _extractor is None:
        probe_repo_id = settings.HF_CONCEPT_PROBE_REPO
        print(f"Initializing ConceptExtractor from {probe_repo_id}...")
        _extractor = ConceptExtractor.from_hf(probe_repo_id=probe_repo_id)
        print("ConceptExtractor initialized successfully")
    return _extractor


@app.function()  # type: ignore
@modal.fastapi_endpoint(method="GET")  # type: ignore
def predict_concepts(
    fen: str,
    threshold: float = 0.5,
) -> ConceptPredictionResponse:
    """
    Predict chess concepts from a position.

    Extracts chess concepts (pins, forks, sacrifices, etc.) from a position
    using trained neural probe classifiers on Leela Chess Zero activations.

    Args:
        fen: Position in FEN notation (required)
        threshold: Probability threshold for positive prediction (default=0.5)

    Returns:
        ConceptPredictionResponse with detected concepts above threshold

    Raises:
        ValueError: Invalid FEN notation
        RuntimeError: Model loading or inference error

    Example:
        GET /predict-concepts?fen=rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR%20w%20KQkq%20-%200%201
    """
    extractor = get_extractor()
    concepts = extractor.extract_concepts(fen, threshold=threshold)
    return ConceptPredictionResponse.from_concepts(fen=fen, concepts=concepts, threshold=threshold)


@app.function()  # type: ignore
@modal.fastapi_endpoint(method="GET")  # type: ignore
def predict_concepts_with_confidence(
    fen: str,
) -> ConceptConfidenceResponse:
    """
    Predict chess concepts with confidence scores.

    Extracts all chess concepts with probability scores, allowing inspection
    of the model's confidence for each concept.

    Args:
        fen: Position in FEN notation (required)

    Returns:
        ConceptConfidenceResponse with all concepts and their probabilities

    Raises:
        ValueError: Invalid FEN notation
        RuntimeError: Model loading or inference error

    Example:
        GET /predict-concepts-with-confidence?fen=rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR%20w%20KQkq%20-%200%201
    """
    extractor = get_extractor()
    predictions = extractor.extract_concepts_with_confidence(fen)
    return ConceptConfidenceResponse.from_predictions(fen=fen, predictions=predictions)
