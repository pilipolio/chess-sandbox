"""Modal endpoints for concept extraction inference."""

import modal
from pydantic import BaseModel

from chess_sandbox.concept_extraction.model.inference import ConceptExtractor
from chess_sandbox.config import settings


class ConceptPrediction(BaseModel):
    """Individual concept prediction with confidence score."""

    name: str
    confidence: float


class ConceptExtractionResponse(BaseModel):
    """Response model for concept extraction endpoint."""

    fen: str
    threshold: float
    concepts: list[ConceptPrediction]


image = (
    modal.Image.debian_slim()
    # .env(
    #     {
    #         "HF_CACHE_DIR": "/root/.cache/huggingface",
    #         "HF_CONCEPT_EXTRACTOR_REPO_ID": settings.HF_CONCEPT_EXTRACTOR_REPO_ID,
    #     }
    # )
    .uv_sync(uv_project_dir="./", frozen=True)
    .uv_pip_install("fastapi[standard]")
    .add_local_python_source("chess_sandbox")
)

app = modal.App(name="chess-concept-extraction", image=image)


_extractor: ConceptExtractor | None = None


@app.function()  # type: ignore
@modal.fastapi_endpoint(method="GET")  # type: ignore
def extract_concepts(
    fen: str,
    threshold: float = 0.1,
) -> ConceptExtractionResponse:
    """
    Extract chess concepts from a position with confidence scores.

    Extracts chess concepts (pins, forks, sacrifices, etc.) from a position
    using trained neural probe classifiers on Leela Chess Zero activations.
    Returns only concepts above the specified probability threshold.

    Args:
        fen: Position in FEN notation (path parameter, required)
        threshold: Probability threshold for filtering concepts (query parameter, default=0.1)

    Returns:
        ConceptExtractionResponse with concepts above threshold and their confidence scores

    Raises:
        ValueError: Invalid FEN notation
        RuntimeError: Model loading or inference error

    Example:
        GET /position/{encoded_fen}/concepts?threshold=0.1
        Where {encoded_fen} is the URL-encoded FEN string
    """
    extractor = get_extractor()
    all_predictions = extractor.extract_concepts_with_confidence([fen])[0]

    return ConceptExtractionResponse(
        fen=fen,
        threshold=threshold,
        concepts=[
            ConceptPrediction(name=name, confidence=confidence)
            for name, confidence in all_predictions
            if confidence >= threshold
        ],
    )


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
        model_repo_id = settings.HF_CONCEPT_EXTRACTOR_REPO_ID
        model_revision = settings.HF_CONCEPT_EXTRACTOR_REVISION
        print(f"Initializing ConceptExtractor from {model_repo_id}@{model_revision}...")
        _extractor = ConceptExtractor.from_hf(probe_repo_id=model_repo_id, revision=model_revision)
        print("ConceptExtractor initialized successfully")
    return _extractor
