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
    .uv_sync(uv_project_dir="./", frozen=True)
    .uv_pip_install("fastapi[standard]")
    .add_local_python_source("chess_sandbox")
    .add_local_file(".env.modal", "/root/.env")
)

app = modal.App(name="chess-concept-extraction", image=image)


_extractor: ConceptExtractor | None = None


@app.function()  # type: ignore
@modal.fastapi_endpoint(method="GET")  # type: ignore
def extract_concepts(
    fen: str,
    threshold: float = 0.1,
) -> ConceptExtractionResponse:
    extractor = get_extractor()
    predictions = extractor.extract_concepts_with_confidence(fen)

    # When passing a single FEN, we get list[tuple[str, float]]
    # The type checker needs help here since the method signature is overloaded
    all_predictions: list[tuple[str, float]] = predictions  # type: ignore[assignment]

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
    global _extractor
    if _extractor is None:
        model_repo_id = settings.HF_CONCEPT_EXTRACTOR_REPO_ID
        model_revision = settings.HF_CONCEPT_EXTRACTOR_REVISION
        print(f"Initializing ConceptExtractor from {model_repo_id}@{model_revision}...")
        _extractor = ConceptExtractor.from_hf(probe_repo_id=model_repo_id, revision=model_revision)
        print("ConceptExtractor initialized successfully")
    return _extractor
