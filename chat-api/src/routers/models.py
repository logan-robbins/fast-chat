"""
OpenAI-compatible Models API router.

Implements the /v1/models endpoints per OpenAI API specification:
    - GET /v1/models - List all available models
    - GET /v1/models/{model} - Retrieve specific model information

Uses the shared model registry (``src.services.model_registry``) as
the single source of truth.  Adding a model to the registry is all
that is needed for it to appear in these endpoints.

OpenAI Models Object Schema:
{
    "id": "gpt-4o",
    "object": "model",
    "created": 1686935002,
    "owned_by": "openai"
}

Reference: https://platform.openai.com/docs/api-reference/models

Last Grunted: 02/05/2026 12:00:00 PM UTC
"""
import structlog
from typing import List

from fastapi import APIRouter
from pydantic import BaseModel

from src.services.model_registry import get_model, get_model_objects
from src.services.errors import model_not_found_error

logger = structlog.get_logger(__name__)

router = APIRouter()


# ============================================================================
# Pydantic Models (OpenAI-compatible)
# ============================================================================

class ModelObject(BaseModel):
    """OpenAI Model object representation.

    Attributes:
        id: The model identifier (e.g. ``"gpt-4o"``).
        object: Always ``"model"``.
        created: Unix timestamp of model creation.
        owned_by: Organisation that owns the model.

    Last Grunted: 02/05/2026 12:00:00 PM UTC
    """

    id: str
    object: str = "model"
    created: int
    owned_by: str


class ModelListResponse(BaseModel):
    """OpenAI Models list response.

    Attributes:
        object: Always ``"list"``.
        data: Array of :class:`ModelObject` instances.

    Last Grunted: 02/05/2026 12:00:00 PM UTC
    """

    object: str = "list"
    data: List[ModelObject]


# ============================================================================
# Endpoints
# ============================================================================

@router.get("/v1/models", response_model=ModelListResponse)
async def list_models():
    """List all available models.

    Returns every model registered in the shared model registry.

    Returns:
        ModelListResponse containing all models.

    Example Response::

        {
            "object": "list",
            "data": [
                {"id": "gpt-4o", "object": "model", "created": 1715367049, "owned_by": "openai"},
                ...
            ]
        }

    Last Grunted: 02/05/2026 12:00:00 PM UTC
    """
    return ModelListResponse(
        object="list",
        data=[ModelObject(**m) for m in get_model_objects()],
    )


@router.get("/v1/models/{model_id}", response_model=ModelObject)
async def retrieve_model(model_id: str):
    """Retrieve a specific model by ID.

    Args:
        model_id: The model identifier (e.g. ``"gpt-4o"``).

    Returns:
        ModelObject matching the specified ID.

    Raises:
        JSONResponse: 404 if model not found (OpenAI error format).

    Last Grunted: 02/05/2026 12:00:00 PM UTC
    """
    spec = get_model(model_id)
    if not spec:
        return model_not_found_error(model_id)

    return ModelObject(
        id=spec.id,
        object="model",
        created=spec.created,
        owned_by=spec.owned_by,
    )
