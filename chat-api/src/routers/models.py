"""
OpenAI-compatible Models API router.

Implements the /v1/models endpoints per OpenAI API specification:
    - GET /v1/models - List all available models
    - GET /v1/models/{model} - Retrieve specific model information

OpenAI Models Object Schema:
{
    "id": "gpt-4o",
    "object": "model",
    "created": 1686935002,
    "owned_by": "openai"
}

Reference: https://platform.openai.com/docs/api-reference/models

Last Grunted: 02/03/2026 10:30:00 AM UTC
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import time

router = APIRouter()


class ModelObject(BaseModel):
    """
    OpenAI Model object representation.
    
    Attributes:
        id: The model identifier (e.g., "gpt-4o", "gpt-4o-mini")
        object: Always "model"
        created: Unix timestamp of model creation
        owned_by: Organization that owns the model
        
    Last Grunted: 02/03/2026 10:30:00 AM UTC
    """
    id: str
    object: str = "model"
    created: int
    owned_by: str


class ModelListResponse(BaseModel):
    """
    OpenAI Models list response.
    
    Attributes:
        object: Always "list"
        data: Array of ModelObject instances
        
    Last Grunted: 02/03/2026 10:30:00 AM UTC
    """
    object: str = "list"
    data: List[ModelObject]


# Available models - these represent the models this BFF proxies to
AVAILABLE_MODELS = [
    ModelObject(
        id="gpt-4o",
        object="model",
        created=1715367049,
        owned_by="openai"
    ),
    ModelObject(
        id="gpt-4o-mini",
        object="model",
        created=1721172741,
        owned_by="openai"
    ),
    ModelObject(
        id="gpt-4-turbo",
        object="model",
        created=1712361441,
        owned_by="openai"
    ),
    ModelObject(
        id="gpt-3.5-turbo",
        object="model",
        created=1677610602,
        owned_by="openai"
    ),
]


@router.get("/v1/models", response_model=ModelListResponse)
async def list_models():
    """
    List all available models.
    
    Returns a list of model objects describing available models for use
    with the chat completions endpoint.
    
    Returns:
        ModelListResponse: Object containing list of available models
        
    Example Response:
        {
            "object": "list",
            "data": [
                {"id": "gpt-4o", "object": "model", "created": 1715367049, "owned_by": "openai"},
                ...
            ]
        }
        
    Last Grunted: 02/03/2026 10:30:00 AM UTC
    """
    return ModelListResponse(
        object="list",
        data=AVAILABLE_MODELS
    )


@router.get("/v1/models/{model_id}", response_model=ModelObject)
async def retrieve_model(model_id: str):
    """
    Retrieve a specific model by ID.
    
    Returns basic information about the model such as owner and availability.
    
    Args:
        model_id: The model identifier (e.g., "gpt-4o", "gpt-4o-mini")
        
    Returns:
        ModelObject: The model object matching the specified ID
        
    Raises:
        HTTPException: 404 if model not found
        
    Example Response:
        {
            "id": "gpt-4o",
            "object": "model",
            "created": 1715367049,
            "owned_by": "openai"
        }
        
    Last Grunted: 02/03/2026 10:30:00 AM UTC
    """
    for model in AVAILABLE_MODELS:
        if model.id == model_id:
            return model
    
    raise HTTPException(
        status_code=404,
        detail={
            "error": {
                "message": f"The model '{model_id}' does not exist",
                "type": "invalid_request_error",
                "param": "model",
                "code": "model_not_found"
            }
        }
    )
