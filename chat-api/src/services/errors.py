"""
OpenAI-compatible error response utilities.

Provides standardized error responses matching the OpenAI API error format:
{
    "error": {
        "message": "Error description",
        "type": "error_type",
        "param": "parameter_name",
        "code": "error_code"
    }
}

Error Types (per OpenAI spec):
    - invalid_request_error: Malformed request or invalid parameters
    - authentication_error: Invalid API key or authentication failure
    - permission_error: Lacking permission for the requested operation
    - not_found_error: Requested resource doesn't exist
    - rate_limit_error: Rate limit exceeded
    - api_error: Server-side error
    - timeout_error: Request timed out

Last Grunted: 02/04/2026 05:30:00 PM UTC
"""
from typing import Optional
from fastapi.responses import JSONResponse
from pydantic import BaseModel


# ============================================================================
# Error Models
# ============================================================================

class OpenAIErrorDetail(BaseModel):
    """
    OpenAI API error detail structure.
    
    Attributes:
        message: Human-readable error description
        type: Error category
        param: Parameter that caused the error (nullable)
        code: Machine-readable error code (nullable)
        
    Last Grunted: 02/04/2026 05:30:00 PM UTC
    """
    message: str
    type: str
    param: Optional[str] = None
    code: Optional[str] = None


class OpenAIError(BaseModel):
    """
    OpenAI API error response wrapper.
    
    Attributes:
        error: The error detail object
        
    Last Grunted: 02/04/2026 05:30:00 PM UTC
    """
    error: OpenAIErrorDetail


# ============================================================================
# Error Response Factory
# ============================================================================

def create_error_response(
    message: str,
    error_type: str = "invalid_request_error",
    param: Optional[str] = None,
    code: Optional[str] = None,
    status_code: int = 400
) -> JSONResponse:
    """
    Create an OpenAI-style error response.
    
    Args:
        message: Human-readable error description
        error_type: Error category (see module docstring for types)
        param: The parameter that caused the error (if applicable)
        code: Machine-readable error code
        status_code: HTTP status code
        
    Returns:
        JSONResponse with OpenAI error format
        
    Example:
        >>> create_error_response(
        ...     message="Invalid model 'foo'",
        ...     error_type="invalid_request_error",
        ...     param="model",
        ...     code="model_not_found",
        ...     status_code=404
        ... )
        
    Last Grunted: 02/04/2026 05:30:00 PM UTC
    """
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "param": param,
                "code": code
            }
        }
    )


# ============================================================================
# Common Error Responses
# ============================================================================

def model_not_found_error(model: str) -> JSONResponse:
    """
    Create error response for unknown model.
    
    Args:
        model: The requested model name
        
    Returns:
        JSONResponse with 404 status
        
    Last Grunted: 02/04/2026 05:30:00 PM UTC
    """
    return create_error_response(
        message=f"The model '{model}' does not exist or you do not have access to it.",
        error_type="invalid_request_error",
        param="model",
        code="model_not_found",
        status_code=404
    )


def missing_parameter_error(param: str) -> JSONResponse:
    """
    Create error response for missing required parameter.
    
    Args:
        param: The missing parameter name
        
    Returns:
        JSONResponse with 400 status
        
    Last Grunted: 02/04/2026 05:30:00 PM UTC
    """
    return create_error_response(
        message=f"'{param}' is a required property",
        error_type="invalid_request_error",
        param=param,
        code="missing_required_parameter",
        status_code=400
    )


def invalid_parameter_error(
    param: str,
    message: str,
    code: Optional[str] = None
) -> JSONResponse:
    """
    Create error response for invalid parameter value.
    
    Args:
        param: The invalid parameter name
        message: Description of what's wrong
        code: Optional error code
        
    Returns:
        JSONResponse with 400 status
        
    Last Grunted: 02/04/2026 05:30:00 PM UTC
    """
    return create_error_response(
        message=message,
        error_type="invalid_request_error",
        param=param,
        code=code or "invalid_parameter",
        status_code=400
    )


def resource_not_found_error(
    resource_type: str,
    resource_id: str,
    param: Optional[str] = None
) -> JSONResponse:
    """
    Create error response for resource not found.
    
    Args:
        resource_type: Type of resource (thread, file, response, etc.)
        resource_id: The requested ID
        param: Parameter name if applicable
        
    Returns:
        JSONResponse with 404 status
        
    Last Grunted: 02/04/2026 05:30:00 PM UTC
    """
    return create_error_response(
        message=f"{resource_type.title()} '{resource_id}' not found",
        error_type="invalid_request_error",
        param=param or f"{resource_type}_id",
        code=f"{resource_type}_not_found",
        status_code=404
    )


def backend_error(status_code: int, detail: Optional[str] = None) -> JSONResponse:
    """
    Create error response for backend service failure.
    
    Args:
        status_code: Backend response status code
        detail: Optional error detail
        
    Returns:
        JSONResponse with backend status code
        
    Last Grunted: 02/04/2026 05:30:00 PM UTC
    """
    message = f"Backend error: {status_code}"
    if detail:
        message = f"{message} - {detail}"
    
    return create_error_response(
        message=message,
        error_type="api_error",
        code="backend_error",
        status_code=status_code
    )


def timeout_error(operation: str = "request") -> JSONResponse:
    """
    Create error response for request timeout.
    
    Args:
        operation: Description of the operation that timed out
        
    Returns:
        JSONResponse with 504 status
        
    Last Grunted: 02/04/2026 05:30:00 PM UTC
    """
    return create_error_response(
        message=f"The {operation} timed out",
        error_type="timeout_error",
        code="timeout",
        status_code=504
    )


def rate_limit_error(retry_after: Optional[int] = None) -> JSONResponse:
    """
    Create error response for rate limiting.
    
    Args:
        retry_after: Seconds until retry is allowed
        
    Returns:
        JSONResponse with 429 status
        
    Last Grunted: 02/04/2026 05:30:00 PM UTC
    """
    message = "Rate limit exceeded. Please retry after some time."
    if retry_after:
        message = f"Rate limit exceeded. Please retry after {retry_after} seconds."
    
    response = JSONResponse(
        status_code=429,
        content={
            "error": {
                "message": message,
                "type": "rate_limit_error",
                "param": None,
                "code": "rate_limit_exceeded"
            }
        }
    )
    
    if retry_after:
        response.headers["Retry-After"] = str(retry_after)
    
    return response


def internal_error(detail: Optional[str] = None) -> JSONResponse:
    """
    Create error response for internal server error.
    
    Args:
        detail: Optional error detail (be careful not to leak sensitive info)
        
    Returns:
        JSONResponse with 500 status
        
    Last Grunted: 02/04/2026 05:30:00 PM UTC
    """
    message = "An internal server error occurred"
    if detail:
        message = f"{message}: {detail}"
    
    return create_error_response(
        message=message,
        error_type="api_error",
        code="internal_error",
        status_code=500
    )


def context_length_exceeded_error(
    token_count: int,
    max_tokens: int,
    model: str
) -> JSONResponse:
    """
    Create error response for context length exceeded.
    
    Args:
        token_count: Actual token count
        max_tokens: Maximum allowed tokens
        model: Model being used
        
    Returns:
        JSONResponse with 400 status
        
    Last Grunted: 02/04/2026 05:30:00 PM UTC
    """
    return create_error_response(
        message=(
            f"This model's maximum context length is {max_tokens} tokens. "
            f"However, your messages resulted in {token_count} tokens. "
            f"Please reduce the length of the messages."
        ),
        error_type="invalid_request_error",
        param="messages",
        code="context_length_exceeded",
        status_code=400
    )
