"""Perplexity API client utilities for web search functionality.

Provides the call_perplexity_api function for making requests to Perplexity's
chat completions endpoint. The API returns AI-synthesized answers with
source citations from web search results.

Supported Perplexity models (as of January 2026):
- sonar: Fast, lightweight search model with citations
- sonar-pro: Advanced model for complex queries, more citations
- sonar-reasoning: Enhanced reasoning with search (default)

Configuration via environment variables:
- PERPLEXITY_API_KEY: Required API key
- PERPLEXITY_API_ENDPOINT: API URL (default: https://api.perplexity.ai/chat/completions)
- PERPLEXITY_MODEL: Model name (default: sonar)
- PERPLEXITY_MAX_TOKENS: Max response tokens (default: 2000)
- PERPLEXITY_TEMPERATURE: Response temperature 0.0-1.0 (default: 0.2)
- PERPLEXITY_TIMEOUT: Request timeout in seconds (default: 30)

Last Grunted: 02/03/2026 03:45:00 PM PST
"""
import os
import logging
from typing import List, Dict, Any, TypedDict, Optional
import requests
import json

logger = logging.getLogger(__name__)


class SearchResult(TypedDict):
    """Structure for a single search result from Perplexity API."""
    title: str
    url: str
    date: Optional[str]


class PerplexityResponse(TypedDict):
    """Structure for Perplexity API response with content and citations."""
    content: str
    search_results: List[SearchResult]


def call_perplexity_api(messages: List[Dict[str, str]]) -> PerplexityResponse:
    """Call the Perplexity API chat completions endpoint.

    Makes a synchronous HTTP request to Perplexity's API with the provided
    messages. The API performs web search and returns an AI-synthesized
    response with source citations.

    Args:
        messages: List of message dictionaries with 'role' and 'content'.
            Typically a single user message with the search query.
            Example: [{"role": "user", "content": "What is Python?"}]

    Returns:
        PerplexityResponse: TypedDict with:
            - content (str): The AI-generated response text
            - search_results (List[SearchResult]): List of source citations
              with 'title', 'url', and optional 'date' fields

    Raises:
        ValueError: If PERPLEXITY_API_KEY environment variable is not set

    Note:
        Errors are handled gracefully - API errors, timeouts, and connection
        failures return a PerplexityResponse with an error message in content
        and an empty search_results list.

    Example:
        >>> response = call_perplexity_api([{"role": "user", "content": "Python history"}])
        >>> print(response['content'])
        "Python was created by Guido van Rossum..."
        >>> print(response['search_results'][0]['url'])
        "https://python.org/about"

    Last Grunted: 02/03/2026 03:45:00 PM PST
    """
    api_key = os.environ.get("PERPLEXITY_API_KEY")
    if not api_key:
        logger.error("PERPLEXITY_API_KEY environment variable not set")
        raise ValueError(
            "PERPLEXITY_API_KEY environment variable not set. "
            "Please add PERPLEXITY_API_KEY=pplx-your-key-here to your .env file"
        )

    url = os.getenv("PERPLEXITY_API_ENDPOINT", "https://api.perplexity.ai/chat/completions")

    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }

    data = {
        "model": os.getenv("PERPLEXITY_MODEL", "sonar"),
        "messages": messages,
        "max_tokens": int(os.getenv("PERPLEXITY_MAX_TOKENS", "2000")),
        "temperature": float(os.getenv("PERPLEXITY_TEMPERATURE", "0.2"))
    }

    logger.debug(
        "Preparing Perplexity API request",
        extra={
            "url": url,
            "model": data["model"],
            "message_count": len(messages),
            "message_preview": messages[0].get("content", "")[:200] if messages else None
        }
    )

    try:
        logger.debug("Sending request to Perplexity API")
        response = requests.post(url, json=data, headers=headers, timeout=int(os.getenv("PERPLEXITY_TIMEOUT", "30")))
        
        logger.debug(
            "Received response from Perplexity API",
            extra={
                "status_code": response.status_code,
                "response_size": len(response.content) if response.content else 0
            }
        )
        
        # Check for HTTP errors and extract error message from response
        if not response.ok:
            try:
                error_data = response.json()
                error_message = error_data.get('message', f'HTTP {response.status_code}: {response.reason}')
                error_type = error_data.get('type', 'API Error')
                error_code = error_data.get('code', response.status_code)
                
                logger.error(
                    "Perplexity API returned error",
                    extra={
                        "status_code": response.status_code,
                        "error_type": error_type,
                        "error_code": error_code,
                        "error_message": error_message,
                        "error_data": error_data
                    }
                )
                
                return PerplexityResponse(
                    content=f"Perplexity API error ({error_type}): {error_message}",
                    search_results=[]
                )
            except (ValueError, KeyError) as json_error:
                # If response is not JSON, use status code and reason
                logger.error(
                    "Perplexity API returned non-JSON error response",
                    extra={
                        "status_code": response.status_code,
                        "reason": response.reason,
                        "response_preview": response.text[:500] if response.text else None,
                        "json_error": str(json_error)
                    }
                )
                return PerplexityResponse(
                    content=f"Perplexity API error: HTTP {response.status_code} - {response.reason}",
                    search_results=[]
                )
        
        result = response.json()
        content = result['choices'][0]['message']['content']
        search_results = result.get('search_results', [])
        
        logger.debug(
            "Successfully parsed Perplexity API response",
            extra={
                "content_length": len(content),
                "search_results_count": len(search_results)
            }
        )

        return PerplexityResponse(
            content=content,
            search_results=search_results
        )

    except requests.exceptions.Timeout as e:
        logger.error(
            "Perplexity API request timed out",
            extra={
                "timeout": int(os.getenv("PERPLEXITY_TIMEOUT", "30")),
                "error": str(e)
            }
        )
        return PerplexityResponse(
            content=f"Web search timed out after {int(os.getenv('PERPLEXITY_TIMEOUT', '30'))} seconds. Please try again.",
            search_results=[]
        )
    except requests.exceptions.ConnectionError as e:
        logger.error(
            "Perplexity API connection error",
            extra={
                "error": str(e),
                "url": url
            }
        )
        return PerplexityResponse(
            content=f"Web search failed: Connection error. Please check your internet connection.",
            search_results=[]
        )
    except requests.exceptions.RequestException as e:
        logger.error(
            "Perplexity API request exception",
            extra={
                "error_type": type(e).__name__,
                "error": str(e),
                "status_code": getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None
            },
            exc_info=True
        )
        return PerplexityResponse(
            content=f"Web search failed: {type(e).__name__}: {str(e)}. Please check your internet connection and API key.",
            search_results=[]
        )
    except KeyError as e:
        logger.error(
            "Unexpected response format from Perplexity API",
            extra={
                "missing_key": str(e),
                "response_keys": list(result.keys()) if 'result' in locals() else None
            },
            exc_info=True
        )
        return PerplexityResponse(
            content=f"Unexpected response format from Perplexity API: Missing key {str(e)}",
            search_results=[]
        )
    except Exception as e:
        logger.error(
            "Unexpected error calling Perplexity API",
            extra={
                "error_type": type(e).__name__,
                "error": str(e)
            },
            exc_info=True
        )
        return PerplexityResponse(
            content=f"Error calling Perplexity API: {type(e).__name__}: {str(e)}",
            search_results=[]
        )
