"""
Multimodal document extraction using OpenAI GPT-4o vision.

Extracts text and visual analysis from PDF and PPTX documents by converting
pages to images and sending them to GPT-4o for OCR and interpretation.

This module is optimized for cloud deployment with parallel processing.
For large documents, pages are processed concurrently for faster throughput.

API Pattern (verified 02/05/2026):
- Uses AsyncOpenAI client with chat.completions.create()
- Images passed as base64 data URIs: "data:image/png;base64,{data}"
- response_format={"type": "json_object"} for structured output
- max_tokens=2000 per page (sufficient for most document pages)

Configuration (environment variables - all optional with sensible defaults):
- OPENAI_API_KEY: Required API key
- VISION_MODEL: Model to use (default: "gpt-4o-mini")
- VISION_MAX_PAGES: Max pages to process (default: 20)

SDK Version (verified 02/05/2026):
- openai>=1.10.0: AsyncOpenAI client with vision support

Last Grunted: 02/05/2026
"""
from __future__ import annotations

import asyncio
import json
import logging
import os

from openai import AsyncOpenAI

from docproc.utils.image_utils import (
    DocumentImage,
    pdf_bytes_to_base64_images,
    pptx_bytes_to_base64_images,
)

logger = logging.getLogger(__name__)

# Simple configuration with sensible defaults
VISION_MODEL = os.getenv("VISION_MODEL", "gpt-4o-mini")
VISION_MAX_PAGES = int(os.getenv("VISION_MAX_PAGES", "20"))

# Extraction prompt - requests JSON output with inline visual descriptions
EXTRACTION_PROMPT = """You are an expert document transcription assistant.
Document: {filename} | Page: {page}

Extract ALL text content from this page image. Also describe any visual elements (charts, tables, diagrams, images) INLINE where they appear.

Respond with JSON:
{{"text": "full extracted text with [VISUAL: type - description] markers for visual elements"}}

Example visual marker: [VISUAL: Bar Chart - Q1-Q4 revenue showing 52% YoY growth, Q4 peak at $3.2M]

If the page has no text, describe any visuals in the text field."""

# Cached client
_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    """Get or create cached AsyncOpenAI client."""
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable required")
        _client = AsyncOpenAI(api_key=api_key, timeout=120.0)
    return _client


async def extract_document_with_vision(
    file_bytes: bytes,
    file_extension: str,
    filename: str,
    visual_analysis: bool = True,
) -> tuple[str, str, str | None]:
    """
    Extract text from document using GPT-4o vision.

    Converts document pages to images and processes them in parallel
    through the vision API for text extraction and visual analysis.
    Pages are processed using asyncio.gather() for concurrent API calls.

    Args:
        file_bytes (bytes): Raw document bytes (PDF or PPTX)
        file_extension (str): File extension including dot (e.g., ".pdf", ".pptx")
        filename (str): Original filename for context in prompts
        visual_analysis (bool): If True, include [VISUAL: ...] markers for
            charts, tables, diagrams, and images inline in extracted text.

    Returns:
        tuple[str, str, str | None]: (extracted_text, visual_info, error)
            - extracted_text: Combined text from all pages with [Page N] headers
            - visual_info: Empty string (visuals now inline in text)
            - error: Error message if failed, else None

    API Details:
        - Uses chat.completions.create() with vision model
        - Images passed as base64 data URIs in content array
        - response_format={"type": "json_object"} ensures valid JSON
        - max_tokens=2000 per page request

    Last Grunted: 02/05/2026
    """
    # Select converter based on file type
    ext = file_extension.lower()
    if ext == ".pdf":
        converter = pdf_bytes_to_base64_images
    elif ext == ".pptx":
        converter = pptx_bytes_to_base64_images
    else:
        return "", "", f"Unsupported file type for vision: {ext}"
    
    # Convert document to images (CPU-bound, run in executor)
    loop = asyncio.get_running_loop()
    images: list[DocumentImage] = await loop.run_in_executor(
        None,
        lambda: converter(file_bytes, max_pages=VISION_MAX_PAGES)
    )
    
    if not images:
        return "", "", "Failed to convert document to images"
    
    logger.info(f"Processing {len(images)} pages from {filename}")
    
    try:
        # Process all pages in parallel
        tasks = [
            _extract_page(image, filename, visual_analysis)
            for image in images
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect successful extractions
        text_parts = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"Page {i+1} failed: {result}")
                text_parts.append(f"[Page {i+1}]\n(extraction failed)")
            else:
                page_text = result
                text_parts.append(f"[Page {images[i].index}]\n{page_text}")
        
        combined_text = "\n\n".join(text_parts)
        logger.info(f"Extracted {len(combined_text)} chars from {filename}")
        
        # Visual info is now embedded inline in text
        return combined_text, "", None
        
    except Exception as exc:
        logger.exception(f"Vision extraction failed: {exc}")
        return "", "", str(exc)


async def _extract_page(
    image: DocumentImage,
    filename: str,
    include_visuals: bool,
) -> str:
    """
    Extract text from a single page using GPT-4o vision.
    
    Uses structured JSON output for reliable parsing. The prompt requests
    JSON with a "text" field containing all extracted content.
    
    Args:
        image (DocumentImage): DocumentImage with page index and base64 data
        filename (str): Document filename for prompt context
        include_visuals (bool): Whether to include [VISUAL: ...] markers
    
    Returns:
        str: Extracted text content from the page. Visual elements are
            embedded as [VISUAL: type - description] markers when enabled.
    
    Raises:
        Exception: If API call fails (timeout, rate limit, etc.)
    
    API Request Structure:
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_uri}}
            ]
        }]
    
    Last Grunted: 02/05/2026
    """
    client = _get_client()
    prompt = EXTRACTION_PROMPT.format(filename=filename, page=image.index)
    
    if not include_visuals:
        prompt = prompt.replace(
            "Also describe any visual elements",
            "Focus only on text content, skip visual elements"
        )
    
    image_url = f"data:image/png;base64,{image.base64_data}"
    
    response = await client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        max_tokens=2000,
        response_format={"type": "json_object"},  # Guaranteed valid JSON
    )
    
    content = response.choices[0].message.content or "{}"
    
    try:
        data = json.loads(content)
        return data.get("text", "")
    except json.JSONDecodeError:
        # Shouldn't happen with response_format, but fallback just in case
        logger.warning("JSON parse failed despite response_format")
        return content
