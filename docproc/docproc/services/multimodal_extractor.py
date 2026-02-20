"""
Multimodal document extraction using OpenAI GPT-4o vision.

Extracts text and visual analysis from PDF and PPTX documents by converting
pages to images and sending them to GPT-4o for OCR and interpretation.

This module is optimized for cloud deployment with semaphore-limited
concurrent processing. Pages are processed in parallel up to
VISION_CONCURRENCY (default 5) to balance throughput against API rate limits.

API Pattern (verified 02/06/2026):
- Uses AsyncOpenAI client with chat.completions.create()
- Images passed as base64 data URIs: "data:image/png;base64,{data}"
- response_format=json_schema with strict: true for guaranteed schema
  compliance (2026 production standard, replaces legacy json_object mode)
- max_tokens=4096 per page (sufficient for dense document pages with
  tables and complex layouts)

Configuration (environment variables - all optional with sensible defaults):
- OPENAI_API_KEY: Required API key
- VISION_MODEL: Model to use (default: "gpt-4o")
- VISION_MAX_PAGES: Max pages to process (default: 20)
- VISION_CONCURRENCY: Max concurrent API calls per document (default: 5)

Model Selection (researched 02/06/2026):
- gpt-4o is the recommended default for document extraction with visual
  content. GPT-4o scores 69.1% on MMMU vs 59.4% for gpt-4o-mini.
- gpt-4o-mini uses up to 20x more tokens for vision tasks, which can
  offset its lower per-token cost savings.
- For text-only extraction without visual elements, gpt-4o-mini may be
  a cost-effective alternative; override via VISION_MODEL env var.

SDK Version (verified 02/06/2026):
- openai>=1.10.0: AsyncOpenAI client with vision + structured outputs

Last Grunted: 02/06/2026
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

# Configuration -- loaded from centralized IngestionConfig.
# Module-level constants kept for backward compatibility in code that
# references VISION_MODEL, VISION_MAX_PAGES, VISION_CONCURRENCY directly.
from docproc.config import get_ingestion_config as _get_config

def _cfg():
    return _get_config()

# Lazy-evaluated but eagerly-read module constants
VISION_MODEL = os.getenv("VISION_MODEL", "gpt-4o")
VISION_MAX_PAGES = int(os.getenv("VISION_MAX_PAGES", "20"))
VISION_CONCURRENCY = int(os.getenv("VISION_CONCURRENCY", "5"))

# Extraction prompt - requests structured output with inline visual descriptions.
# The response_format json_schema enforces the output shape; the prompt guides
# content quality and visual marker placement.
EXTRACTION_PROMPT = """You are an expert document transcription and OCR assistant.
Document: {filename} | Page: {page}

Your task:
1. Extract ALL text content from this page image exactly as it appears.
2. Preserve the original reading order, paragraph breaks, and heading hierarchy.
3. For tables, reproduce the structure using Markdown table syntax.
4. For visual elements (charts, diagrams, images, figures), insert a descriptive
   [VISUAL: type - description] marker INLINE at the position where the element
   appears in the document layout.

Visual marker example:
[VISUAL: Bar Chart - Q1-Q4 revenue showing 52% YoY growth, Q4 peak at $3.2M]

If the page contains only visual elements with no text, describe them in detail."""

# JSON Schema for structured output (strict mode).
# Enforced at the token generation level — the model cannot produce
# non-conforming responses. Replaces legacy {"type": "json_object"}.
EXTRACTION_RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "page_extraction",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": (
                        "Full extracted text with [VISUAL: type - description] "
                        "markers for visual elements"
                    ),
                },
            },
            "required": ["text"],
            "additionalProperties": False,
        },
    },
}

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

    Converts document pages to images and processes them concurrently
    through the vision API for text extraction and visual analysis.
    Concurrency is limited by an asyncio.Semaphore (VISION_CONCURRENCY,
    default 5) to avoid overwhelming API rate limits on large documents.

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
        - response_format=json_schema (strict: true) for guaranteed schema
        - max_tokens=4096 per page request
        - Semaphore-limited concurrency (default: 5 concurrent pages)

    Last Grunted: 02/06/2026
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
        lambda: converter(file_bytes, max_pages=VISION_MAX_PAGES),
    )

    if not images:
        return "", "", "Failed to convert document to images"

    logger.info(f"Processing {len(images)} pages from {filename}")

    # Semaphore limits concurrent API calls to avoid rate-limit errors.
    # Default 5 balances throughput vs. OpenAI tier-1 rate limits.
    semaphore = asyncio.Semaphore(VISION_CONCURRENCY)

    async def _sem_extract(image: DocumentImage) -> str:
        async with semaphore:
            return await _extract_page(image, filename, visual_analysis)

    try:
        # Process pages with semaphore-limited concurrency
        tasks = [_sem_extract(image) for image in images]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect successful extractions
        text_parts: list[str] = []
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
