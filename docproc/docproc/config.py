"""
Centralized configuration for the docproc ingestion pipeline.

All pipeline settings in one place, loaded from environment variables with
sensible defaults. Uses Pydantic Settings for validation and type coercion.

Usage:
    from docproc.config import get_ingestion_config
    
    config = get_ingestion_config()
    print(config.chunk_size)  # 300 (tokens)

Environment Variables (all optional -- defaults are production-ready):
    # Chunking
    CHUNK_SIZE=300              # Tokens per chunk (300 = optimal for RAG)
    CHUNK_OVERLAP=50            # Token overlap between chunks
    CHUNK_METHOD=token          # "token" or "recursive"

    # Vision / Multimodal
    VISION_MODEL=gpt-4o         # Model for PDF/PPTX vision extraction
    VISION_MAX_PAGES=20         # Max pages to process per document
    VISION_DPI=200              # Render DPI for page images
    VISION_MAX_EDGE=1568        # Max image dimension (pixels)
    VISION_MAX_TOKENS=4000      # Max tokens per page extraction
    PDF_ALWAYS_USE_VISION=false # Always use vision for PDFs (skip pdfplumber)

    # Embedding
    EMBEDDING_MODEL=text-embedding-3-small
    EMBEDDING_DIMENSIONS=       # Optional dimension override

    # LLM
    LLM_MODEL=gpt-4o-mini      # Model for summarization
    USE_LOCAL_MODELS=false      # Use Ollama instead of OpenAI
    OLLAMA_BASE_URL=http://localhost:11434

    # Storage
    DEFAULT_VECTOR_COLLECTION=document_summaries
    CHROMA_PERSIST_DIR=         # Empty = in-memory (dev)

    # Processing
    PDF_TEXT_MIN_CHARS=200      # Min chars for text-only PDF extraction
    SUMMARIZATION_THRESHOLD=100000  # Chars before map-reduce summarization
    SHARED_VOLUME_PATH=/shared-files
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class IngestionConfig(BaseSettings):
    """Centralized configuration for the docproc ingestion pipeline.
    
    All settings are loaded from environment variables with no prefix.
    Existing env var names (CHUNK_SIZE, VISION_MODEL, etc.) work unchanged.
    """

    model_config = SettingsConfigDict(
        env_prefix="",           # No prefix -- use exact env var names
        case_sensitive=False,    # CHUNK_SIZE and chunk_size both work
        extra="ignore",          # Don't fail on unknown env vars
    )

    # ─── Chunking ────────────────────────────────────────────────────────
    chunk_size: int = Field(
        default=300,
        description="Chunk size in tokens. 300 is optimal for RAG retrieval.",
        ge=50,
        le=4000,
    )
    chunk_overlap: int = Field(
        default=50,
        description="Overlap between chunks in tokens.",
        ge=0,
    )
    chunk_method: str = Field(
        default="token",
        description="Chunking method: 'token' (tiktoken-aware) or 'recursive' (char-based).",
    )

    @field_validator("chunk_method")
    @classmethod
    def validate_chunk_method(cls, v: str) -> str:
        allowed = {"token", "recursive"}
        if v not in allowed:
            raise ValueError(f"chunk_method must be one of {allowed}, got '{v}'")
        return v

    # ─── Vision / Multimodal ─────────────────────────────────────────────
    vision_model: str = Field(
        default="gpt-4o",
        description=(
            "OpenAI model for vision-based document extraction. "
            "gpt-4o gives best quality for tables/charts; "
            "gpt-4o-mini is cheaper but lower quality on complex layouts."
        ),
    )
    vision_max_pages: int = Field(
        default=20,
        description="Maximum pages/slides to process per document.",
        ge=1,
        le=100,
    )
    vision_dpi: int = Field(
        default=200,
        description=(
            "DPI for rendering PDF pages to images. "
            "200 is a good balance; use 300 for scanned documents."
        ),
        ge=72,
        le=600,
    )
    vision_max_edge: int = Field(
        default=1568,
        description=(
            "Maximum image dimension in pixels. "
            "1568 is the highest GPT-4o processes at full detail without tiling."
        ),
        ge=400,
        le=2048,
    )
    vision_max_tokens: int = Field(
        default=4000,
        description="Max output tokens per page during vision extraction.",
        ge=256,
        le=16000,
    )
    pdf_always_use_vision: bool = Field(
        default=False,
        description=(
            "When True, always use GPT-4o vision for PDF extraction "
            "instead of trying pdfplumber text extraction first. "
            "Produces higher quality results for PDFs with complex layouts, "
            "tables, or mixed text/image content, at higher cost."
        ),
    )

    # ─── Embedding ───────────────────────────────────────────────────────
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model. text-embedding-3-small (1536d) is cost-effective.",
    )
    embedding_dimensions: Optional[int] = Field(
        default=None,
        description="Optional dimension override for OpenAI embedding truncation.",
    )

    # ─── LLM ─────────────────────────────────────────────────────────────
    llm_model: str = Field(
        default="gpt-4o-mini",
        description="Model for summarization and other LLM tasks.",
    )
    use_local_models: bool = Field(
        default=False,
        description="Use Ollama for local model inference instead of OpenAI.",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama server URL when use_local_models=True.",
    )
    ollama_chat_model: str = Field(
        default="llama3.2:latest",
        description="Ollama chat model name.",
    )
    ollama_embed_model: str = Field(
        default="nomic-embed-text",
        description="Ollama embedding model name.",
    )

    # ─── Storage ─────────────────────────────────────────────────────────
    default_vector_collection: str = Field(
        default="document_summaries",
        description="Default ChromaDB collection name.",
    )
    chroma_persist_dir: Optional[str] = Field(
        default=None,
        description="ChromaDB persistence directory. None = in-memory (dev).",
    )

    # ─── Processing ──────────────────────────────────────────────────────
    pdf_text_min_chars: int = Field(
        default=200,
        description="Minimum characters from pdfplumber before falling back to vision.",
        ge=0,
    )
    summarization_threshold: int = Field(
        default=100_000,
        description="Character count above which map-reduce summarization is used.",
        ge=1000,
    )
    shared_volume_path: str = Field(
        default="/shared-files",
        description="Path to shared file volume for inter-service file access.",
    )


@lru_cache
def get_ingestion_config() -> IngestionConfig:
    """Get the singleton ingestion pipeline configuration.
    
    Configuration is loaded once from environment variables and cached.
    All settings use sensible production-ready defaults.
    
    Returns:
        IngestionConfig: Validated pipeline configuration.
    """
    return IngestionConfig()
