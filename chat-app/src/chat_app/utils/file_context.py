"""Utilities for building structured file context from DB records.

Provides dataclasses and functions for processing file records into
a structured format suitable for LLM prompts. Handles:
- Filtering processed files only
- Truncating summaries to prevent context overflow
- Deduplicating vector collections
- Sorting by processing time

Last Grunted: 02/03/2026 03:15:00 PM PST
"""
from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Iterable, List


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProcessedFileSummary:
    """Represents a processed file summary ready for LLM consumption."""

    id: str
    filename: str
    mime_type: str
    summary_text: str
    last_processed: int
    vector_collection: str


@dataclass
class FileContext:
    """Structured file context for constructing LLM prompts."""

    summaries: List[ProcessedFileSummary] = field(default_factory=list)
    vector_collections: List[str] = field(default_factory=list)


def build_file_context(
    files: Iterable, 
    logger_override: logging.Logger | None = None,
    max_summary_length: int = 2000,
    max_total_context_length: int = 10000
) -> FileContext:
    """
    Convert DB file records into processed summaries and collection metadata.
    
    Args:
        files: Iterable of file records from DB.
        logger_override: Optional logger to use.
        max_summary_length: Max chars per file summary.
        max_total_context_length: Max chars for total summary payload.
    """

    log = logger_override or logger
    summaries: list[ProcessedFileSummary] = []
    vector_collections: list[str] = []
    seen_collections: set[str] = set()
    
    total_summary_length = 0

    for file_record in files or []:
        status = (getattr(file_record, "status", "") or "").lower()
        summary_text = getattr(file_record, "summary_text", "") or ""

        if status != "processed":
            log.debug("Skipping file %s with status '%s'", getattr(file_record, "id", "unknown"), status)
            continue

        if not summary_text.strip():
            log.warning(
                "Processed file %s missing summary text; skipping",
                getattr(file_record, "id", "unknown"),
            )
            continue

        # Truncate individual summary if needed
        if len(summary_text) > max_summary_length:
            summary_text = summary_text[:max_summary_length] + "... [truncated]"

        last_processed = getattr(file_record, "last_processed", 0) or 0
        try:
            last_processed_int = int(last_processed)
        except (TypeError, ValueError):
            last_processed_int = 0

        summary = ProcessedFileSummary(
            id=getattr(file_record, "id", ""),
            filename=getattr(file_record, "file_name", ""),
            mime_type=getattr(file_record, "mime_type", ""),
            summary_text=summary_text,
            last_processed=last_processed_int,
            vector_collection=getattr(file_record, "vector_collection", "") or "",
        )
        summaries.append(summary)
        total_summary_length += len(summary_text)

        if summary.vector_collection and summary.vector_collection not in seen_collections:
            seen_collections.add(summary.vector_collection)
            vector_collections.append(summary.vector_collection)

    summaries.sort(key=lambda item: (item.last_processed, item.filename, item.id))
    
    # Check total length
    if total_summary_length > max_total_context_length:
        log.warning(
            "Total summary length (%d) exceeds limit (%d). Omitting summaries from context.",
            total_summary_length,
            max_total_context_length
        )
        # Re-create summaries with empty text
        # We keep the file metadata but remove the content to save space
        summaries = [
            ProcessedFileSummary(
                id=s.id,
                filename=s.filename,
                mime_type=s.mime_type,
                summary_text="", # Omitted
                last_processed=s.last_processed,
                vector_collection=s.vector_collection
            )
            for s in summaries
        ]

    if vector_collections:
        log.info("Extracted %d vector collection(s) from processed files", len(vector_collections))

    return FileContext(summaries=summaries, vector_collections=vector_collections)


__all__ = ["FileContext", "ProcessedFileSummary", "build_file_context"]

