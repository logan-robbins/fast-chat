"""
Document summarization service.

Modern approach: Use single LLM call for most documents since GPT-4o
supports 128K token context. Falls back to map-reduce only for
extremely long documents.

Summarization Strategy:
1. Single-call (default): Documents <= 100K chars (~25K tokens)
   - Faster, higher quality (model sees full context)
   - Uses temperature=0 for deterministic output
   
2. Map-reduce (fallback): Documents > 100K chars
   - Splits into 20K char chunks with 500 char overlap
   - Summarizes chunks in parallel (map phase)
   - Combines summaries into final summary (reduce phase)

Token Estimation:
- 1 token ≈ 4 characters (English text)
- 100K chars ≈ 25K tokens
- GPT-4o context: 128K tokens

SDK Version (verified 02/03/2026):
- langchain_openai>=0.0.5: ChatOpenAI with invoke() method

Last Grunted: 02/03/2026 02:45:00 PM PST
"""
import asyncio
import logging
import os
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from docproc.utils.llm_config import get_llm_model

logger = logging.getLogger(__name__)

# Threshold for single-call vs map-reduce (characters)
# ~100K chars = ~25K tokens, well within GPT-4o's 128K limit
SINGLE_CALL_THRESHOLD = int(os.getenv("SUMMARIZATION_THRESHOLD", "100000"))

# Map-reduce settings (only used for very long documents)
MAP_CHUNK_SIZE = 20000  # Large chunks for better context
MAP_CHUNK_OVERLAP = 500


async def summarize_text_async(
    title: str,
    text: str,
    model: str = "gpt-4o-mini",
) -> str:
    """
    Summarize text using the most efficient method for its length.
    
    Automatically selects between single-call and map-reduce based on
    document length. Single-call is preferred when possible as it
    produces higher quality summaries.
    
    Args:
        title (str): Document title for context in the prompt
        text (str): Full document text to summarize
        model (str): LLM model to use (default: "gpt-4o-mini")
    
    Returns:
        str: Summary text. Returns error message for empty documents.
    
    Threshold:
        - <= 100K chars: Single LLM call (handles 95%+ of documents)
        - > 100K chars: Map-reduce pattern
    
    Token Math:
        100K chars ≈ 25K tokens (at 4 chars/token)
        GPT-4o context = 128K tokens
        25K input + response overhead = safe single call
    
    Last Grunted: 02/03/2026 02:45:00 PM PST
    """
    if not text or not text.strip():
        return "Document contains no extractable text."
    
    text_length = len(text)
    
    if text_length <= SINGLE_CALL_THRESHOLD:
        # Fast path: Single call (handles 95%+ of documents)
        logger.info(f"Summarizing '{title}' with single call ({text_length} chars)")
        return await _summarize_single_call(title, text, model)
    else:
        # Fallback: Map-reduce for very long documents
        logger.info(f"Summarizing '{title}' with map-reduce ({text_length} chars)")
        return await _summarize_map_reduce(title, text, model)


async def _summarize_single_call(title: str, text: str, model: str) -> str:
    """
    Summarize document in a single LLM call.
    
    This is the preferred method - simpler and often higher quality
    since the model sees the full context.
    """
    llm = get_llm_model(model)
    
    prompt = f"""Summarize the following document titled "{title}".

Provide a comprehensive summary that captures:
- Main topics and key points
- Important details and findings
- Any conclusions or recommendations

Document:
{text}

Summary:"""
    
    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(
        None,
        lambda: llm.invoke(prompt)
    )
    
    return response.content.strip()


async def _summarize_map_reduce(title: str, text: str, model: str) -> str:
    """
    Summarize very long document using map-reduce pattern.
    
    1. Split into large chunks
    2. Summarize each chunk (map)
    3. Combine summaries into final summary (reduce)
    
    Only used for documents > 100K characters.
    """
    llm = get_llm_model(model)
    loop = asyncio.get_running_loop()
    
    # Split into large chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAP_CHUNK_SIZE,
        chunk_overlap=MAP_CHUNK_OVERLAP,
    )
    chunks = splitter.split_text(text)
    logger.info(f"Split into {len(chunks)} chunks for map-reduce")
    
    # Map: Summarize each chunk in parallel
    async def summarize_chunk(chunk: str, idx: int) -> str:
        prompt = f"Summarize this section (part {idx + 1} of {len(chunks)}):\n\n{chunk}\n\nSummary:"
        response = await loop.run_in_executor(
            None,
            lambda: llm.invoke(prompt)
        )
        return response.content.strip()
    
    tasks = [summarize_chunk(chunk, i) for i, chunk in enumerate(chunks)]
    chunk_summaries = await asyncio.gather(*tasks)
    
    # Reduce: Combine summaries
    combined = "\n\n".join(
        f"Section {i + 1}:\n{summary}"
        for i, summary in enumerate(chunk_summaries)
        if summary and "[Error" not in summary
    )
    
    reduce_prompt = f"""The following are summaries of sections from a document titled "{title}":

{combined}

Please provide a single, unified summary that captures all key points:"""
    
    final_response = await loop.run_in_executor(
        None,
        lambda: llm.invoke(reduce_prompt)
    )
    
    return final_response.content.strip()


async def summarize_single_file(
    filename: str,
    text: str,
    visual_info: str = "",
    model: str = "gpt-4o-mini",
) -> dict:
    """
    Summarize a single file's content.
    
    Convenience wrapper around summarize_text_async with error handling
    and structured response format.
    
    Args:
        filename (str): Name of the file (used as title in prompt)
        text (str): Extracted text content to summarize
        visual_info (str): Visual analysis (deprecated - now inline in text,
            kept for API backwards compatibility)
        model (str): LLM model to use (default: "gpt-4o-mini")
    
    Returns:
        dict: Result dictionary with keys:
            - filename (str): Original filename
            - status (str): "success" or "error"
            - summary (str | None): Generated summary or None on error
            - visual_analysis (str): Pass-through of visual_info parameter
            - error (str | None): Error message or None on success
    
    Note:
        Empty or whitespace-only text returns error status immediately
        without calling the LLM.
    
    Last Grunted: 02/03/2026 02:45:00 PM PST
    """
    try:
        if not text.strip():
            return {
                "filename": filename,
                "status": "error",
                "summary": None,
                "visual_analysis": "",
                "error": "No extractable text in document.",
            }
        
        summary = await summarize_text_async(filename, text, model)
        
        if not summary:
            raise ValueError("Empty summary generated")
        
        logger.info(f"Summarized {filename}: {len(summary)} chars")
        
        return {
            "filename": filename,
            "status": "success",
            "summary": summary,
            "visual_analysis": visual_info,  # Pass through (now inline in text)
            "error": None,
        }
        
    except Exception as exc:
        logger.exception(f"Summarization failed for {filename}")
        return {
            "filename": filename,
            "status": "error",
            "summary": None,
            "visual_analysis": visual_info,
            "error": str(exc),
        }
