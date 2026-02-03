# docproc - Document Processing Library

A Python library for document extraction, summarization, and vector storage.

## Features

- **Smart file type routing**: PDF, PPTX, DOCX, XLSX, CSV, TXT
- **Vision-based extraction**: GPT-4o for scanned PDFs and presentations
- **Text extraction fallback**: Fast native extraction for text-based documents
- **Summarization**: Single-call for most docs, map-reduce for very long ones
- **Vector storage**: ChromaDB (in-memory or persistent)

## Installation

```bash
pip install -e ./docproc
```

## Quick Start

```python
from docproc import (
    extract_text_from_file,
    summarize_single_file,
    store_document_with_chunks,
    ensure_vector_store_initialized,
)

# Initialize vector store
await ensure_vector_store_initialized()

# Process a file
filename, text, error, visual_info = await extract_text_from_file(file)

# Summarize
result = await summarize_single_file(filename, text)

# Store with embeddings
doc_id, hash, chunks, stored = await store_document_with_chunks(
    filename=filename,
    file_content=content,
    full_text=text,
    summary=result["summary"],
    visual_analysis="",
    model="gpt-4o-mini",
)
```

## Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | - | OpenAI API key |
| `CHROMA_PERSIST_DIR` | No | (in-memory) | Path to persist ChromaDB |
| `LLM_MODEL` | No | `gpt-4o-mini` | Chat model |
| `EMBEDDING_MODEL` | No | `text-embedding-3-small` | Embedding model |

## File Type Support

| Type | Method | Vision API? |
|------|--------|-------------|
| `.txt`, `.md`, `.json` | Direct read | No |
| `.csv`, `.tsv` | Pandas | No |
| `.xlsx`, `.xls` | Pandas + openpyxl | No |
| `.docx` | python-docx | No |
| `.pdf` (text) | pdfplumber | No |
| `.pdf` (scanned) | GPT-4o vision | Yes |
| `.pptx` | GPT-4o vision | Yes |

## Local Development with Ollama

```bash
export USE_LOCAL_MODELS=true
export OLLAMA_BASE_URL=http://localhost:11434
```
