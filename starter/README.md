# RAG Pipeline

This repository contains a FastAPI health endpoint and a Markdown loading and
chunking module. The current code does not connect chunks to an embedding model,
vector store, retriever, or answer generator.

## Current functionality

### API

The FastAPI application provides one endpoint:

- `GET /health` returns `{"status": "ok"}`.

### Markdown loading and chunking

The Markdown module can:

- Read `.md` files as UTF-8 text.
- Parse optional YAML front matter.
- Preserve unrecognized front matter fields in source metadata.
- Split Markdown into sections by headings.
- Avoid treating headings inside fenced code blocks as section headings.
- Pack whole paragraphs into chunks up to a configured character target.
- Attach document and section metadata to each chunk.

The Markdown loader is not connected to the FastAPI application.

## Project layout

- `api/` — FastAPI application.
- `config/` — Pydantic settings and logging configuration.
- `scripts/` — Markdown loader script and evaluation runner files.
- `src/rag/` — Markdown loader and other RAG module files.
- `src/schemas/` — API schemas and shared document and chunk models.
- `src/eval/` — Evaluation module files.
- `tests/` — Test module files.
- `data/` — Golden set JSONL files and data storage directories.

## Setup

Install project dependencies from the project root:

```bash
pip install -e .
```

Settings are read from `.env`. `config/settings.py` defines settings for OpenAI,
chunk sizing, vector store configuration, and the Markdown loader script.

The checked-in `.env.example` does not use all setting names defined in
`config/settings.py`. Review the settings module before using `.env.example`
as a configuration template.

## Run the API

From the project root, run:

```bash
uvicorn api.main:app --reload
```

The health endpoint is available at:

```text
http://127.0.0.1:8000/health
```

## Run the Markdown loader script

The script reads the directory configured by `MARKDOWN_TEST_DIRECTORY` in
`config/settings.py` or `.env`. It searches that directory and its
subdirectories for `.md` and `.txt` files.

For Markdown files, it prints the number of chunks and the first 100 characters
of each chunk. For `.txt` files, it prints the Markdown loader's file extension
error.

Run the script from the project root:

```bash
python scripts/test_markdown_loader.py
```

The Markdown loader imports PyYAML. PyYAML is not listed as a direct dependency
in `pyproject.toml`.

## Current limitations

- The API implements only a health endpoint. It does not provide document upload or query endpoints.
- The project does not have an implemented end-to-end RAG flow for ingestion, embedding, retrieval, or answer generation.
- The Markdown loader is not connected to the API.
- The Streamlit entry file does not currently implement a user interface.
- PDF and general text loaders are not implemented.
- The evaluation runner and test files do not currently contain executable evaluation or test logic.
- The older `src/rag/md_loader.py` corpus adapter refers to names that are not defined or imported in that file.
