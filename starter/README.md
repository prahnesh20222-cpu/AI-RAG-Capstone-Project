# RAG Knowledge Engineering Platform

This project aims to build a reliable knowledge foundation for Retrieval-Augmented Generation (RAG): convert information sources into traceable, structured knowledge that applications can search and use as evidence. The longer-term direction includes knowledge-aware AI agents.

The repository is an early-stage implementation. The goal and future directions below describe intended capabilities; the current status section distinguishes those from what the code currently does.

## Project goal

Documents and other information sources often contain useful structure and context that can be lost when they are reduced to plain text and vectors. This project aims to preserve source meaning, structure, metadata, and provenance through processing so retrieved evidence can be inspected and traced back to its origin.

The intended processing flow is:

```text
Source documents → structure extraction → semantic enrichment → canonical Markdown
                 → chunking and metadata → embeddings → vector storage
                 → retrieval → grounded RAG responses
```

Extraction should preserve source content and structure. Semantic enrichment may add topics, concepts, entities, decisions, or other metadata, but must not replace original evidence. Model-generated fields should be validated, and uncertain results should be reviewable.

## Current implementation

### API

The FastAPI application currently exposes one endpoint:

- `GET /health` returns `{"status": "ok"}`.

There are no implemented document-upload, query, or metrics endpoints.

### Markdown processing

`src/rag/markdown_loader.py` implements a Markdown-specific loading and chunking path. It can:

- Read `.md` files as UTF-8.
- Parse optional YAML front matter into shared document metadata, retaining unrecognized fields as source metadata.
- Split content into sections by Markdown headings while ignoring headings inside fenced code blocks.
- Pack whole paragraphs within each section up to a configured character target.
- Attach document, section, source, and count metadata to chunks.
- Recursively load Markdown files from a corpus directory through `load_markdown_chunks`.

The chunk target is measured in characters, not tokens. This processing path is not connected to the FastAPI application.

The script `scripts/test_markdown_loader.py` scans the configured corpus directory for `.md` and `.txt` files. It processes Markdown files and prints chunk previews; `.txt` files are identified but rejected by the Markdown-only loader.

### Other code in the repository

The repository includes shared document and API schemas, configuration and logging modules, evaluation-related Python modules and scripts, and placeholder modules describing planned chunking, embedding, retrieval, generation, vector storage, and evaluation functionality. The presence of those files or declared dependencies does not mean those capabilities are implemented as a working end-to-end pipeline. The evaluation scripts and Streamlit entry file do not currently provide executable evaluation workflows or a user interface.

## Technical details and setup

The project uses Python, FastAPI, Pydantic Settings, and a `src/` package layout. Install the dependencies declared in `pyproject.toml` from the project root:

```bash
pip install -e .
```

Configuration is loaded from `.env` by `config/settings.py`. Defined settings include OpenAI model and API fields, character-based chunk sizing (`TARGET_CHUNK_SIZE`, default `500`), overlap (`CHUNK_OVERLAP`, default `50`), corpus directory, vector-store and Qdrant fields, and `TOP_K` (default `5`). These settings do not establish that the corresponding model or vector-store functionality is implemented. Review `config/settings.py` and `.env.example` before configuring the project; the example file does not currently match all setting names.

The Markdown loader imports PyYAML, but PyYAML is not listed as a direct dependency in `pyproject.toml`. Install it separately if it is not already available in the environment.

### Run the API

From the project root:

```bash
uvicorn api.main:app --reload
```

The health endpoint is available at `http://127.0.0.1:8000/health`.

### Run the Markdown loader script

Set `CORPUS_DIRECTORY` in `config/settings.py` or `.env`, then run from the project root:

```bash
python scripts/test_markdown_loader.py
```

The script searches the configured directory and subdirectories for `.md` and `.txt` files. Only Markdown files are processed into chunks.

## Repository layout

- `api/` — FastAPI application.
- `config/` — settings and logging configuration.
- `scripts/` — Markdown loader utility and evaluation runner entry points.
- `src/rag/` — Markdown processing and RAG-related modules, including placeholders.
- `src/schemas/` — shared document, chunk, RAG, and API models.
- `src/pipeline/` — pipeline-related module currently containing a placeholder store module.
- `src/eval/` — evaluation-related modules.
- `ui/` — Streamlit entry file, currently a placeholder.
- `tests/` — test files; the current files do not contain executable test logic.
- `data/` — evaluation and data storage directories.
- `corpus/` — corpus directory used by local processing configuration.

## Known limitations

- There is no working end-to-end ingestion, embedding, vector indexing, retrieval, and answer-generation flow.
- The Markdown processor is not wired into the API; PDF and general text loaders are not implemented.
- The Streamlit entry file does not implement a UI.
- Evaluation runner files and current test files do not provide executable evaluation or test logic.
- The older `src/rag/md_loader.py` corpus adapter refers to names that are not defined or imported in that file.
- PyYAML is imported by the Markdown loader but is absent from the declared dependencies.

## Design principles

- Preserve original evidence and source provenance throughout processing.
- Keep deterministic parsing, chunking, metadata handling, validation, and storage separate from model interpretation.
- Use controlled vocabularies for classification and validate structured model output.
- Keep chunks traceable to source documents and relevant locations.
- Evaluate quality before adding complexity, and control processing cost.

## Future directions

The following are proposals, not implemented features or committed technology choices:

1. Add source adapters and structure-aware extraction for formats such as PDF, webpages, transcripts, DOCX, and presentations.
2. Add bounded semantic enrichment for topics, concepts, entities, decisions, and controlled glossary terms. The strategy document proposes evaluating a local Qwen3 4B model for these tasks; it is not currently wired into this repository.
3. Produce canonical Markdown with source metadata and provenance as an intermediate representation.
4. Improve chunking and metadata using document structure and meaning, then evaluate chunk sizes for the intended tasks.
5. Select and implement embedding and vector-storage components, followed by traceable retrieval and grounded answer generation.
6. Build evaluation and observability for retrieval quality, answer quality, failures, latency, and cost.
7. Consider hybrid retrieval, reranking, query reformulation, multi-corpus search, graph relationships, corpus governance, and agent integrations when evaluation or user needs justify them.

The target architecture is incremental: source adapters, extraction, enrichment, canonical documents, chunking, embeddings, vector storage, retrieval, RAG application, and evaluation/observability. Each stage should preserve provenance and expose validated outputs before later stages depend on them.
