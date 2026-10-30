"""Markdown file loading, front-matter parsing, sectioning, and chunking.

This module uses the shared document and chunk models. Markdown-specific
structure is represented by MarkdownSection and stored in chunk metadata.
"""

import asyncio
import hashlib
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from config.logging_config import setup_logger
from config.settings import settings
from src.rag.base_loader import BaseDocumentLoader
from src.schemas.document_models import (
    ChunkingConfig,
    ChunkMetadata,
    Document,
    DocumentChunk,
    DocumentMetadata,
)

logger = setup_logger(__name__)


class MarkdownSection(BaseModel):
    """Markdown-specific section identified by its heading hierarchy."""

    model_config = ConfigDict(extra="forbid")

    level: int = Field(ge=1, le=6)
    title: str
    heading_path: list[str] = Field(default_factory=list)
    content: str


class MarkdownLoader(BaseDocumentLoader):
    """Read one Markdown file as UTF-8 text without blocking the event loop."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    async def load(self) -> str:
        if not self.path.exists():
            raise FileNotFoundError(f"File not found: {self.path}")
        if not self.path.is_file():
            raise ValueError(f"Expected a file: {self.path}")
        if self.path.suffix.lower() != ".md":
            raise ValueError(f"Expected a Markdown file: {self.path}")
        logger.info("Loading Markdown file: %s", self.path)
        return await asyncio.to_thread(self.path.read_text, encoding="utf-8")


class MarkdownParser:
    """Parse optional YAML front matter and Markdown heading sections."""

    FRONT_MATTER_END_MARKERS = {"---", "..."}
    _HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
    _FENCE_RE = re.compile(r"^(`{3,}|~{3,})(.*)$")

    def parse(self, raw_markdown: str, path: str | Path) -> Document:
        raw_metadata, body = self._split_front_matter(raw_markdown)
        known_fields = set(DocumentMetadata.model_fields)
        common = {key: value for key, value in raw_metadata.items() if key in known_fields}
        source_metadata = dict(raw_metadata.get("source_metadata") or {})
        source_metadata.update(
            {key: value for key, value in raw_metadata.items()
             if key not in known_fields and key != "source_metadata"}
        )
        common["source_metadata"] = source_metadata
        try:
            metadata = DocumentMetadata.model_validate(common)
        except ValidationError:
            logger.exception("Invalid Markdown metadata in %s", path)
            raise
        return Document(path=str(path), content=body, metadata=metadata)

    @classmethod
    def _split_front_matter(cls, raw_markdown: str) -> tuple[dict[str, Any], str]:
        lines = raw_markdown.splitlines()
        if not lines or lines[0].strip() != "---":
            return {}, raw_markdown.strip()

        closing = next(
            (i for i in range(1, len(lines)) if lines[i].strip() in cls.FRONT_MATTER_END_MARKERS),
            None,
        )
        if closing is None:
            raise ValueError("Markdown starts with '---' but has no closing front matter marker")

        try:
            metadata = yaml.safe_load("\n".join(lines[1:closing])) or {}
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML front matter: {exc}") from exc
        if not isinstance(metadata, dict):
            raise ValueError("YAML front matter must contain a mapping of fields")
        return metadata, "\n".join(lines[closing + 1:]).strip()

    def parse_sections(self, document: Document) -> list[MarkdownSection]:
        sections: list[MarkdownSection] = []
        heading_path: list[str] = []
        level, title = 1, "Document"
        current: list[str] = []
        fence_char: str | None = None
        fence_length = 0

        def flush() -> None:
            content = "\n".join(current).strip()
            if content:
                sections.append(MarkdownSection(
                    level=level, title=title,
                    heading_path=heading_path.copy(), content=content,
                ))
            current.clear()

        for line in document.content.splitlines():
            stripped = line.strip()
            fence = self._FENCE_RE.match(stripped)
            if fence:
                marker = fence.group(1)
                char = marker[0]
                if fence_char is None:
                    fence_char, fence_length = char, len(marker)
                elif char == fence_char and len(marker) >= fence_length and not fence.group(2).strip():
                    fence_char, fence_length = None, 0
                current.append(line)
                continue
            if fence_char is not None:
                current.append(line)
                continue

            match = self._HEADING_RE.match(line)
            if match:
                flush()
                level, title = len(match.group(1)), match.group(2).strip()
                heading_path[:] = heading_path[:level - 1]
                heading_path.append(title)
            else:
                current.append(line)
        flush()
        return sections


class MarkdownChunker:
    """Pack whole paragraphs within each Markdown section."""

    _FENCE_RE = re.compile(r"^(`{3,}|~{3,})(.*)$")

    def __init__(self, config: ChunkingConfig | None = None):
        self.config = config or ChunkingConfig()

    def chunk(self, document: Document, sections: list[MarkdownSection]) -> list[DocumentChunk]:
        document_id = document.metadata.document_id or hashlib.sha1(
            str(Path(document.path).resolve()).encode("utf-8")
        ).hexdigest()[:16]
        document_type = document.metadata.source_type or document.metadata.document_type or "markdown"
        chunks: list[DocumentChunk] = []

        for section in sections:
            for text in self._pack_paragraphs(section.content):
                index = len(chunks)
                digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
                metadata = ChunkMetadata(
                    document_id=document_id,
                    document_type=document_type,
                    source=document.metadata.source,
                    file_name=Path(document.path).name,
                    file_path=str(Path(document.path).resolve()),
                    section=section.title,
                    section_path=section.heading_path,
                    token_count=len(re.findall(r"\S+", text)),
                    character_count=len(text),
                    topics=document.metadata.topics,
                    source_url=document.metadata.source_url,
                    retrieved_at=document.metadata.retrieved_at,
                    provenance=document.metadata.provenance,
                    source_metadata=document.metadata.source_metadata,
                )
                chunks.append(DocumentChunk(
                    chunk_id=f"{document_id}-chunk-{index:04d}-{digest}",
                    text=text,
                    metadata=metadata,
                ))
        return chunks

    def _pack_paragraphs(self, text: str) -> list[str]:
        paragraphs = self._split_paragraphs(text)
        packed: list[str] = []
        current: list[str] = []
        size = 0
        for paragraph in paragraphs:
            candidate_size = size + (2 if current else 0) + len(paragraph)
            if current and candidate_size > self.config.max_chunk_chars:
                packed.append("\n\n".join(current))
                current, size = [paragraph], len(paragraph)
            else:
                current.append(paragraph)
                size = candidate_size
        if current:
            packed.append("\n\n".join(current))
        return packed

    @classmethod
    def _split_paragraphs(cls, text: str) -> list[str]:
        result: list[str] = []
        current: list[str] = []
        fence_char: str | None = None
        fence_length = 0

        def flush() -> None:
            block = "\n".join(current).strip()
            if block:
                result.append(block)
            current.clear()

        for line in text.splitlines():
            stripped = line.strip()
            fence = cls._FENCE_RE.match(stripped)
            if fence:
                marker = fence.group(1)
                char = marker[0]
                if fence_char is None:
                    fence_char, fence_length = char, len(marker)
                elif char == fence_char and len(marker) >= fence_length and not fence.group(2).strip():
                    fence_char, fence_length = None, 0
                current.append(line)
            elif fence_char is None and not stripped:
                flush()
            else:
                current.append(line)
        flush()
        return result


async def process_markdown_file(
    path: str | Path,
    parser: MarkdownParser | None = None,
    chunker: MarkdownChunker | None = None,
) -> list[DocumentChunk]:
    """Load, parse, and chunk one Markdown file."""
    parser = parser or MarkdownParser()
    chunker = chunker or MarkdownChunker()
    raw = await MarkdownLoader(path).load()
    document = parser.parse(raw, path)
    return chunker.chunk(document, parser.parse_sections(document))


async def load_markdown_chunks(
    corpus_dir: str | Path,
    max_chunk_chars: int | None = None,
) -> list[DocumentChunk]:
    """Recursively load Markdown files and return shared DocumentChunk models."""
    corpus_dir = Path(corpus_dir)
    if not corpus_dir.is_dir():
        raise NotADirectoryError(f"Corpus directory not found: {corpus_dir}")
    config = ChunkingConfig(
        max_chunk_chars=max_chunk_chars or settings.TARGET_CHUNK_SIZE
    )
    parser = MarkdownParser()
    chunker = MarkdownChunker(config)
    paths = sorted(path for path in corpus_dir.rglob("*") if path.is_file() and path.suffix.lower() == ".md")

    async def process(path: Path) -> list[DocumentChunk]:
        try:
            return await process_markdown_file(path, parser, chunker)
        except (ValueError, ValidationError, FileNotFoundError):
            logger.exception("Skipping invalid Markdown document: %s", path)
            return []

    results = await asyncio.gather(*(process(path) for path in paths))
    return [chunk for result in results for chunk in result]
