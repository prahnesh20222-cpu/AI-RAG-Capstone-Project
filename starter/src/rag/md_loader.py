import asyncio
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field, ConfigDict, field_validator
from datetime import date, datetime


from config.logging_config import setup_logger #logger module
from config.settings import settings

logger = setup_logger(__name__)
logger.info("Document loader module initialized.")

# Template for our file loaders
class BaseDocumentLoader(ABC):
    def __init__(self, path: str | Path):
        self.path = Path(path)

    @abstractmethod
    async def load(self) -> str:
        pass

#DocumentMetadata is a data model. It defines the metadata fields that a Markdown document can contain.
class DocumentMetadata(BaseModel):
    """
    Metadata extracted from the YAML front matter of a Markdown document.

    The corpus strategy deliberately allows different source types to carry
    different metadata. Therefore extra fields are preserved rather than
    rejected. Known fields are typed and validated by Pydantic.
    """

    model_config = ConfigDict(extra="allow") # This allows extra fields in the metadata to be preserved rather than rejected.

    document_id: str | None = None
    source: str | None = None
    source_type: str | None = None
    document_type: str | None = None

    session_type: str | None = None
    session_date: str | None = None
    language: str | None = None
    technical_depth: str | None = None

    rag_ready: bool | None = None
    chunking_strategy: str | None = None

    topics: list[str] = Field(default_factory=list)
    glossary_terms: list[str] = Field(default_factory=list)

    source_url: str | None = None
    retrieved_at: str | None = None
    provenance: str | None = None

    speaker_names_preserved: bool | None = None
    transcript_cleaned: bool | None = None

    # Validate the topics and glossary_terms fields.
    # Run this function before normal field validation
    @field_validator("topics", "glossary_terms", mode="before")
    @classmethod
    def normalize_string_lists(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, (list, tuple, set)):
            return [str(item) for item in value]
        raise TypeError("Expected a string or a list of strings.")

    # Validate the session_date, retrieved_at fields.
    # Run this function before normal field validation
    @field_validator("session_date", "retrieved_at", mode="before")
    @classmethod
    def normalize_dates(cls, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, date):
            return value.isoformat()
        return str(value)

class MarkdownDocument(BaseModel):
    model_config = ConfigDict(extra="forbid") #good practice to forbid extra fields in a class unless required.

    path: str
    metadata: DocumentMetadata #belogs to DocumentMetadata class
    body: str

class MarkdownSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: int = Field(ge=1, le=6) # Valid range for Markdown headers.
    title: str
    heading_path: list[str]
    content: str

class ChunkMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    document_id: str
    document_type: str
    source: str
    file_name: str
    file_path: str

    section: str
    section_path: list[str]

    token_count: int = Field(ge=0)
    character_count: int = Field(ge=0)

    topic: str | None = None
    topics: list[str] = Field(default_factory=list)
    glossary_terms: list[str] = Field(default_factory=list)

    speaker: str | None = None
    timestamp: str | None = None

    source_url: str | None = None
    retrieved_at: str | None = None
    provenance: str | None = None

class DocumentChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    text: str = Field(min_length=1)
    metadata: ChunkMetadata


class ChunkingConfig(BaseModel):
    max_chunk_chars: int = Field(default=500, ge=1)


class MarkdownLoader(BaseDocumentLoader):
    """Loads Markdown files (.md)."""

    async def load(self) -> str:
        logger.info(f"Loading markdown file: {self.path.name}")

        if not self.path.exists():
            logger.error(f"File not found: {self.path}")
            raise FileNotFoundError(f"File not found: {self.path}")

        if self.path.suffix.lower() != ".md":
            raise ValueError(f"Expected a Markdown file: {self.path}")

        # Run synchronous file I/O in a separate thread and expose it through an asynchronous interface.
        # Multiple files can be processed concurrently.
        loop = asyncio.get_running_loop()

        def _read_file() -> str:
            try:
                with self.path.open("r", encoding="utf-8") as file:
                    return file.read()
            except Exception:
                logger.exception(f"Error reading markdown file: {self.path}")
                raise
        
        return await loop.run_in_executor(None, _read_file) 

class MarkdownParser:
    FRONT_MATTER_START = "---"
    FRONT_MATTER_END_MARKERS = {"---", "..."}

    def parse(
        self,
        raw_markdown: str,
        path: str | Path,
    ) -> MarkdownDocument:
        metadata_dict, body = self._split_front_matter(raw_markdown)

        try:
            metadata = DocumentMetadata.model_validate(metadata_dict)
            return MarkdownDocument(
                path=str(path),
                metadata=metadata,
                body=body,
            )
        except ValidationError:
            logger.exception("Invalid document metadata: %s", path)
            raise

    def _split_front_matter(
        self,
        raw_markdown: str,
    ) -> tuple[dict[str, Any], str]:
        lines = raw_markdown.splitlines()

        if not lines or lines[0].strip() != self.FRONT_MATTER_START:
            logger.warning("No YAML front matter found")
            return {}, raw_markdown.strip()

        closing_index = None

        for index in range(1, len(lines)):
            if lines[index].strip() in self.FRONT_MATTER_END_MARKERS:
                closing_index = index
                break

        if closing_index is None:
            raise ValueError(
                "Markdown starts with YAML front matter marker '---', "
                "but no closing YAML marker was found."
            )

        yaml_text = "\n".join(lines[1:closing_index])

        try:
            metadata = yaml.safe_load(yaml_text) or {}
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML front matter: {exc}") from exc

        if not isinstance(metadata, dict):
            raise ValueError(
                "YAML front matter must contain a mapping of metadata fields."
            )

        body = "\n".join(lines[closing_index + 1:]).strip()

        return metadata, body

    def parse_sections(
        self,
        document: MarkdownDocument,
    ) -> list[MarkdownSection]:
        lines = document.body.splitlines()
        sections: list[MarkdownSection] = []

        heading_path: list[str] = []
        current_level = 1
        current_title = "Document"
        current_lines: list[str] = []

        in_fenced_code = False
        fence_character: str | None = None

        def flush() -> None:
            nonlocal current_lines

            content = "\n".join(current_lines).strip()

            if content:
                sections.append(
                    MarkdownSection(
                        level=current_level,
                        title=current_title,
                        heading_path=heading_path.copy(),
                        content=content,
                    )
                )

            current_lines = []

        for line in lines:
            stripped = line.strip()
            fence_match = re.match(r"^(`{3,}|~{3,})(.*)$", stripped)

            if fence_match:
                marker = fence_match.group(1)[0]

                if not in_fenced_code:
                    in_fenced_code = True
                    fence_character = marker
                elif marker == fence_character:
                    in_fenced_code = False
                    fence_character = None

                current_lines.append(line)
                continue

            if in_fenced_code:
                current_lines.append(line)
                continue

            match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)

            if match:
                flush()

                current_level = len(match.group(1))
                current_title = match.group(2).strip()

                heading_path[:] = heading_path[: current_level - 1]
                heading_path.append(current_title)
            else:
                current_lines.append(line)

        flush()
        return sections


class TokenCounter:
    """Retained utility from the Markdown chunker."""

    def __init__(self) -> None:
        self._encoder = None

        try:
            import tiktoken

            self._encoder = tiktoken.get_encoding("cl100k_base")
            logger.info("Using tiktoken cl100k_base for token estimates")
        except ImportError:
            logger.warning(
                "tiktoken is not installed; using whitespace token estimates"
            )
        except Exception:
            logger.exception("Could not initialize tiktoken")
            self._encoder = None

    def count(self, text: str) -> int:
        if self._encoder is not None:
            return len(self._encoder.encode(text))
        return len(re.findall(r"\S+", text))


class MarkdownChunker:
    """
    Section-aware, paragraph-preserving Markdown chunker.

    Markdown headings define semantic boundaries, paragraphs are
    packed within a section up to max_chunk_chars, and paragraphs are never
    split.
    """

    def __init__(self, config: ChunkingConfig):
        self.config = config

    def chunk(
        self,
        document: MarkdownDocument,
        sections: list[MarkdownSection],
    ) -> list[DocumentChunk]:
        document_id = self._document_id(document)

        document_type = (
            document.metadata.source_type
            or document.metadata.document_type
            or "markdown"
        )

        chunks: list[DocumentChunk] = []

        for section in sections:
            if not section.content.strip():
                continue

            paragraphs = self._split_paragraphs(section.content)

            section_chunks = self._chunk_section(
                section=section,
                paragraphs=paragraphs,
            )

            for text in section_chunks:
                chunk_index = len(chunks)

                metadata = ChunkMetadata(
                    document_id=document_id,
                    document_type=document_type,
                    source=document.metadata.source or "",
                    file_name=Path(document.path).name,
                    file_path=str(Path(document.path).resolve()),
                    section=section.title,
                    section_path=section.heading_path,
                    token_count=self._estimate_token_count(text),
                    character_count=len(text),
                    topics=document.metadata.topics,
                    glossary_terms=document.metadata.glossary_terms,
                    source_url=document.metadata.source_url,
                    retrieved_at=document.metadata.retrieved_at,
                    provenance=document.metadata.provenance,
                )

                chunk_id = self._chunk_id(
                    document_id=document_id,
                    index=chunk_index,
                    text=text,
                )

                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        text=text,
                        metadata=metadata,
                    )
                )

        return chunks

    def _chunk_section(
        self,
        section: MarkdownSection,
        paragraphs: list[str],
    ) -> list[str]:
        chunks: list[str] = []
        current_paragraphs: list[str] = []
        current_size = 0

        for paragraph in paragraphs:
            paragraph_size = len(paragraph)

            if not current_paragraphs:
                current_paragraphs = [paragraph]
                current_size = paragraph_size
                continue

            candidate_size = (
                current_size
                + 2
                + paragraph_size
            )

            if candidate_size > self.config.max_chunk_chars:
                chunks.append("\n\n".join(current_paragraphs))

                current_paragraphs = [paragraph]
                current_size = paragraph_size
            else:
                current_paragraphs.append(paragraph)
                current_size = candidate_size

        if current_paragraphs:
            chunks.append("\n\n".join(current_paragraphs))

        return chunks

    @staticmethod
    def _split_paragraphs(text: str) -> list[str]:
        lines = text.splitlines()
        blocks: list[str] = []
        current: list[str] = []

        in_fenced_code = False
        fence_character: str | None = None

        def flush() -> None:
            if current:
                block = "\n".join(current).strip()
                if block:
                    blocks.append(block)
                current.clear()

        for line in lines:
            stripped = line.strip()
            fence_match = re.match(r"^(`{3,}|~{3,})(.*)$", stripped)

            if fence_match:
                marker = fence_match.group(1)[0]

                if not in_fenced_code:
                    in_fenced_code = True
                    fence_character = marker
                elif marker == fence_character:
                    in_fenced_code = False
                    fence_character = None

                current.append(line)
                continue

            if not in_fenced_code and not stripped:
                flush()
                continue

            current.append(line)

        flush()
        return blocks

    @staticmethod
    def _estimate_token_count(text: str) -> int:
        return len(re.findall(r"\S+", text))

    @staticmethod
    def _document_id(document: MarkdownDocument) -> str:
        if document.metadata.document_id:
            return document.metadata.document_id

        path_bytes = str(Path(document.path).resolve()).encode("utf-8")
        return hashlib.sha1(path_bytes).hexdigest()[:16]

    @staticmethod
    def _chunk_id(
        document_id: str,
        index: int,
        text: str,
    ) -> str:
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
        return f"{document_id}-chunk-{index:04d}-{digest}"

# ===========================================================================
# CORPUS ADAPTER
# ===========================================================================


def load_markdown_chunks(
    corpus_dir: Path,
    max_chunk_chars: int = TARGET_CHUNK_SIZE,
) -> list[DocumentChunk]:
    """Load and chunk every Markdown file using the supplied Markdown logic."""

    markdown_files = sorted(corpus_dir.rglob("*.md"))
    parser = MarkdownParser()
    chunker = MarkdownChunker(
        config=ChunkingConfig(max_chunk_chars=settings.TARGET_CHUNK_SIZE)
    )

    all_chunks: list[DocumentChunk] = []

    for path in markdown_files:
        try:
            chunks = asyncio.run(
                process_markdown_file(path, parser, chunker)
            )
            all_chunks.extend(chunks)
            print(f"  {path.name}: {len(chunks)} chunks")
        except (ValueError, ValidationError, FileNotFoundError):
            logger.exception("Skipping invalid Markdown document: %s", path)

    return all_chunks


async def process_markdown_file(
    path: Path,
    parser: MarkdownParser,
    chunker: MarkdownChunker,
) -> list[DocumentChunk]:
    loader = MarkdownLoader(path)
    raw_markdown = await loader.load()
    document = parser.parse(raw_markdown=raw_markdown, path=path)
    sections = parser.parse_sections(document)
    chunks = chunker.chunk(document=document, sections=sections)

    logger.info("Processed %s -> %d chunks", path.name, len(chunks))
    return chunks


def document_chunk_to_payload(chunk: DocumentChunk) -> dict[str, Any]:
    """
    Convert a validated Markdown chunk into the common RAG payload shape.

    The original Markdown metadata remains nested under `metadata`; the few
    top-level fields make retrieval/citation code compatible with the text
    corpus representation.
    """

    metadata = chunk.metadata.model_dump(mode="json")
    return {
        "chunk_id": chunk.chunk_id,
        "source": metadata["source"] or metadata["file_name"],
        "title": metadata["section"],
        "section": metadata["section"],
        "text": chunk.text,
        "document_type": metadata["document_type"],
        "metadata": metadata,
    }


def load_corpus(corpus_dir: Path) -> list[dict[str, Any]]:
    """
    Load and chunk both supported source types.

    .txt files are discovered recursively and use the original Mini-RAG
    loader + heuristic section-aware chunker.
    .md files are discovered recursively and use the supplied Markdown
    parser/chunker path.
    """

    corpus: list[dict[str, Any]] = []

    # Preserve the source-specific chunking strategy:
    #   .txt -> original Mini-RAG text loader + heuristic chunker
    #   .md  -> Markdown parser + section-aware chunker
    text_docs = load_text_corpus(corpus_dir)
    for doc in text_docs:
        corpus.extend(chunk_text_document(doc))

    markdown_chunks = load_markdown_chunks(corpus_dir)
    corpus.extend(
        document_chunk_to_payload(chunk)
        for chunk in markdown_chunks
    )

    return corpus
