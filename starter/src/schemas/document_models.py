"""Shared document and chunk models for file loaders in the RAG pipeline."""

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DocumentMetadata(BaseModel):
    """Metadata shared by documents, with source-specific data preserved."""

    model_config = ConfigDict(extra="allow")

    document_id: str | None = None
    source: str | None = None
    source_type: str | None = None
    document_type: str | None = None
    title: str | None = None
    language: str | None = None
    topics: list[str] = Field(default_factory=list)
    source_url: str | None = None
    retrieved_at: str | None = None
    provenance: str | None = None
    source_metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("topics", mode="before")
    @classmethod
    def normalize_topics(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, (list, tuple, set)):
            return [str(item) for item in value]
        raise ValueError("topics must be a string or a list of strings")

    @field_validator("retrieved_at", mode="before")
    @classmethod
    def normalize_retrieved_at(cls, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, (date, datetime)):
            return value.isoformat()
        return str(value)


class Document(BaseModel):
    """Format-neutral document produced by a file loader and parser."""

    model_config = ConfigDict(extra="forbid")

    path: str
    content: str
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)


class ChunkMetadata(BaseModel):
    """Common chunk metadata. Format-specific details stay in source_metadata."""

    model_config = ConfigDict(extra="allow")

    document_id: str
    document_type: str
    source: str | None = None
    file_name: str
    file_path: str
    section: str | None = None
    section_path: list[str] = Field(default_factory=list)
    page_number: int | None = None
    token_count: int = Field(ge=0)
    character_count: int = Field(ge=0)
    topics: list[str] = Field(default_factory=list)
    source_url: str | None = None
    retrieved_at: str | None = None
    provenance: str | None = None
    source_metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    """Format-neutral text chunk passed to embedding and retrieval stages."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    text: str = Field(min_length=1)
    metadata: ChunkMetadata


class ChunkingConfig(BaseModel):
    """Shared chunk sizing configuration for all document types."""

    max_chunk_chars: int = Field(default=500, ge=1)
