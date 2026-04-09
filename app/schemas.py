from pydantic import BaseModel, Field
from typing import Any


class SectionSemanticMetadata(BaseModel):
    document_type: str | None = None
    keywords: list[str] = Field(default_factory=list)
    roles: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    time_markers: list[str] = Field(default_factory=list)
    applies_to: list[str] = Field(default_factory=list)
    summary: str | None = None


class ChunkMetadata(BaseModel):
    source_path: str
    file_name: str
    document_title: str | None = None
    category: str | None = None
    section_title: str | None = None
    section_level: int | None = None
    page_number: int | None = None
    document_date: str | None = None

    document_type: str | None = None
    keywords: list[str] = Field(default_factory=list)
    roles: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    time_markers: list[str] = Field(default_factory=list)
    applies_to: list[str] = Field(default_factory=list)
    section_summary: str | None = None

    source_fingerprint: str | None = None
    semantic_enriched: bool = False
    semantic_model: str | None = None
    semantic_version: str | None = None
    semantic_source_hash: str | None = None

    chunk_index: int = 0


class DocumentChunk(BaseModel):
    chunk_id: str
    text: str
    metadata: ChunkMetadata


class SourceHit(BaseModel):
    chunk_id: str
    score: float
    text: str
    metadata: ChunkMetadata


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=3)


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceHit]
    debug: dict[str, Any] | None = None