from pydantic import BaseModel, Field
from typing import Any

class ChunkMetadata(BaseModel):
    source_path: str
    file_name: str
    document_title: str | None = None
    category: str | None = None
    section_title: str | None = None
    page_number: int | None = None
    document_date: str | None = None
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