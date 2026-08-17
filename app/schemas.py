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
    # Full rubrikkedja ("7. Anställning av universitetslektor >
    # 7.2 Behörighet"), byggd ur avsnittsnumreringen vid ingest.
    # None när dokumentet saknar numrerad struktur.
    section_path: str | None = None
    page_number: int | None = None
    # Dokumentets gällandedatum (ISO), extraherat ur dokumenthuvudet
    # i första hand (Beslutsdatum/Revised/Fastställd) och ur
    # filnamnskonventionen i andra hand. None när inget hittats.
    document_date: str | None = None
    # Diarienummer ur dokumenthuvudet (t.ex. "C 2025/1205") — nyckel
    # för framtida versionskedjor ("ersätter C 2022/43").
    diarienummer: str | None = None

    document_type: str | None = None
    # Normativ tyngd: norm | guidance | record. Härleds ur sökvägen
    # vid ingest och styr normkälleregeln i syntesen.
    document_weight: str | None = None
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


class EvidenceObject(BaseModel):
    evidence_id: str
    source_path: str
    file_name: str
    document_title: str | None = None
    section_title: str | None = None
    evidence_type: str
    evidence_text: str
    supporting_before: str | None = None
    supporting_after: str | None = None
    referring_passages: list[str] = Field(default_factory=list)
    source_fingerprint: str | None = None
    chunk_ids: list[str] = Field(default_factory=list)



class ChatRequest(BaseModel):
    question: str = Field(..., min_length=3)
    session_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceHit]
    session_id: str | None = None
    debug: dict[str, Any] | None = None
