from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
import hashlib

from docling.document_converter import DocumentConverter

from app.schemas import DocumentChunk, ChunkMetadata

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".xlsx"}
_converter = DocumentConverter()


@dataclass
class RawDocument:
    path: Path
    text: str
    title: str | None = None


def iter_document_paths(root: Path) -> list[Path]:
    return [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def extract_text_with_fallback(path: Path) -> RawDocument:
    try:
        result = _converter.convert(str(path))
        doc = result.document

        text = doc.export_to_markdown()
        if not text or not text.strip():
            text = doc.export_to_text()

        return RawDocument(
            path=path,
            text=text or "",
            title=path.stem,
        )
    except Exception as e:
        text = f"[EXTRACTION_FAILED: {path.name}: {type(e).__name__}: {e}]"
        return RawDocument(path=path, text=text, title=path.stem)


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def infer_category(path: Path, docs_root: Path) -> str | None:
    try:
        rel = path.relative_to(docs_root)
        return rel.parts[0] if len(rel.parts) > 1 else None
    except Exception:
        return None


def make_chunk_id(path: Path, idx: int, text: str) -> str:
    h = hashlib.sha1(f"{path}:{idx}:{text}".encode("utf-8")).hexdigest()
    return h


def ingest_path(path: Path, docs_root: Path) -> list[DocumentChunk]:
    raw = extract_text_with_fallback(path)
    pieces = chunk_text(raw.text)

    chunks: list[DocumentChunk] = []
    for idx, piece in enumerate(pieces):
        meta = ChunkMetadata(
            source_path=str(path),
            file_name=path.name,
            document_title=raw.title,
            category=infer_category(path, docs_root),
            section_title=None,
            page_number=None,
            document_date=None,
            chunk_index=idx,
        )
        chunks.append(
            DocumentChunk(
                chunk_id=make_chunk_id(path, idx, piece),
                text=piece,
                metadata=meta,
            )
        )
    return chunks