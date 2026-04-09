from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
import hashlib
import re

from docling.document_converter import DocumentConverter

from app.schemas import DocumentChunk, ChunkMetadata, SectionSemanticMetadata

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".xlsx"}
_converter = DocumentConverter()


@dataclass
class RawDocument:
    path: Path
    text: str
    title: str | None = None


@dataclass
class StructuredSection:
    title: str | None
    level: int | None
    text: str
    order: int
    semantic: SectionSemanticMetadata | None = None


def iter_document_paths(root: Path) -> list[Path]:
    return [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def compute_source_fingerprint(path: Path) -> str:
    st = path.stat()
    raw = f"{path}:{st.st_size}:{st.st_mtime_ns}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


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


def normalize_chunk_text(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines()]
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> list[str]:
    text = normalize_chunk_text(text)
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end == n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def split_markdown_sections(md: str) -> list[StructuredSection]:
    md = md.strip()
    if not md:
        return []

    heading_re = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
    lines = md.splitlines()

    sections: list[StructuredSection] = []
    current_title: str | None = None
    current_level: int | None = None
    current_lines: list[str] = []
    order = 0

    def flush_current() -> None:
        nonlocal order, current_lines, current_title, current_level
        text = "\n".join(current_lines).strip()
        if text:
            sections.append(
                StructuredSection(
                    title=current_title,
                    level=current_level,
                    text=text,
                    order=order,
                )
            )
            order += 1
        current_lines = []

    for line in lines:
        m = heading_re.match(line)
        if m:
            flush_current()
            current_level = len(m.group(1))
            current_title = m.group(2).strip()
        else:
            current_lines.append(line)

    flush_current()

    if sections:
        return sections

    blocks = [b.strip() for b in re.split(r"\n\s*\n", md) if b.strip()]
    return [
        StructuredSection(
            title=None,
            level=None,
            text=block,
            order=i,
        )
        for i, block in enumerate(blocks)
    ]


def infer_document_title(raw: RawDocument) -> str | None:
    md = raw.text or ""
    for line in md.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("# "):
            return line[2:].strip()
        break
    return raw.title


def infer_category(path: Path, docs_root: Path) -> str | None:
    try:
        rel = path.relative_to(docs_root)
        parts = rel.parts

        if len(parts) >= 3 and parts[0] in {"IIT-lokala regler och rutiner", "IIT-lokala-regler-och-rutiner"}:
            return parts[1]

        if len(parts) > 1:
            return parts[0]

        return None
    except Exception:
        return None


def make_chunk_id(path: Path, idx: int, text: str) -> str:
    h = hashlib.sha1(f"{path}:{idx}:{text}".encode("utf-8")).hexdigest()
    return h


def build_chunks_from_sections(
    path: Path,
    document_title: str | None,
    category: str | None,
    sections: list[StructuredSection],
    source_fingerprint: str,
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    global_idx = 0

    for section in sections:
        pieces = chunk_text(section.text)

        for piece in pieces:
            semantic = section.semantic or SectionSemanticMetadata()

            meta = ChunkMetadata(
                source_path=str(path),
                file_name=path.name,
                document_title=document_title,
                category=category,
                section_title=section.title,
                section_level=section.level,
                page_number=None,
                document_date=None,
                document_type=semantic.document_type,
                keywords=semantic.keywords,
                roles=semantic.roles,
                actions=semantic.actions,
                time_markers=semantic.time_markers,
                applies_to=semantic.applies_to,
                section_summary=semantic.summary,
                source_fingerprint=source_fingerprint,
                semantic_enriched=False,
                semantic_model=None,
                semantic_version=None,
                semantic_source_hash=None,
                chunk_index=global_idx,
            )
            chunks.append(
                DocumentChunk(
                    chunk_id=make_chunk_id(path, global_idx, piece),
                    text=piece,
                    metadata=meta,
                )
            )
            global_idx += 1

    return chunks


def ingest_path(
    path: Path,
    docs_root: Path,
) -> list[DocumentChunk]:
    raw = extract_text_with_fallback(path)

    if not raw.text.strip():
        return []

    document_title = infer_document_title(raw)
    category = infer_category(path, docs_root)
    source_fingerprint = compute_source_fingerprint(path)

    sections = split_markdown_sections(raw.text)

    return build_chunks_from_sections(
        path=path,
        document_title=document_title,
        category=category,
        sections=sections,
        source_fingerprint=source_fingerprint,
    )