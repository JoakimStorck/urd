from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
import hashlib
import re

from docling.document_converter import DocumentConverter

from app.schemas import (
    DocumentChunk,
    ChunkMetadata,
    EvidenceObject,
    SectionSemanticMetadata,
)

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
        import logging
        logging.getLogger(__name__).warning(
            "Extraction failed for %s: %s: %s", path.name, type(e).__name__, e
        )
        return RawDocument(path=path, text="", title=path.stem)


def normalize_chunk_text(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines()]
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _build_context_prefix(
    document_title: str | None,
    section_title: str | None,
) -> str:
    """
    Bygg ett kontextuellt prefix som bäddas in i chunk-texten.

    Detta gör att embeddings fångar dokumentets kontext, inte bara
    den isolerade textbiten. En chunk som säger "detta gäller" får
    nu med sig *vad* och *var* i sin vektorrepresentation.
    """
    parts = []
    if document_title:
        parts.append(f"Dokument: {document_title}")
    if section_title:
        parts.append(f"Avsnitt: {section_title}")

    if not parts:
        return ""

    return "\n".join(parts) + "\n---\n"


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


def make_evidence_id(path: Path, kind: str, order: int, text: str) -> str:
    h = hashlib.sha1(f"{path}:{kind}:{order}:{text}".encode("utf-8")).hexdigest()
    return h


def _split_paragraphs(text: str) -> list[str]:
    blocks = [normalize_chunk_text(b) for b in re.split(r"\n\s*\n", text) if b.strip()]
    return [b for b in blocks if b]


def _is_table_block(block: str) -> bool:
    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
    if len(lines) < 2:
        return False
    has_pipe_rows = sum("|" in ln for ln in lines) >= 2
    has_separator = any(re.search(r"\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?", ln) for ln in lines)
    return has_pipe_rows and has_separator


def _is_bullet_list(block: str) -> bool:
    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
    return len(lines) >= 2 and all(re.match(r"^[-*•]\s+", ln) for ln in lines)


def _is_numbered_list(block: str) -> bool:
    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
    return len(lines) >= 2 and all(re.match(r"^\d+[\.)]\s+", ln) for ln in lines)


def _is_figure_block(block: str) -> bool:
    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
    if not lines:
        return False
    if any(re.match(r"^!\[.*\]\(.*\)$", ln) for ln in lines):
        return True
    first = lines[0]
    return bool(re.match(r"^(figur|figure)\s*\d*\s*[:.-]?\s+", first, flags=re.IGNORECASE))


def _figure_text(block: str) -> str:
    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
    caption = None
    for ln in lines:
        m = re.match(r"^(figur|figure)\s*\d*\s*[:.-]?\s*(.+)$", ln, flags=re.IGNORECASE)
        if m and m.group(2).strip():
            caption = m.group(2).strip()
            break
        img = re.match(r"^!\[(.*?)\]\(.*\)$", ln)
        if img and img.group(1).strip():
            caption = img.group(1).strip()
            break
    if caption:
        return f"[Figur: {caption}]"
    return "[Figur]"


def _table_text(block: str) -> str:
    return "[Tabell]\n" + normalize_chunk_text(block)


def _list_text(block: str, numbered: bool) -> str:
    label = "[Numrerad lista]" if numbered else "[Punktlista]"
    return label + "\n" + normalize_chunk_text(block)


def _build_referring_passages(
    paragraphs: list[str],
    evidence_type: str,
    evidence_text: str,
    block_index: int,
) -> list[str]:
    refs: list[str] = []
    if evidence_type == "figure":
        patterns = [r"\bfigur\b", r"\bfigure\b"]
    elif evidence_type == "table":
        patterns = [r"\btabell\b", r"\btable\b"]
    else:
        patterns = [r"\bföljande\b", r"\bnedanstående\b", r"\bovenstående\b", r"\bstegen\b", r"\bpunkterna\b"]

    figure_number = None
    m = re.search(r"\b(?:figur|figure)\s*(\d+)\b", evidence_text, flags=re.IGNORECASE)
    if m:
        figure_number = m.group(1)
    table_number = None
    m = re.search(r"\btabell\s*(\d+)\b", evidence_text, flags=re.IGNORECASE)
    if m:
        table_number = m.group(1)

    for idx, para in enumerate(paragraphs):
        if idx == block_index:
            continue
        low = para.casefold()
        if any(re.search(pat, low, flags=re.IGNORECASE) for pat in patterns):
            refs.append(para)
            continue
        if figure_number and re.search(rf"\bfigur\s*{re.escape(figure_number)}\b", para, flags=re.IGNORECASE):
            refs.append(para)
            continue
        if table_number and re.search(rf"\btabell\s*{re.escape(table_number)}\b", para, flags=re.IGNORECASE):
            refs.append(para)
            continue
    return refs[:4]


def extract_evidence_objects_from_sections(
    path: Path,
    document_title: str | None,
    sections: list[StructuredSection],
    source_fingerprint: str,
) -> list[EvidenceObject]:
    evidence_objects: list[EvidenceObject] = []
    order = 0

    for section in sections:
        paragraphs = _split_paragraphs(section.text)
        for idx, block in enumerate(paragraphs):
            evidence_type: str | None = None
            evidence_text: str | None = None

            if _is_figure_block(block):
                evidence_type = "figure"
                evidence_text = _figure_text(block)
            elif _is_table_block(block):
                evidence_type = "table"
                evidence_text = _table_text(block)
            elif _is_numbered_list(block):
                evidence_type = "numbered_list"
                evidence_text = _list_text(block, numbered=True)
            elif _is_bullet_list(block):
                evidence_type = "bullet_list"
                evidence_text = _list_text(block, numbered=False)

            if evidence_type is None or evidence_text is None:
                continue

            support_before = paragraphs[idx - 1] if idx > 0 else None
            support_after = paragraphs[idx + 1] if idx + 1 < len(paragraphs) else None
            referring = _build_referring_passages(
                paragraphs=paragraphs,
                evidence_type=evidence_type,
                evidence_text=evidence_text,
                block_index=idx,
            )

            evidence_objects.append(
                EvidenceObject(
                    evidence_id=make_evidence_id(path, evidence_type, order, evidence_text),
                    source_path=str(path),
                    file_name=path.name,
                    document_title=document_title,
                    section_title=section.title,
                    evidence_type=evidence_type,
                    evidence_text=evidence_text,
                    supporting_before=support_before,
                    supporting_after=support_after,
                    referring_passages=referring,
                    source_fingerprint=source_fingerprint,
                    chunk_ids=[],
                )
            )
            order += 1

    return evidence_objects


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
        context_prefix = _build_context_prefix(document_title, section.title)

        for piece in pieces:
            semantic = section.semantic or SectionSemanticMetadata()

            # Bädda in kontextuellt prefix i den text som indexeras
            contextualized_text = context_prefix + piece

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
                    text=contextualized_text,
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


def ingest_evidence_path(
    path: Path,
    docs_root: Path,
) -> list[EvidenceObject]:
    raw = extract_text_with_fallback(path)
    if not raw.text.strip():
        return []

    document_title = infer_document_title(raw)
    source_fingerprint = compute_source_fingerprint(path)
    sections = split_markdown_sections(raw.text)

    return extract_evidence_objects_from_sections(
        path=path,
        document_title=document_title,
        sections=sections,
        source_fingerprint=source_fingerprint,
    )
