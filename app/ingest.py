from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
import hashlib
import logging
import re

from docling.document_converter import DocumentConverter

from app.config import settings

from app.schemas import (
    DocumentChunk,
    ChunkMetadata,
    EvidenceObject,
)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".xlsx"}
_converter = DocumentConverter()


@dataclass
class RawDocument:
    path: Path
    text: str
    title: str | None = None
    # Satt när extraktionen misslyckades med ett undantag. Tom text
    # UTAN error betyder att konverteringen lyckades men inte gav
    # någon text (t.ex. helt bildbaserat dokument utan OCR-träff).
    # Skillnaden är diagnostiskt viktig och ska visas för användaren.
    error: str | None = None


@dataclass
class StructuredSection:
    title: str | None
    level: int | None
    text: str
    order: int


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
        return RawDocument(
            path=path,
            text="",
            title=path.stem,
            error=f"{type(e).__name__}: {e}",
        )


def normalize_chunk_text(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines()]
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _build_context_prefix(
    document_title: str | None,
    section_title: str | None,
    ancestors: list[str] | None = None,
) -> str:
    """
    Bygg ett kontextuellt prefix som bäddas in i chunk-texten.

    Detta gör att embeddings fångar dokumentets kontext, inte bara
    den isolerade textbiten. En chunk som säger "detta gäller" får
    nu med sig *vad* och *var* i sin vektorrepresentation.

    Avsnittsraden bär hela rubrikkedjan när den är känd, så att
    "7.2 Behörighet" och "8.2 Behörighet" blir olika texter för
    embeddingmodellen och cross-encodern i stället för två nästan
    identiska strängar.
    """
    parts = []
    if document_title:
        parts.append(f"Dokument: {document_title}")

    chain = [t for t in (ancestors or []) if t]
    if section_title:
        chain = chain + [section_title]
    if chain:
        parts.append("Avsnitt: " + " > ".join(chain))

    if not parts:
        return ""

    return "\n".join(parts) + "\n---\n"


# Tillåtna klippunkter för chunkning: efter meningsslut följt av
# whitespace, eller vid radbrytning (bevarar list- och tabellrader
# hela). Positionerna är index DÄR nästa segment börjar.
_CHUNK_BOUNDARY_RE = re.compile(r"(?<=[.!?:;])\s+|\n+")


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> list[str]:
    """
    Dela text i chunkar om ~chunk_size tecken med klipp på menings-
    eller radgräns.

    Tidigare klipptes texten på exakt teckenposition, vilket delade
    ord och meningar mitt itu ("...decided by" | "y Vice Chancellor...").
    Det skadar både cross-encoderns bedömning och syntesens källtrohet.

    Nu väljs klippunkten som den sista tillåtna gränsen (meningsslut
    eller radbrytning) i intervallet (start + chunk_size/2, start +
    chunk_size]. Finns ingen gräns där — t.ex. en mycket lång mening
    eller tabellrad — faller vi tillbaka på hårt klipp vid chunk_size,
    så att chunkar aldrig växer obegränsat.

    Överlappet är också gränsmedvetet: nästa chunk börjar vid den
    första gränsen inom [klipp - overlap, klipp), så att upprepad
    text alltid börjar på en hel mening/rad. Finns ingen gräns i
    fönstret blir det inget överlapp — ett rent meningsklipp behöver
    det sällan.
    """
    text = normalize_chunk_text(text)
    if not text:
        return []

    n = len(text)
    if n <= chunk_size:
        return [text]

    boundaries = [m.end() for m in _CHUNK_BOUNDARY_RE.finditer(text)]

    chunks: list[str] = []
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            candidates = [
                b for b in boundaries
                if start + chunk_size // 2 < b <= end
            ]
            if candidates:
                end = candidates[-1]

        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)

        if end >= n:
            break

        overlap_candidates = [
            b for b in boundaries
            if end - overlap <= b < end
        ]
        next_start = overlap_candidates[0] if overlap_candidates else end
        start = max(next_start, start + 1)

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


# ---------------------------------------------------------------------------
# Datum och diarienummer ur dokumenthuvudet.
#
# INNEHÅLLET är primärkälla (dokumentlagret kan i framtiden flyttas
# från filer till databas — filnamnskonventionen är sekundär).
# BeHDa-mallen ger 'Beslutsdatum 2025-05-06' och 'Diarienummer
# C 2025/1205' i dokumentets huvud; bilagor bär 'Revised 2025-03-11';
# centrala styrdokument har fastställandefraser. Filnamnets
# '20250311_...' och 'rev20250909' används bara när innehållet inte
# ger något. Hellre None än gissning.
# ---------------------------------------------------------------------------

_ISO_DATE = r"(\d{4}-\d{2}-\d{2})"

# Prioritetsordnade innehållsmönster. Revisionsdatum före
# beslutsdatum — det är revisionen som anger gällande version.
_CONTENT_DATE_PATTERNS = [
    re.compile(r"[Rr]evi(?:sed|derad)(?:\s+den)?\s*:?\s*" + _ISO_DATE),
    re.compile(r"Beslutsdatum\s*:?\s*\n?\s*" + _ISO_DATE),
    re.compile(
        r"(?:Fastställd|Fastställt|Beslutad|Beslutat|Gäller från(?:\s+och\s+med)?)"
        r"(?:\s+den)?\s*:?\s*" + _ISO_DATE,
    ),
]

_FILENAME_REV_RE = re.compile(r"rev(\d{8})")
_FILENAME_DATE_RE = re.compile(r"^(\d{8})[_\s-]|^(\d{4}-\d{2}-\d{2})")

_DIARIENUMMER_RE = re.compile(
    r"Diarienummer\s*:?\s*\n?\s*([A-ZÅÄÖ][A-Za-zÅÄÖåäö]{0,3}\s?\d{4}\s?[/-]\s?\d+)"
)

# Hur långt in i dokumentet huvudet får sträcka sig. BeHDa-huvudet
# ligger först, men docling kan kasta om sektionsordningen något.
_HEADER_CHARS = 3000


def _valid_iso_date(date_str: str) -> bool:
    try:
        year, month, day = (int(p) for p in date_str.split("-"))
    except ValueError:
        return False
    return 1990 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31


def _compact_to_iso(compact: str) -> str | None:
    iso = f"{compact[0:4]}-{compact[4:6]}-{compact[6:8]}"
    return iso if _valid_iso_date(iso) else None


def extract_document_header_info(
    text: str,
    file_name: str,
) -> tuple[str | None, str | None, str | None]:
    """
    Extrahera (document_date, diarienummer, datumkälla) för ett dokument.

    document_date är ISO-format eller None. datumkälla är "innehåll",
    "filnamn" eller None — för loggning och tester, lagras inte.
    """
    header = text[:_HEADER_CHARS] if text else ""

    document_date: str | None = None
    date_source: str | None = None

    for pattern in _CONTENT_DATE_PATTERNS:
        m = pattern.search(header)
        if m and _valid_iso_date(m.group(1)):
            document_date = m.group(1)
            date_source = "innehåll"
            break

    if document_date is None:
        m = _FILENAME_REV_RE.search(file_name)
        if m:
            document_date = _compact_to_iso(m.group(1))
        if document_date is None:
            m = _FILENAME_DATE_RE.match(file_name)
            if m:
                raw_date = m.group(1) or m.group(2)
                document_date = (
                    _compact_to_iso(raw_date) if len(raw_date) == 8
                    else (raw_date if _valid_iso_date(raw_date) else None)
                )
        if document_date is not None:
            date_source = "filnamn"

    diarienummer: str | None = None
    m = _DIARIENUMMER_RE.search(header)
    if m:
        diarienummer = re.sub(r"\s+", " ", m.group(1).strip())

    return document_date, diarienummer, date_source


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


_DOC_TYPE_RULES: dict | None = None


def _load_document_type_rules() -> dict:
    """
    Läs härledningsreglerna, från instansen om de finns där.

    Samma mönster som synonyms och concepts: repot bär en mall,
    instansen kan ha en egen. Tyst fallback till tom konfiguration —
    utan regler blir document_type null, alltså samma läge som före
    ändringen.
    """
    global _DOC_TYPE_RULES
    if _DOC_TYPE_RULES is not None:
        return _DOC_TYPE_RULES

    import yaml
    for candidate in (Path(".urd") / "document_types.yaml",
                      Path(__file__).parent / "document_types.yaml"):
        if not candidate.exists():
            continue
        try:
            with candidate.open(encoding="utf-8") as f:
                _DOC_TYPE_RULES = yaml.safe_load(f) or {}
            # ingest.py har ingen modulnivå-logger; filen hämtar den
            # där den behövs. Att skriva logger.info här gav
            # NameError vid varje ingest — py_compile fångar inte
            # odefinierade namn.
            logging.getLogger(__name__).info(
                "Laddade %d dokumenttypsregler från %s.",
                len(_DOC_TYPE_RULES.get("rules", [])), candidate,
            )
            return _DOC_TYPE_RULES
        except Exception as e:
            logging.getLogger(__name__).warning(
                "Kunde inte läsa %s (%s).", candidate, e
            )
    _DOC_TYPE_RULES = {}
    return _DOC_TYPE_RULES


def infer_document_type(
    path: Path, docs_root: Path
) -> tuple[str | None, str | None]:
    """
    Härled dokumenttyp och normativ tyngd ur sökvägen.

    Returnerar (type, weight). Deterministiskt och gratis. Detta är
    ENDA källan till document_type sedan enrich togs bort; fältet bär
    normkälle- och aktualitetsreglerna i syntesprompten.

    Okänd sökväg ger (None, None).
    """
    rules = _load_document_type_rules()
    if not rules:
        return None, None

    name = path.name.lower()
    try:
        parts = [p.lower() for p in path.relative_to(docs_root).parts]
    except Exception:
        parts = [p.lower() for p in path.parts]

    for rule in rules.get("rules", []):
        needle = rule.get("match_filename")
        if needle and needle.lower() in name:
            return rule.get("type"), rule.get("weight")

        wanted = rule.get("match_path")
        if wanted and all(
            any(w.lower() == p for p in parts) for w in wanted
        ):
            return rule.get("type"), rule.get("weight")

    for hint in rules.get("filename_hints", []):
        if hint.get("contains", "").lower() in name:
            return hint.get("type"), hint.get("weight")

    return None, None


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
        patterns = [
            r"\bföljande\b",
            r"\bnedanstående\b",
            r"\bovanstående\b",
            r"\bovan\b",
            r"\bnedan\b",
            r"\bstegen\b",
            r"\bpunkterna\b",
            r"\benligt listan\b",
            r"\benligt tabellen\b",
        ]

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


# ---------------------------------------------------------------------------
# Rubrikhierarki ur avsnittsnumrering.
#
# Docling ger inte pålitliga rubriknivåer: i anställningsordningen
# ligger samtliga 188 rubriker på nivå 2, och 158 av dem bär en
# rubrik som är semantiskt icke-unik inom dokumentet ("Behörighet"
# 16 gånger, "Bedömningsgrunder" 16, "Ansökan" 16). En chunk som
# bara märks med närmaste rubrik blir därmed oskiljbar från femton
# andra — cross-encodern kan inte välja rätt kapitel annat än av
# slump, och ett svar om lektor kan byggas på biträdande lektors
# behörighetskrav.
#
# Numreringen bär den hierarki som nivåfältet saknar: 8.5.2 hör
# under 8.5 som hör under 8. Föräldratiteln hämtas i tur och
# ordning ur (1) en sektion med matchande nummer och (2) dokumentets
# egen innehållsförteckning. Steg 2 behövs eftersom extraktionen
# tappar hela kapitelrubriker — kapitel 2, 7, 10 och 14 saknas i
# anställningsordningen, och kapitel 7 är universitetslektor.
# Hellre ingen kedja än en gissad: saknas numret i båda källorna
# utelämnas den nivån.
# ---------------------------------------------------------------------------

_SECTION_NUMBER_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)[.:)]?\s+(\S.*)$")

# Innehållsförteckningsrad: nummer, titel, minst fyra punktledare.
# Punktledarna är signaturen — de förekommer inte i brödtext.
_TOC_ENTRY_RE = re.compile(r"(\d+(?:\.\d+)*)[.:)]?\s+([^|]{3,}?)\s*\.{4,}")


def section_number(title: str | None) -> tuple[int, ...] | None:
    """Numreringen i en rubrik som tupel: '8.5.2 Sakkunnigbedömning' -> (8,5,2)."""
    if not title:
        return None
    m = _SECTION_NUMBER_RE.match(title)
    if not m:
        return None
    return tuple(int(p) for p in m.group(1).split("."))


def build_number_titles(
    sections: list["StructuredSection"],
    full_text: str | None = None,
) -> dict[tuple[int, ...], str]:
    """
    Avbilda avsnittsnummer på rubriktitel för ett dokument.

    Sektionstitlarna är primärkälla. Innehållsförteckningen fyller
    bara luckor — den är mindre tillförlitlig (radbrytningar,
    tabellformatering) och får aldrig skriva över en rubrik som
    faktiskt finns som sektion.
    """
    number_titles: dict[tuple[int, ...], str] = {}

    for section in sections:
        num = section_number(section.title)
        if num and num not in number_titles and section.title:
            number_titles[num] = section.title.strip()

    if full_text:
        for m in _TOC_ENTRY_RE.finditer(full_text):
            num = tuple(int(p) for p in m.group(1).split("."))
            title = m.group(2).strip().rstrip(".").strip()
            if num in number_titles or not title:
                continue
            number_titles[num] = f"{m.group(1)} {title}"

    return number_titles


def section_ancestors(
    title: str | None,
    number_titles: dict[tuple[int, ...], str],
) -> list[str]:
    """Föräldrarubriker till en sektion, från yttersta nivån och inåt."""
    num = section_number(title)
    if not num or len(num) < 2:
        return []
    return [
        number_titles[num[:depth]]
        for depth in range(1, len(num))
        if num[:depth] in number_titles
    ]


def build_chunks_from_sections(
    path: Path,
    document_title: str | None,
    category: str | None,
    sections: list[StructuredSection],
    source_fingerprint: str,
    document_date: str | None = None,
    diarienummer: str | None = None,
    full_text: str | None = None,
    doc_type: str | None = None,
    doc_weight: str | None = None,
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    global_idx = 0

    number_titles = build_number_titles(sections, full_text)

    for section in sections:
        pieces = chunk_text(
            section.text,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )
        ancestors = section_ancestors(section.title, number_titles)
        context_prefix = _build_context_prefix(
            document_title, section.title, ancestors
        )
        section_path = " > ".join(
            [t for t in ancestors if t] + ([section.title] if section.title else [])
        ) or None

        for piece in pieces:
            # Bädda in kontextuellt prefix i den text som indexeras
            contextualized_text = context_prefix + piece

            meta = ChunkMetadata(
                source_path=str(path),
                file_name=path.name,
                document_title=document_title,
                category=category,
                section_title=section.title,
                section_level=section.level,
                section_path=section_path,
                page_number=None,
                document_date=document_date,
                diarienummer=diarienummer,
                # Härledd ur sökvägen vid ingest: deterministiskt och
                # alltid tillgängligt. Bär normkälle- och
                # aktualitetsreglerna i syntesprompten.
                document_type=doc_type,
                document_weight=doc_weight,
                source_fingerprint=source_fingerprint,
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

    document_date, diarienummer, _date_source = extract_document_header_info(
        raw.text, path.name
    )
    document_title = infer_document_title(raw)
    category = infer_category(path, docs_root)
    doc_type, doc_weight = infer_document_type(path, docs_root)
    source_fingerprint = compute_source_fingerprint(path)

    sections = split_markdown_sections(raw.text)

    return build_chunks_from_sections(
        path=path,
        document_title=document_title,
        category=category,
        sections=sections,
        source_fingerprint=source_fingerprint,
        document_date=document_date,
        diarienummer=diarienummer,
        full_text=raw.text,
        doc_type=doc_type,
        doc_weight=doc_weight,
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


def ingest_path_with_evidence(
    path: Path,
    docs_root: Path,
) -> tuple[list[DocumentChunk], list[EvidenceObject], str | None]:
    """
    Samlad ingest: parsar dokumentet en gång och returnerar
    (textchunkar, evidensobjekt, felorsak).

    Felorsaken är None när allt gick bra. Vid tomt resultat skiljer
    den på extraktionsundantag ("<Undantagstyp>: ...") och tyst tom
    text ("extraktionen gav ingen text") — ett dokument som hamnar
    utanför indexet ska aldrig göra det osynligt.

    Effektivare än att köra ingest_path och ingest_evidence_path
    separat, eftersom docling-konverteringen, sektionsindelningen
    och titel-/fingerprint-härledningen bara görs en gång. De äldre
    funktionerna bevaras för bakåtkompatibilitet.
    """
    raw = extract_text_with_fallback(path)
    if not raw.text.strip():
        reason = raw.error or "extraktionen gav ingen text (tomt konverteringsresultat)"
        return [], [], reason

    document_date, diarienummer, _date_source = extract_document_header_info(
        raw.text, path.name
    )

    document_title = infer_document_title(raw)
    category = infer_category(path, docs_root)
    doc_type, doc_weight = infer_document_type(path, docs_root)
    source_fingerprint = compute_source_fingerprint(path)
    sections = split_markdown_sections(raw.text)

    chunks = build_chunks_from_sections(
        path=path,
        document_title=document_title,
        category=category,
        sections=sections,
        source_fingerprint=source_fingerprint,
        document_date=document_date,
        diarienummer=diarienummer,
        full_text=raw.text,
        doc_type=doc_type,
        doc_weight=doc_weight,
    )

    evidence_objects = extract_evidence_objects_from_sections(
        path=path,
        document_title=document_title,
        sections=sections,
        source_fingerprint=source_fingerprint,
    )

    return chunks, evidence_objects, None
