"""
Dokumentinspektion som operationer.

Svarar på vad indexet innehåller för ett dokument: identitet, räknare,
sektionsstruktur, chunkar, evidensobjekt och attestobservationer.

LAGRETS TRE REGLER GÄLLER HÄR (se planeringsdokumentet):

1. En operation skriver aldrig ut något.
2. En renderare räknar aldrig ut något.
3. Frånvaro av träff är ett resultat, inte ett undantag.
4. Ett resultat deklarerar sin nivå, och nivåbyten är namngivna.

Modulen ersätter analysen i scripts/inspect_doc.py, som räknade och
skrev ut i samma funktioner och därför bara kunde konsumeras av
skriptet självt. Samma analys ska kunna bäras av `urd docs inspect`,
av ett punktkommando i interaktivt läge och av en endpoint — utan att
logiken finns i tre exemplar.

BEROENDET ÄR DET SMALASTE SOM RÄCKER. Operationerna tar en
chunk-källa (protokollet ChunkSource nedan), inte RagService. Ingen av
dem behöver embedder, cross-encoder eller LLM, och `stats` har redan
etablerat att indexläsning inte ska kosta en modelladdning. Det gör
dem dessutom testbara med en attrapp, vilket sektionsanalysen aldrig
har varit.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from app.schemas import SourceHit


class ChunkSource(Protocol):
    """
    Det operationerna behöver av lagringen — inte mer.

    QdrantStore uppfyller det. En attrapp med två listor gör det
    också, vilket är hela poängen: analysen ska gå att pröva mot
    konstruerade fall utan att index och modeller startas.
    """

    def iter_all_chunks(self) -> list[SourceHit]: ...
    def iter_all_evidence(self) -> list[SourceHit]: ...


# ---------------------------------------------------------------------------
# Resultattyper
# ---------------------------------------------------------------------------

class DocumentResolution(BaseModel):
    """
    Utfallet av att matcha ett mönster mot indexets dokument.

    Tomt `matched` är ett giltigt resultat, inte ett fel. `total_indexed`
    och `near_misses` finns för att frånvaron ska gå att säga ut: "0 av
    235, men dessa ligger nära" är ett svar, medan tystnad inte är det.
    """
    level: Literal["document"] = "document"
    pattern: str
    matched: list[str] = Field(default_factory=list)
    total_indexed: int = 0
    near_misses: list[str] = Field(default_factory=list)

    @property
    def found(self) -> bool:
        return bool(self.matched)


class DocumentIdentity(BaseModel):
    source_path: str
    file_name: str
    document_title: str | None = None
    category: str | None = None
    document_date: str | None = None
    diarienummer: str | None = None
    document_type: str | None = None
    document_weight: str | None = None
    source_fingerprint: str | None = None


class DocumentCounts(BaseModel):
    sections: int = 0
    chunks: int = 0
    characters: int = 0
    evidence_objects: int = 0
    # None betyder att Attest inte konsulterats, 0 att det konsulterats
    # och gav noll. Skillnaden är inte kosmetisk: den senare är ett fynd
    # om uttaget, den förra ingenting alls.
    attest_observations: int | None = None


class SectionView(BaseModel):
    title: str | None = None
    level: int | None = None
    chunks: int = 0
    characters: int = 0
    first_index: int = 0
    # Rubrikkedjan som den ser ut med nivåstack (närmaste förälder) och
    # som den ser ut med avsnittsnumrering. De skiljer sig när Docling
    # tappat nivåer, vilket är det vanliga i beståndet.
    chain: list[str] = Field(default_factory=list)
    numbered_chain: list[str] = Field(default_factory=list)


class HeadingCollision(BaseModel):
    """En rubriktext som förekommer i flera sektioner inom dokumentet."""
    key: str
    sections: list[SectionView] = Field(default_factory=list)
    resolved_by_numbering: int = 0


class SectionReport(BaseModel):
    level: Literal["section"] = "section"
    sections: list[SectionView] = Field(default_factory=list)
    without_level: int = 0
    collisions: list[HeadingCollision] = Field(default_factory=list)
    sections_in_collision: int = 0
    resolved_by_numbering: int = 0
    orphans: list[str] = Field(default_factory=list)


class ChunkView(BaseModel):
    level: Literal["chunk"] = "chunk"
    chunk_index: int
    section_title: str | None = None
    characters: int = 0
    text: str = ""


class EvidenceView(BaseModel):
    level: Literal["evidence"] = "evidence"
    section_title: str | None = None
    evidence_type: str | None = None
    text: str = ""


class AttestObservationView(BaseModel):
    level: Literal["observation"] = "observation"
    subject: str
    object: str
    kind: str
    construction: str
    status: str | None = None
    scope: str | None = None
    ambiguous: bool = False
    sentence: str = ""


class DocumentAttestView(BaseModel):
    observations: list[AttestObservationView] = Field(default_factory=list)
    total: int = 0
    by_kind: dict[str, int] = Field(default_factory=dict)
    available: bool = True


class DocumentReport(BaseModel):
    level: Literal["document"] = "document"
    identity: DocumentIdentity
    counts: DocumentCounts
    sections: SectionReport | None = None
    chunks: list[ChunkView] | None = None
    evidence: list[EvidenceView] | None = None
    attest: DocumentAttestView | None = None


# ---------------------------------------------------------------------------
# Mönsterupplösning
# ---------------------------------------------------------------------------

def _norm(s: str) -> str:
    """
    NFC-normalisera och casefolda.

    macOS och vissa verktyg lagrar 'ö' dekomponerat, vilket annars ger
    tysta mismatchar mellan mönster och filnamn.
    """
    return unicodedata.normalize("NFC", s).casefold()


def _near_misses(pattern: str, source_paths: list[str], limit: int = 8) -> list[str]:
    """
    Dokument som delar minst ett ordfragment med mönstret.

    Grovt med flit: syftet är att göra ett nollresultat begripligt,
    inte att gissa vad användaren menade.
    """
    tokens = [t for t in re.split(r"[\s_\-.]+", _norm(pattern)) if len(t) >= 4]
    if not tokens:
        return sorted(source_paths)[:limit]
    scored = []
    for path in source_paths:
        norm = _norm(path)
        hits = sum(1 for t in tokens if t in norm)
        if hits:
            scored.append((hits, path))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [p for _, p in scored[:limit]]


def resolve_document(pattern: str, source_paths: list[str]) -> DocumentResolution:
    """
    Matcha ett mönster mot indexets dokumentsökvägar.

    Flera träffar är ett giltigt utfall — anroparen avgör om den vill
    visa alla eller be om precisering. Operationen väljer aldrig åt
    användaren.
    """
    norm_pattern = _norm(pattern)
    matched = sorted(p for p in source_paths if norm_pattern in _norm(p))
    return DocumentResolution(
        pattern=pattern,
        matched=matched,
        total_indexed=len(source_paths),
        near_misses=[] if matched else _near_misses(pattern, source_paths),
    )


# ---------------------------------------------------------------------------
# Sektionsanalys
# ---------------------------------------------------------------------------

_NUMBERING_RE = re.compile(r"^\s*\d+(?:[.:]\d+)*[.:)]?\s+")


def _strip_numbering(title: str) -> str:
    """
    Ta bort ledande avsnittsnumrering ("7.3.4 ", "12 ", "3) ").

    Numreringen är en ordningsmarkör, inte innehåll. För embedding och
    cross-encoder är "7.2 Behörighet" och "8.2 Behörighet" praktiskt
    taget samma sträng — det är den skillnaden analysen ska fånga.
    """
    return _NUMBERING_RE.sub("", title).strip()


def _sections_in_order(chunks: list[SourceHit]) -> list[SectionView]:
    """
    Rekonstruera dokumentets sektioner ur de indexerade chunkarna.

    chunk_index är löpande i dokumentordning, så chunkar i samma
    sektion ligger sammanhängande. Qdrants scroll-ordning är däremot
    inte garanterad, därför sorteras det explicit.
    """
    ordered = sorted(chunks, key=lambda c: c.metadata.chunk_index)
    sections: list[SectionView] = []
    for c in ordered:
        title = c.metadata.section_title
        level = c.metadata.section_level
        if sections and sections[-1].title == title and sections[-1].level == level:
            sections[-1].chunks += 1
            sections[-1].characters += len(c.text)
        else:
            sections.append(SectionView(
                title=title, level=level, chunks=1,
                characters=len(c.text), first_index=c.metadata.chunk_index,
            ))
    return sections


def _attach_chains(sections: list[SectionView]) -> None:
    """
    Bygg rubrikkedja per sektion med en nivåstack, in-place.

    Sektioner utan nivå — text före första rubriken, eller dokument där
    Docling inte hittat rubrikstruktur — bär ingen hierarkisk
    information och får inte störa stacken.
    """
    stack: list[tuple[int, str]] = []
    for s in sections:
        if s.level is None:
            s.chain = [s.title] if s.title else []
            continue
        while stack and stack[-1][0] >= s.level:
            stack.pop()
        s.chain = [t for _, t in stack] + ([s.title] if s.title else [])
        if s.title:
            stack.append((s.level, s.title))


def _attach_numbered_chains(
    sections: list[SectionView], full_text: str | None
) -> None:
    """
    Bygg rubrikkedja ur avsnittsnumreringen, som vid ingest.

    Importen är lat med flit: app.ingest instansierar en Docling-
    konverterare på modulnivå, och den kostnaden hör inte till en
    indexläsning.
    """
    from app.ingest import build_number_titles, section_ancestors

    class _S:
        def __init__(self, title): self.title = title

    number_titles = build_number_titles(
        [_S(s.title) for s in sections], full_text
    )
    for s in sections:
        ancestors = section_ancestors(s.title, number_titles)
        s.numbered_chain = ancestors + ([s.title] if s.title else [])


def analyze_sections(
    chunks: list[SourceHit], full_text: str | None = None
) -> SectionReport:
    """
    Rubrikstruktur och kollisioner inom ett dokument.

    KOLLISIONERNA ÄR DEN AVGÖRANDE SIFFRAN. En chunk indexeras med
    dokumenttitel plus rubrikkedja; två sektioner med samma rubriktext
    blir oskiljbara för embedding, BM25 och cross-encoder om kedjan
    inte skiljer dem åt. I anställningsordningen förekommer
    "Behörighet" sexton gånger. Rapporten visar hur många av
    kollisionerna som faktiskt upplöses av den numreringsbaserade
    kedjan, alltså hur mycket av problemet som redan är åtgärdat.
    """
    sections = _sections_in_order(chunks)
    _attach_chains(sections)
    _attach_numbered_chains(sections, full_text)

    by_key: dict[str, list[SectionView]] = {}
    for s in sections:
        if not s.title:
            continue
        by_key.setdefault(_strip_numbering(s.title).casefold(), []).append(s)

    collisions: list[HeadingCollision] = []
    in_collision = 0
    resolved = 0
    for key, group in sorted(by_key.items()):
        if len(group) < 2:
            continue
        distinct = {" > ".join(s.numbered_chain).casefold() for s in group}
        n_resolved = len(distinct) if len(distinct) > 1 else 0
        collisions.append(HeadingCollision(
            key=key, sections=group, resolved_by_numbering=n_resolved,
        ))
        in_collision += len(group)
        resolved += n_resolved

    orphans = [
        s.title for c in collisions for s in c.sections
        if s.title and len(s.numbered_chain) <= 1
    ]

    return SectionReport(
        sections=sections,
        without_level=sum(1 for s in sections if s.level is None),
        collisions=collisions,
        sections_in_collision=in_collision,
        resolved_by_numbering=resolved,
        orphans=orphans,
    )


# ---------------------------------------------------------------------------
# Inspektion
# ---------------------------------------------------------------------------

def _identity(chunks: list[SourceHit]) -> DocumentIdentity:
    m = sorted(chunks, key=lambda c: c.metadata.chunk_index)[0].metadata
    return DocumentIdentity(
        source_path=m.source_path,
        file_name=m.file_name,
        document_title=m.document_title,
        category=m.category,
        document_date=m.document_date,
        diarienummer=m.diarienummer,
        document_type=m.document_type,
        document_weight=m.document_weight,
        source_fingerprint=m.source_fingerprint,
    )


def inspect_document(
    source_path: str,
    store: ChunkSource,
    *,
    sections: bool = False,
    chunks: bool = False,
    evidence: bool = False,
    attest: bool = False,
    conn=None,
    all_chunks: list[SourceHit] | None = None,
) -> DocumentReport | None:
    """
    Vad indexet innehåller för ett dokument.

    Grundnivån — identitet och räknare — kostar ingenting och lämnas
    alltid. Fördjupningarna begärs uttryckligen, så en anropare som
    bara visar en översikt inte betalar för resten.

    `all_chunks` är en cache för den som redan har hela chunkmängden
    (interaktivt läge håller den i BM25-indexet). Utan den läses den
    ur lagringen.

    conn är en öppen attest-anslutning; None öppnar en egen vid behov.
    Samma mönster som corpus_guard.check_answer.

    Returnerar None när dokumentet inte finns i indexet — anroparen har
    då redan fått veta det av resolve_document.
    """
    pool = all_chunks if all_chunks is not None else store.iter_all_chunks()
    doc_chunks = [c for c in pool if c.metadata.source_path == source_path]
    if not doc_chunks:
        return None
    doc_chunks.sort(key=lambda c: c.metadata.chunk_index)

    evidence_hits = [
        e for e in store.iter_all_evidence()
        if e.metadata.source_path == source_path
    ] if evidence else []
    n_evidence = len(evidence_hits) if evidence else sum(
        1 for e in store.iter_all_evidence()
        if e.metadata.source_path == source_path
    )

    attest_view: DocumentAttestView | None = None
    n_obs: int | None = None
    if attest:
        attest_view = _attest_for_document(source_path, conn)
        n_obs = attest_view.total if attest_view.available else None

    report = DocumentReport(
        identity=_identity(doc_chunks),
        counts=DocumentCounts(
            sections=len(_sections_in_order(doc_chunks)),
            chunks=len(doc_chunks),
            characters=sum(len(c.text) for c in doc_chunks),
            evidence_objects=n_evidence,
            attest_observations=n_obs,
        ),
        attest=attest_view,
    )

    if sections:
        full_text = "\n".join(c.text for c in doc_chunks)
        report.sections = analyze_sections(doc_chunks, full_text)

    if chunks:
        report.chunks = [
            ChunkView(
                chunk_index=c.metadata.chunk_index,
                section_title=c.metadata.section_title,
                characters=len(c.text),
                text=c.text,
            )
            for c in doc_chunks
        ]

    if evidence:
        report.evidence = [
            EvidenceView(
                section_title=e.metadata.section_title,
                evidence_type=e.metadata.document_type,
                text=e.text,
            )
            for e in evidence_hits
        ]

    return report


def _attest_for_document(source_path: str, conn=None) -> DocumentAttestView:
    """
    Observationer ur ETT dokument.

    Uppslagen i attest.py går från termens håll — vem binds till rollen,
    vilka roller binds till personen. Ingen av dem svarar på vad ett
    enskilt dokument gav, trots att source_path är indexerad. Den vyn
    behövs på tre ställen: när korpuskontrollen binder fel och man vill
    se vad källan faktiskt gav, när täckningen ska bedömas per
    dokumenttyp, och när en ny konstruktion ska granskas mot en känd
    texttyp i stället för mot ett slumpat stickprov.
    """
    try:
        from app import attest
        rows = attest.observations_for_document(conn, source_path)
    except Exception:
        return DocumentAttestView(available=False)

    by_kind: dict[str, int] = {}
    views: list[AttestObservationView] = []
    for r in rows:
        by_kind[r["kind"]] = by_kind.get(r["kind"], 0) + 1
        views.append(AttestObservationView(
            subject=r["subject"], object=r["object"], kind=r["kind"],
            construction=r["construction"], status=r["status"],
            scope=r["scope"], ambiguous=bool(r["ambiguous"]),
            sentence=r["sentence"] or "",
        ))
    return DocumentAttestView(
        observations=views, total=len(views), by_kind=by_kind,
    )


# ---------------------------------------------------------------------------
# Rendering
#
# Renderarna räknar aldrig ut något. De ligger i samma modul som
# operationerna eftersom de ändras tillsammans; CLI och interaktivt läge
# delar dem, så att samma rapport inte formateras i två exemplar som
# glider isär vid första ändringen.
# ---------------------------------------------------------------------------

def format_resolution(res: DocumentResolution) -> str:
    if res.found:
        if len(res.matched) == 1:
            return f"{res.matched[0]}"
        lines = [f"{len(res.matched)} dokument matchar {res.pattern!r}:"]
        lines += [f"  {p}" for p in res.matched]
        return "\n".join(lines)

    lines = [
        f"Inget av {res.total_indexed} indexerade dokument matchar "
        f"{res.pattern!r}."
    ]
    if res.near_misses:
        lines.append("Närliggande kandidater:")
        lines += [f"  {p}" for p in res.near_misses]
    return "\n".join(lines)


class TermCoverage(BaseModel):
    """
    Avståndet mellan hur ofta en term FÖREKOMMER i beståndet och hur
    ofta uttaget fått ut en observation ur den. Nivå: bestånd.
    """
    level: Literal["corpus"] = "corpus"
    term: str
    text_occurrences: int
    documents_with_text: int
    observations: int
    documents_with_observations: int
    per_kind: dict[str, int]
    # Dokument där termen står i texten men inget uttag skett. Detta är
    # den handlingsbara listan — allt annat är sammanfattning.
    documents_without_observations: list[str]


def term_coverage(
    term: str,
    store: ChunkSource,
    conn=None,
) -> TermCoverage:
    """
    Mät uttagets täckning för en term.

    VARFÖR DETTA BEHÖVS. Beläggningsmodellen räknar hur ofta något
    skrivits, men säger ingenting om hur mycket som skrevs och inte
    fångades. Uppmätt 2026-08-18: "studierektor" står i 222
    textförekomster i 60 dokument, medan Attest har 16
    identitetsobservationer i 13. Utan det måttet gick det inte att
    skilja "beståndet säger inget om saken" från "uttaget missade
    det", och det är två helt olika fel med två helt olika åtgärder.

    Ett stort avstånd är INTE i sig ett fel. Delegations- och
    handläggningsordningar nämner roller generiskt — "studierektor
    beslutar om…" — vilket är agens, inte identitet. Måttet pekar ut
    var man ska titta, det dömer inte.

    Termen matchas som ordprefix, så att böjningar och bestämd form
    räknas med. Sammansättningar där termen är efterled
    (forskarstudierektor) räknas inte, av samma skäl som uppslaget
    kräver avslutande ordföljd: de är andra ord.
    """
    pattern = re.compile(rf"\b{re.escape(term)}\w*", re.IGNORECASE)

    text_occurrences = 0
    docs_with_text: set[str] = set()
    for chunk in store.iter_all_chunks():
        found = len(pattern.findall(chunk.text))
        if found:
            text_occurrences += found
            docs_with_text.add(chunk.metadata.source_path)

    observations = 0
    docs_with_obs: set[str] = set()
    per_kind: dict[str, int] = {}
    own_conn = conn is None
    try:
        if own_conn:
            from app import attest
            conn = attest.connect()
        # Positionsindex, inte namn: anropare kan skicka en anslutning
        # utan sqlite3.Row som radfabrik. Attest sätter den per
        # uppslagsfunktion, inte på anslutningen.
        rows = conn.execute(
            "SELECT kind, source_path FROM observations"
            " WHERE subject_key LIKE ? OR object_key LIKE ?",
            (f"%{term.lower()}%", f"%{term.lower()}%"),
        ).fetchall()
        for kind, source_path in rows:
            observations += 1
            docs_with_obs.add(source_path)
            per_kind[kind] = per_kind.get(kind, 0) + 1
    finally:
        if own_conn and conn is not None:
            conn.close()

    return TermCoverage(
        term=term,
        text_occurrences=text_occurrences,
        documents_with_text=len(docs_with_text),
        observations=observations,
        documents_with_observations=len(docs_with_obs),
        per_kind=dict(sorted(per_kind.items(), key=lambda x: -x[1])),
        documents_without_observations=sorted(docs_with_text - docs_with_obs),
    )


def format_term_coverage(cov: TermCoverage) -> str:
    lines = [
        f"Täckning för {cov.term!r}",
        "",
        f"  I texten:        {cov.text_occurrences} förekomster"
        f" i {cov.documents_with_text} dokument",
        f"  I Attest:        {cov.observations} observationer"
        f" i {cov.documents_with_observations} dokument",
    ]
    if cov.per_kind:
        delar = ", ".join(f"{k} {n}" for k, n in cov.per_kind.items())
        lines.append(f"  Per dragtyp:     {delar}")
    saknas = len(cov.documents_without_observations)
    lines.append(f"  Utan uttag:      {saknas} dokument")
    for path in cov.documents_without_observations[:15]:
        lines.append(f"      {path.rsplit('/', 1)[-1]}")
    if saknas > 15:
        lines.append(f"      ... och {saknas - 15} till")
    lines += [
        "",
        "  Ett stort avstånd är inte i sig ett fel: generiska omnämnanden",
        "  ('studierektor beslutar om...') är agens, inte identitet.",
        "  Måttet pekar ut var man ska titta.",
    ]
    return "\n".join(lines)


class StageResult(BaseModel):
    """Ett dokuments utfall i ett steg av kedjan. Nivå: dokument."""
    level: Literal["document"] = "document"
    reached: bool
    rank: int | None = None
    score: float | None = None
    pool_size: int = 0
    top_competitor: str | None = None


class ChunkScore(BaseModel):
    """Cross-encoderns bedömning av en chunk. Nivå: chunk."""
    level: Literal["chunk"] = "chunk"
    section_title: str | None = None
    chunk_index: int | None = None
    probability: float
    filtered: bool


class RetrievalTrace(BaseModel):
    """
    Var i kedjan ett dokument föll bort. Nivå: dokument.

    fell_out_at ÄR DATA, INTE PROSA. Ett dokument som saknas i både
    semantisk pool och BM25 har ett kandidatinsamlingsfel — språkgap
    eller embeddingkvalitet. Ett som kommer in men filtreras bort har
    ett rerankingfel. Olika fel, olika åtgärd, och anroparen ska inte
    behöva härleda skillnaden ur en utskrift.
    """
    level: Literal["document"] = "document"
    question: str
    source_path: str
    semantic: StageResult
    bm25: StageResult
    reranking: list[ChunkScore]
    required_passage: bool = False
    fell_out_at: Literal[
        "candidate_collection", "reranking", "selection", "passed"
    ]


def _stage(hits: list[SourceHit], source_path: str) -> StageResult:
    for rank, hit in enumerate(hits, start=1):
        if hit.metadata.source_path == source_path:
            return StageResult(
                reached=True, rank=rank, score=hit.score, pool_size=len(hits),
            )
    return StageResult(
        reached=False,
        pool_size=len(hits),
        top_competitor=hits[0].metadata.file_name if hits else None,
    )


def trace_retrieval(source_path: str, question: str, rag) -> RetrievalTrace:
    """
    Spåra ett dokument genom retrievalkedjan.

    DEN IAKTTAR, DEN KÖR INTE OM. Anropet går genom
    RagService.collect_and_rank — exakt den kod som besvarar frågor.
    En diagnos som reproducerar kedjan för hand glider från den, och
    ljuger då mest i de fall den finns till för: skriptets tidigare
    kopia saknade broader-expansionen, operationstermerna, den ankrade
    attestpoolen och dokumentexpansionens andra pass.

    Detta är den enda operationen som tar RagService, och inte som ett
    beroendeval: den ÄR retrievalkedjan.

    Frågeoperationen bestäms av regellagret, samma väg som answer()
    använder när LLM-klassificeraren inte hunnit säga sitt. Utan den
    skulle attestsignalen aldrig slå till i spårningen och diagnosen
    visa en annan kedja än den verkliga.
    """
    from app.question_rules import rule_based_operation

    operation = rule_based_operation(question) or "direct_lookup"
    pool = rag.collect_and_rank(
        question=question,
        search_text=question,
        rerank_text=question,
        question_operation=operation,
    )

    semantic = _stage(pool.semantic_hits, source_path)
    bm25 = _stage(pool.bm25_hits, source_path)

    scored = [
        ChunkScore(
            section_title=d.get("section_title"),
            chunk_index=d.get("chunk_index"),
            probability=d.get("relevance_prob", 0.0),
            filtered=bool(d.get("filtered")),
        )
        for d in pool.rerank_debug
        if d.get("source_path") == source_path
    ]
    scored.sort(key=lambda c: c.probability, reverse=True)

    survived = any(
        h.metadata.source_path == source_path for h in pool.reranked
    )
    required = any(loc[0] == source_path for loc in pool.attest_locations)

    if survived:
        fell_out_at = "passed"
    elif scored:
        fell_out_at = "reranking"
    elif semantic.reached or bm25.reached:
        fell_out_at = "selection"
    else:
        fell_out_at = "candidate_collection"

    return RetrievalTrace(
        question=question,
        source_path=source_path,
        semantic=semantic,
        bm25=bm25,
        reranking=scored[:12],
        required_passage=required,
        fell_out_at=fell_out_at,
    )


_FELL_OUT_TEXT = {
    "candidate_collection":
        "Dokumentet kommer inte in i kandidatpoolen. Felet sitter i\n"
        "  kandidatinsamlingen (embedding/BM25), inte i rerankern.",
    "reranking":
        "Dokumentet når rerankern men filtreras bort av golvet.\n"
        "  Cross-encodern bedömer det som irrelevant för frågan.",
    "selection":
        "Dokumentet når poolen men ingen av dess chunkar bedömdes.",
    "passed":
        "Dokumentet passerar hela kedjan.",
}


def format_retrieval_trace(trace: RetrievalTrace) -> str:
    def stage(namn: str, r: StageResult) -> str:
        if r.reached:
            return (f"  {namn:<10} plats {r.rank} av {r.pool_size}"
                    f"  score {r.score:.4f}")
        top = f"  (topp: {r.top_competitor})" if r.top_competitor else ""
        return f"  {namn:<10} ej i poolen ({r.pool_size} kandidater){top}"

    lines = [
        f"Retrievaldiagnos: {trace.source_path.rsplit('/', 1)[-1]}",
        f"Fråga: {trace.question!r}",
        "",
        stage("semantisk", trace.semantic),
        stage("BM25", trace.bm25),
    ]
    if trace.required_passage:
        lines.append("  attest     dokumentet bär en reserverad passage")
    if trace.reranking:
        lines.append("")
        lines.append("  cross-encoder (sannolikhet, ✗ = under golvet):")
        for c in trace.reranking:
            mark = "✗" if c.filtered else "✓"
            lines.append(f"    {mark} {c.probability:.4f}  {c.section_title}")
    lines += ["", "  " + _FELL_OUT_TEXT[trace.fell_out_at]]
    return "\n".join(lines)


def format_document_report(report: DocumentReport) -> str:
    i, c = report.identity, report.counts
    lines = [
        "=" * 72,
        f"DOKUMENT: {i.source_path}",
    ]
    if i.document_title:
        lines.append(f"  {'titel:':<15}{i.document_title}")
    for label, value in (
        ("kategori", i.category), ("datum", i.document_date),
        ("diarienummer", i.diarienummer), ("dokumenttyp", i.document_type),
        ("normativ vikt", i.document_weight),
    ):
        if value:
            lines.append(f"  {label + ':':<15}{value}")
    lines.append(
        f"  {'innehåll:':<15}{c.sections} sektioner, {c.chunks} chunkar, "
        f"{c.characters} tecken, {c.evidence_objects} evidensobjekt"
    )
    if c.attest_observations is not None:
        lines.append(f"  {'attest:':<15}{c.attest_observations} observationer")

    if report.sections:
        lines.append("")
        lines.append(_format_sections(report.sections))
    if report.attest:
        lines.append("")
        lines.append(_format_attest(report.attest))
    if report.evidence is not None:
        lines.append("")
        lines.append(f"EVIDENSOBJEKT ({len(report.evidence)})")
        for e in report.evidence:
            lines.append(f"--- [{e.section_title}] {e.evidence_type}")
            lines.append(e.text)
    if report.chunks is not None:
        lines.append("")
        lines.append(f"CHUNKAR ({len(report.chunks)})")
        for ch in report.chunks:
            lines.append(
                f"--- chunk {ch.chunk_index}  [{ch.section_title}]  "
                f"{ch.characters} tecken"
            )
            lines.append(ch.text)
    return "\n".join(lines)


def _format_sections(rep: SectionReport) -> str:
    lines = [f"STRUKTUR: {len(rep.sections)} sektioner"]
    if rep.without_level:
        lines.append(
            f"  Varning: {rep.without_level} sektioner utan nivå — "
            "rubrikstrukturen är ofullständig."
        )
    for s in rep.sections:
        indent = "  " * (s.level or 1)
        title = s.title or "(text utan rubrik)"
        lines.append(f"  {indent}{title}   [{s.chunks} chunkar]")

    lines.append("")
    lines.append("SEMANTISKT ICKE-UNIKA RUBRIKER (numrering bortstädad):")
    if not rep.collisions:
        lines.append(
            "  Inga — varje rubrik är särskiljbar även utan sin numrering."
        )
        return "\n".join(lines)

    for col in rep.collisions:
        lines.append(f"  {col.key!r} förekommer {len(col.sections)} gånger:")
        for s in col.sections:
            chain = " > ".join(s.numbered_chain) or "(ingen kedja)"
            lines.append(f"     {chain}   [{s.chunks} chunkar]")
    lines.append("")
    lines.append(
        f"  {rep.sections_in_collision} sektioner bär en semantiskt "
        f"icke-unik rubrik; {rep.resolved_by_numbering} blir särskiljbara "
        "med numreringsbaserad rubrikkedja."
    )
    if rep.orphans:
        lines.append(
            f"  {len(rep.orphans)} av dem saknar föräldrarubrik helt."
        )
    return "\n".join(lines)


def _format_attest(view: DocumentAttestView) -> str:
    if not view.available:
        return "ATTEST: indexet är otillgängligt (kör 'urd attest-build')."
    if not view.observations:
        return (
            "ATTEST: 0 observationer ur detta dokument.\n"
            "  Dokumentet bidrar inte till entitetsuppslag, korpuskontroll "
            "eller uppräkning."
        )
    kinds = ", ".join(f"{k} {n}" for k, n in sorted(view.by_kind.items()))
    lines = [f"ATTEST: {view.total} observationer ({kinds})"]
    for o in view.observations:
        flags = []
        if o.ambiguous:
            flags.append("TVETYDIG")
        if o.status:
            flags.append(o.status)
        if o.scope:
            flags.append(f"för {o.scope}")
        suffix = f"  [{', '.join(flags)}]" if flags else ""
        lines.append(f"  [{o.construction}] {o.subject} -> {o.object}{suffix}")
        if o.sentence:
            lines.append(f"     {o.sentence[:120]}")
    return "\n".join(lines)
