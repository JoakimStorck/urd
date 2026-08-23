"""
Attest — korpusbaserad beläggning av påståenden.

Modulen samlar grammatiska observationer ur hela dokumentbeståndet och
svarar på hur väl ett påstående är BELAGT i korpus. Den svarar inte på
om påståendet är sant.

    Attestering != sanning.

    Attest räknar observationer av grammatiska relationer. En felaktig
    uppgift som upprepas i tio protokoll får hög beläggning. Utdata
    heter därför `documents` och `observations`, aldrig `confidence`
    eller `probability` — siffrorna mäter korpus, inte verkligheten.

VARFÖR DETTA INTE ÄR CLAIMSLAGRET

Claimslagret lagrade en LLM:s TOLKNING av vad texten betydde: en
omformulering, gjord i förväg, som blev systemets sanning. Attest
lagrar OBSERVATIONER: den här grammatiska relationen förekommer i den
här meningen i det här dokumentet. Deterministiskt utvunnet, spårbart
till en läsbar mening, och reviderbart genom omräkning i stället för
omextraktion.

Regeln från white paper gäller: aggregatet får peka ut var svaret
finns, men originaltexten ska bära det. Attest rangordnar och flaggar
— den formulerar aldrig.

TIDSMODELLEN

Allt blir historia så snart något nytt hänt. En bindning är inte ett
tillstånd utan ett INTERVALL: företrädaren var innehavare fram till en
tidpunkt, och den nuvarande skiljer sig bara genom att slutpunkten
ännu inte observerats. Ingenting invalideras, ingenting raderas —
"vem är X" och "vem var X 2023" är samma fråga med olika tidpunkt.

Ett öppet intervall är inte ett påstående om evig giltighet, utan
frånvaron av en observerad slutpunkt. Därför redovisas alltid hur
färskt det senaste belägget är.

UNIKHET MÄTS, ANTAS INTE

Uppdrag (prefekt, proprefekt, studierektor) har en innehavare i taget;
titlar (professor, universitetslektor, HR-specialist) bärs av många
samtidigt. Skillnaden avgör om ett nytt belägg SLUTER ett föregående
intervall eller bara läggs till. Attest antar inte vilket som är
vilket: bärs rollen av flera personer överlappande i tid är den
icke-unik, och det faller ut ur datat.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

from app.config import settings
from app.grammar import extract_features, is_available
from app.morphology import is_inflection_of

logger = logging.getLogger(__name__)

DB_PATH = Path(".urd") / "attest.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS observations (
    id           INTEGER PRIMARY KEY,
    subject      TEXT NOT NULL,
    subject_key  TEXT NOT NULL,
    relation     TEXT NOT NULL,
    object       TEXT NOT NULL,
    object_key   TEXT NOT NULL,
    kind         TEXT NOT NULL,
    construction TEXT NOT NULL,
    ambiguous    INTEGER NOT NULL DEFAULT 0,
    -- Bindningens status ur tillsättningsverbet: tillsatt, föreslagen,
    -- förlängd, avslutad. NULL för appositioner, som inte uttrycker
    -- någon. Ett förslag är inte en tillsättning — beståndet är fullt
    -- av förslag som bifalls i nästa punkt, eller inte.
    status       TEXT,
    -- Rollens avgränsning: "studierektor FÖR mikrodataanalys". Inte en
    -- egen roll — samma roll, olika uppdrag. NULL när texten inte
    -- anger någon, vilket är en fullgod observation och inte en
    -- ofullständig. Systemet fyller aldrig i luckan.
    scope        TEXT,
    strength     TEXT,
    sentence     TEXT,
    source_path  TEXT NOT NULL,
    file_name    TEXT,
    category     TEXT,
    document_date TEXT,
    fingerprint  TEXT,
    -- Dokumentets INNEHÅLL, som sha256 över dess chunktexter. Skilt
    -- från fingerprint, som är en ändringsstämpel över sökväg,
    -- storlek och mtime och därför skiljer två identiska kopior åt.
    -- Uppmätt 2026-08-22: 23 dokument i beståndet ligger under flera
    -- sökvägar, fyra av dem under tre. Utan innehållsidentitet räknas
    -- de som skilda belägg.
    content_hash TEXT,
    chunk_index  INTEGER
);
CREATE INDEX IF NOT EXISTS ix_subject ON observations(subject_key);
CREATE INDEX IF NOT EXISTS ix_object  ON observations(object_key);
CREATE INDEX IF NOT EXISTS ix_source  ON observations(source_path);

-- Per dokument, för inkrementell uppdatering. Ett dokuments bidrag
-- kan raderas och ersättas utan att något annat räknas om — därför
-- lagras observationer och inte färdiga frekvenser.
CREATE TABLE IF NOT EXISTS documents (
    source_path  TEXT PRIMARY KEY,
    fingerprint  TEXT,
    content_hash TEXT,
    document_date TEXT,
    category     TEXT,
    num_obs      INTEGER,
    built_at     TEXT
);
"""


def _key(text: str) -> str:
    """
    Uppslagsnyckel för aggregering.

    NYCKELN NORMALISERAR HUR ETT ORD SKRIVS, ALDRIG VILKET ORD DET ÄR.

    Ortografi tas bort: gemener, bindestreck, punkter, skiljetecken.
    Därmed samlas "pro-prefekt"/"proprefekt",
    "HR-specialist"/"hR-specialist" och "T.f."/"tf" under samma nyckel,
    liksom stavningsvarianter av personnamn i olika protokoll.

    Innehåll bevaras: mellanslag tas INTE bort, så "biträdande lektor"
    förblir skild från "lektor" — det är hela kapitelskillnaden i
    anställningsordningen. Bestämningsord som tillförordnad och
    biträdande ändrar rollen och får aldrig normaliseras bort.

    Förkortningar kopplas inte till sina långformer: "bitr-lektor" och
    "biträdande lektor" blir skilda nycklar. Att koppla ihop dem kräver
    en ordlista över förkortningar, och sådana växer utan gräns. Om
    beståndet självt definierar förkortningen — "biträdande lektor
    (bitr-lektor)" — fångas kopplingen av parentesuttaget som en
    forkortning-relation, ur beståndets egen praxis.

    `subject`/`object` bevarar alltid källans ordalydelse; nyckeln
    används bara för att gruppera och visas aldrig.
    """
    t = text.lower().replace("-", "").replace(".", "")
    t = re.sub(r"[^\wÅÄÖåäö\s]", " ", t)
    return " ".join(t.split())


def connect(path: Path | None = None) -> sqlite3.Connection:
    p = path or DB_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p))
    conn.executescript(SCHEMA)
    _migrate(conn)
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """
    Lägg till kolumner som saknas i äldre databaser.

    Observationerna är oföränderliga och byggs om med attest-build, så
    en saknad kolumn är inte datakritisk — men att krascha på ett
    index byggt före en schemaändring vore onödigt. Nya kolumner blir
    NULL tills indexet byggs om.
    """
    have = {r[1] for r in conn.execute("PRAGMA table_info(observations)")}
    for column, ddl in (
        ("status", "TEXT"), ("scope", "TEXT"), ("content_hash", "TEXT"),
    ):
        if column not in have:
            conn.execute(f"ALTER TABLE observations ADD COLUMN {column} {ddl}")
            logger.info(
                "attest: la till kolumn %r — bygg om med 'urd attest-build' "
                "för att fylla den.", column
            )
    # content_hash fylls i efterhand av build(), utan omtolkning av
    # dokumenten: hashen räknas ur chunktexterna som redan finns i
    # indexet. En befintlig databas behöver alltså inte parsas om.
    have_doc = {r[1] for r in conn.execute("PRAGMA table_info(documents)")}
    if "content_hash" not in have_doc:
        conn.execute("ALTER TABLE documents ADD COLUMN content_hash TEXT")
    conn.commit()


# ---------------------------------------------------------------------------
# Byggning
# ---------------------------------------------------------------------------

# Åtskillnadsdrag lagras INTE. De är kvadratiska i antalet
# verbargument — 1 922 av 4 560 rader i ett urval om 20 dokument — och
# de säger ingenting om beläggning, bara om att två led är olika. De
# behövs enbart som motsägelse vid klassificering av ett enskilt svar,
# och där finns de redan via predication.py. Att utesluta dem halverar
# tabellen utan att förlora något Attest använder.
# forkortning och tillhorighet lagras men är INTE identiteter:
# "Erik Nilsson -> HDa" är en arbetsplats, "Utvärderingsutskotten ->
# UUU" en termdefinition. Båda är korrekta observationer och värdefulla
# — förkortningarna är beståndets egen ordlista — men en rollfråga får
# aldrig besvaras med dem.
# patiens är passiva satser utan utsatt agent ("beslut ska
# diarieföras"): subjektet är det som handlingen drabbar, inte den som
# handlar. Lagras separat så att ett uppslag på agens aldrig kan
# returnera ett drag som betyder motsatsen.
_STORED_KINDS = {
    "identitet", "agens", "patiens", "modalitet",
    "forkortning", "tillhorighet",
    # Termekvivalens: två namn på samma sak, inte en person i en roll.
    # Skild från identitet så att ett uppslag på "vem är X" aldrig kan
    # returnera beståndets tvåspråkiga ordlista som personkandidat.
    # Ordlistan är i sig värdefull — white paperns tredje synonymväg —
    # men den hör till termhanteringen, inte till personfrågor.
    "oversattning",
}


def _observations_from_chunk(chunk) -> list[dict]:
    md = chunk.metadata
    rows: list[dict] = []
    for f in extract_features(chunk.text):
        if f.b is None or f.kind not in _STORED_KINDS:
            continue
        rows.append({
            "subject": f.a, "subject_key": _key(f.a),
            "relation": f.relation,
            "object": f.b, "object_key": _key(f.b),
            "kind": f.kind,
            "construction": f.relation,
            "ambiguous": 1 if f.ambiguous else 0,
            "status": (f.extra or {}).get("status"),
            "scope": (f.extra or {}).get("scope"),
            "strength": f.strength,
            "sentence": f.sentence[:400],
            "source_path": md.source_path,
            "file_name": md.file_name,
            "category": md.category,
            "document_date": md.document_date,
            "fingerprint": md.source_fingerprint,
            "chunk_index": md.chunk_index,
        })
    return rows


def build(chunks, conn: sqlite3.Connection, only_changed: bool = False,
          limit: int | None = None, progress=None) -> dict:
    """
    Bygg eller uppdatera observationsindexet.

    only_changed=True hoppar över dokument vars fingerprint är
    oförändrat. Det är inkrementaliteten: att köra om tusentals
    dokument dagligen är onödigt när fingerprintet redan säger vad som
    ändrats.
    """
    if not is_available():
        raise RuntimeError(
            "Stanza är inte tillgängligt — Attest kan inte byggas. "
            "Installera med 'pip install stanza' och hämta modellen "
            "med stanza.download('sv')."
        )

    by_doc: dict[str, list] = {}
    for c in chunks:
        by_doc.setdefault(c.metadata.source_path, []).append(c)

    known = {
        row[0]: row[1]
        for row in conn.execute("SELECT source_path, fingerprint FROM documents")
    }

    paths = sorted(by_doc)
    if limit:
        paths = paths[:limit]

    # INNEHÅLLSHASH FÖR ALLA DOKUMENT, även de som hoppas över nedan.
    # Hashen räknas ur chunktexterna som redan ligger i indexet, inte ur
    # filen på disk: den fångar därmed också två kopior som exporterats
    # om och skiljer sig på bytenivå men bär samma text. Kontextprefixet
    # i chunktexten består av dokumenttitel och rubrikkedja, som är
    # sökvägsoberoende — annars vore hashen lika värdelös som
    # fingerprintet.
    #
    # Backfillen är skild från byggslingan därför att only_changed
    # hoppar över oförändrade dokument. Utan den skulle en befintlig
    # databas aldrig få sina hashar, och kolumnen förbli NULL i
    # tysthet.
    hashes = {p: _content_hash(by_doc[p]) for p in paths}
    for path, h in hashes.items():
        conn.execute(
            "UPDATE observations SET content_hash = ?"
            " WHERE source_path = ? AND (content_hash IS NULL OR content_hash <> ?)",
            (h, path, h),
        )
        conn.execute(
            "UPDATE documents SET content_hash = ?"
            " WHERE source_path = ? AND (content_hash IS NULL OR content_hash <> ?)",
            (h, path, h),
        )
    conn.commit()

    t0 = time.perf_counter()
    stats = {"documents": 0, "skipped": 0, "observations": 0, "removed": 0}

    for i, path in enumerate(paths, start=1):
        doc_chunks = by_doc[path]
        fp = doc_chunks[0].metadata.source_fingerprint
        if only_changed and known.get(path) == fp and fp is not None:
            stats["skipped"] += 1
            continue

        cur = conn.execute("DELETE FROM observations WHERE source_path = ?", (path,))
        stats["removed"] += cur.rowcount if cur.rowcount > 0 else 0

        rows: list[dict] = []
        for c in doc_chunks:
            rows.extend(_observations_from_chunk(c))
        for r in rows:
            r["content_hash"] = hashes[path]

        if rows:
            conn.executemany(
                "INSERT INTO observations (subject, subject_key, relation, object,"
                " object_key, kind, construction, ambiguous, status, scope, strength, sentence,"
                " source_path, file_name, category, document_date, fingerprint,"
                " content_hash, chunk_index) VALUES (:subject, :subject_key, :relation,"
                " :object, :object_key, :kind, :construction, :ambiguous, :status, :scope,"
                " :strength, :sentence, :source_path, :file_name, :category, :document_date,"
                " :fingerprint, :content_hash, :chunk_index)",
                rows,
            )
        md = doc_chunks[0].metadata
        conn.execute(
            "INSERT OR REPLACE INTO documents (source_path, fingerprint,"
            " content_hash, document_date, category, num_obs, built_at)"
            " VALUES (?,?,?,?,?,?,?)",
            (path, fp, hashes[path], md.document_date, md.category, len(rows),
             time.strftime("%Y-%m-%dT%H:%M:%S")),
        )
        conn.commit()
        stats["documents"] += 1
        stats["observations"] += len(rows)
        if progress:
            progress(i, len(paths), path, len(rows))

    # Dokument som försvunnit ur beståndet ska inte lämna spår kvar.
    if not limit:
        for path in list(known):
            if path not in by_doc:
                conn.execute("DELETE FROM observations WHERE source_path = ?", (path,))
                conn.execute("DELETE FROM documents WHERE source_path = ?", (path,))
                stats["removed"] += 1
        conn.commit()

    stats["seconds"] = round(time.perf_counter() - t0, 1)
    return stats


# ---------------------------------------------------------------------------
# Uppslag
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    subject: str
    object: str
    documents: int
    observations: int
    ambiguous_only: bool
    first_date: str | None
    last_date: str | None
    constructions: list[str]
    sentences: list[str]
    sources: list[str]
    unambiguous_documents: int = 0
    statuses: list[str] = field(default_factory=list)
    confirmed_documents: int = 0
    scopes: list[str] = field(default_factory=list)
    # Var bindningen står: (source_path, chunk_index) per observation,
    # starkast först. Retrievalen behöver den EXAKTA chunken, inte bara
    # dokumentet — en bindning i ett protokoll om tjugosex chunkar
    # hjälper inte om fel chunk hämtas. Uppgiften har funnits i
    # tabellen sedan schemat skrevs men aggregerades bort.
    locations: list[tuple[str, int]] = field(default_factory=list)
    strength: float = 0.0
    recency: float = 0.0
    relevance: float = 0.0
    days_since_last: int | None = None

    def as_dict(self) -> dict:
        return {
            "subject": self.subject, "object": self.object,
            "documents": self.documents, "observations": self.observations,
            "unambiguous_documents": self.unambiguous_documents,
            "statuses": self.statuses,
            "confirmed_documents": self.confirmed_documents,
            "scopes": self.scopes,
            "ambiguous_only": self.ambiguous_only,
            "first_date": self.first_date, "last_date": self.last_date,
            "days_since_last": self.days_since_last,
            "strength": round(self.strength, 3),
            "recency": round(self.recency, 3),
            "relevance": round(self.relevance, 3),
            "constructions": self.constructions,
            "sentences": self.sentences[:3],
            "sources": self.sources[:5],
        }


def _same_person(a: str, b: str) -> bool:
    """
    Är två namnformer samma person?

    REGELN ÄR STRUKTURELL, INTE ETT AVSTÅNDSMÅTT.

        "A Lind" / "A Maria Lind"      -> samma person
        "A Lund" / "A Lundgren"        -> OLIKA personer

    Redigeringsavståndet är i praktiken lika stort i båda fallen, och
    ingen tröskel kan skilja dem: skillnaden ligger i verkligheten, inte
    i tecknen. Fuzzy matching löser det därför inte.

    Det som bär är att första och sista NAMNLEDET är identiska och att
    skillnaden ligger i mellanled. Ett extra mellannamn är ett extra
    LED; ett längre efternamn är ett annat ORD.

    Böjningstolerans på ändpunkterna fångar genitivvarianter (samma
    efternamn med och utan -s) utan att slå ihop två skilda efternamn,
    eftersom avledningsändelser inte är böjningsändelser.

    SAMMANVÄGNINGEN SKER VID UPPSLAG, inte vid lagring. Observationerna
    behåller källans ordalydelse; det är bara i aggregeringen de förs
    samman. Då går beslutet att ändra utan att indexet byggs om.
    """
    at, bt = a.split(), b.split()
    if not at or not bt:
        return False
    if at == bt:
        return True
    if len(at) < 2 or len(bt) < 2:
        return False

    def ends_match(x: str, y: str) -> bool:
        if x == y:
            return True
        return is_inflection_of(x, y) or is_inflection_of(y, x)

    if not ends_match(at[0], bt[0]) or not ends_match(at[-1], bt[-1]):
        return False
    # Ändpunkterna stämmer; mellanleden får skilja sig i antal men de
    # som finns i båda måste stämma överens i ordning.
    mid_a, mid_b = at[1:-1], bt[1:-1]
    shorter, longer = (mid_a, mid_b) if len(mid_a) <= len(mid_b) else (mid_b, mid_a)
    it = iter(longer)
    return all(any(ends_match(m, x) for x in it) for m in shorter)


def _content_hash(doc_chunks) -> str:
    """
    Dokumentets innehåll som sha256 över dess chunktexter.

    Räknas ur indexet, inte ur filen på disk: två kopior som
    exporterats om och skiljer sig på bytenivå men bär samma text får
    då samma hash, och backfillen behöver ingen filåtkomst. Chunkarna
    sorteras på chunk_index så att hashen inte beror på hämtordningen.

    Kontextprefixet i chunktexten består av dokumenttitel och
    rubrikkedja. Båda är sökvägsoberoende — vore de det inte skulle
    hashen dela fingerprintets fel.
    """
    import hashlib

    ordnade = sorted(doc_chunks, key=lambda c: c.metadata.chunk_index)
    h = hashlib.sha256()
    for c in ordnade:
        h.update(c.text.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _doc_key(row) -> str:
    """
    Distinktnyckel för dokumenträkning: innehållet, inte sökvägen.

    Samma dokument arkiverat i flera mappar ska räknas som ETT belägg.
    Uppmätt 2026-08-22: 23 dokument i beståndet ligger under flera
    sökvägar, fyra av dem under tre — högskolans regelträd arkiverar
    samma normdokument under flera ämnesmappar. Räknade per sökväg blir
    ett belägg två eller tre, och ett översättningspar
    (parentes:identitet) nådde på det viset relevans 0,312, över
    reservationsgolvet 0,25, och fick reservera samma passage två
    gånger.

    NYCKELN ÄR content_hash, INTE fingerprint. source_fingerprint är en
    ändringsstämpel över sökväg, storlek och mtime — sökvägen ingår i
    hashen, så två kopior kan per konstruktion aldrig dela fingerprint.
    En tidigare version av den här funktionen byggde på motsatsen och
    var därför verkningslös.

    Fallback i två steg för rader från en databas som ännu inte fyllt
    kolumnen: fingerprint, därefter sökväg. Båda ger den gamla
    överräkningen, vilket är rätt utfall att falla tillbaka på — hellre
    känt beteende än att klumpa ihop rader vars innehåll är okänt.
    """
    return (
        row.get("content_hash")
        or row.get("fingerprint")
        or row["source_path"]
    )


def _locations(rows) -> list[tuple[str, int]]:
    """
    Var bindningen står, starkast belägg först.

    Ordningen är entydigt före tvetydigt, bekräftat före föreslaget,
    därefter nyast först. Den som bara vill ha EN passage ska få den
    bäst belagda — inte den som råkade ligga först i tabellen.

    Att ett FÖRSLAG rankas efter en bekräftad tillsättning är samma
    skäl som ger det halv vikt i relevansmodellen: förslaget kan ha
    bifallits utan att namnet upprepades, eller ha fallit.

    Dedupliceringen går på innehåll (_doc_key), inte sökväg: samma fil
    i två mappar är samma passage, och att reservera båda ger syntesen
    samma text två gånger. Utåt bärs source_path — _chunks_at slår upp
    i indexet på sökväg.
    """
    # Stabil sortering, svagaste nyckel först: datum, sedan status,
    # sist tvetydighet — så att tvetydighet väger tyngst.
    ordered = sorted(rows, key=lambda r: (r["document_date"] or ""), reverse=True)
    ordered.sort(key=lambda r: (1 if r["ambiguous"] else 0,
                                1 if r["status"] == "föreslagen" else 0))
    out: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for r in ordered:
        if r["chunk_index"] is None:
            continue
        key = (_doc_key(r), r["chunk_index"])
        if key in seen:
            continue
        seen.add(key)
        loc = (r["source_path"], r["chunk_index"])
        if loc not in out:
            out.append(loc)
    return out


def _most_common(values) -> str:
    """Vanligaste skrivformen; vid lika utfall den alfabetiskt första."""
    counts: dict[str, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return max(sorted(counts), key=lambda v: counts[v])


# RELEVANSMODELL
#
# Ersätter unikhetsregeln, som byggde på ett antagande som visade sig
# falskt: att ett uppdrag har en innehavare i taget. Vid högskolan
# finns fyra prefekter, flera proprefekter och tiotals studierektorer —
# och antalet proprefekter per institution bestäms av prefekten, alltså
# utan regel att koda mot. Överlappande intervall kan lika gärna betyda
# två samtidiga innehavare som en felaktig extraktion, och systemet kan
# inte veta vilket.
#
# Modellen RANGORDNAR därför utan att utesluta. Ingen kandidat stryks
# för att en annan är nyare; alla redovisas med sitt underlag.
#
# Tre komponenter, medvetet hållna åtskilda i utdata så att en
# rangordning går att förklara i ett svar:
#
# STYRKA växer med antalet OBEROENDE DOKUMENT, avtagande. Språnget
# från ett belägg till två är stort, från två till tre betydande, från
# tio till elva försumbart. En enda observation kan vara ett
# extraktionsfel, en felskrivning i protokollet eller en tvetydig
# konstruktion. Antalet observationer räknas inte: samma protokollmall
# upprepad arton gånger i sju dokument är sju belägg, inte arton.
#
# AKTUALITET avtar med tiden sedan senaste belägg, mätt mot BESTÅNDETS
# horisont och inte mot dagens datum. Slutar dokumenten i mars är allt
# därefter okänt, och en observation från februari ska inte straffas
# för att kalendern gått vidare.
#
# TVETYDIGHET reducerar vikten. En samordnad titelkonstruktion tillåter
# två läsningar och kan inte väga som ett entydigt belägg. Vikten är
# PROVISORISK — vi har ett bekräftat tvetydigt fall i beståndet, vilket
# inte räcker för att kalibrera. Konservativt lågt värde tills fler
# fall finns.
_AMBIGUOUS_WEIGHT = 0.25

# Ett förslag väger hälften av en tillsättning. Det är verklig men
# svagare evidens: förslaget kan ha bifallits utan att namnet
# upprepades, eller ha fallit. Provisoriskt värde.
_PROPOSED_WEIGHT = 0.5

# Halveringstid i dagar för aktualitet. Två år, satt mot
# mandatperioderna i beståndet (tre år med möjlighet till förnyelse).
# Kortare halveringstid lät en enda färsk observation slå ut ett
# väletablerat förhållande, vilket motsäger att ett enstaka belägg kan
# vara ett fel. Provisoriskt värde.
_RECENCY_HALFLIFE_DAYS = 730.0


def _parse_date(value: str | None):
    from datetime import date
    if not value:
        return None
    try:
        y, m, d = value[:10].split("-")
        return date(int(y), int(m), int(d))
    except (ValueError, AttributeError):
        return None


def compute_relevance(candidates: list[Candidate], horizon: str | None = None) -> None:
    """
    Beräkna styrka, aktualitet och relevans in-place.

    horizon är beståndets senaste dokumentdatum. Saknas det används
    kandidaternas eget senaste belägg, vilket ger den nyaste kandidaten
    full aktualitet.
    """
    import math

    h = _parse_date(horizon)
    if h is None:
        dates = [_parse_date(c.last_date) for c in candidates]
        dates = [d for d in dates if d]
        h = max(dates) if dates else None

    for c in candidates:
        # Vägt dokumentantal: tvetydiga belägg räknas som bråkdelar.
        proposed_only = max(
            c.unambiguous_documents - c.confirmed_documents, 0
        )
        weighted = (
            c.confirmed_documents
            + proposed_only * _PROPOSED_WEIGHT
            + (c.documents - c.unambiguous_documents) * _AMBIGUOUS_WEIGHT
        )
        # Avtagande avkastning, kalibrerad mot omdömet att ETT belägg
        # kan vara ett fel, TVÅ stärker rejält och TRE är mycket
        # starkt:  1 -> 0.33, 2 -> 0.67, 3 -> 0.82, 5 -> 0.93.
        # Aldrig noll, aldrig ett.
        c.strength = (
            weighted ** 2 / (weighted ** 2 + 2.0) if weighted > 0 else 0.0
        )

        last = _parse_date(c.last_date)
        if last and h:
            days = max((h - last).days, 0)
            c.days_since_last = days
            c.recency = 0.5 ** (days / _RECENCY_HALFLIFE_DAYS)
        else:
            c.days_since_last = None
            c.recency = 0.5      # okänt datum: varken gynnas eller straffas

        c.relevance = c.strength * c.recency


def _rows_to_candidates(rows) -> list[Candidate]:
    # Gruppera på nyckelpar, och slå därefter ihop grupper vars subjekt
    # är samma person i olika namnform. Objektnyckeln måste vara
    # identisk: samma person i två roller är två bindningar.
    groups: dict[tuple, list] = {}
    for r in rows:
        key = (r["subject_key"], r["object_key"])
        merged = None
        for existing in groups:
            if existing[1] != key[1]:
                continue
            if _same_person(existing[0], key[0]):
                merged = existing
                break
        groups.setdefault(merged or key, []).append(r)

    out: list[Candidate] = []
    for (_, _), rs in groups.items():
        dates = sorted(x["document_date"] for x in rs if x["document_date"])
        # Dokument räknas per INNEHÅLL (_doc_key), inte per sökväg —
        # annars dubblerar en fil i två mappar beläggningen. De råa
        # antalen observations/sentences lämnas medvetet odedupli-
        # cerade: de redovisar tabellens faktiska rader; styrkan gör
        # det inte.
        docs = {_doc_key(x) for x in rs}
        # Visa den VANLIGASTE skrivformen, inte den första raden i
        # gruppen. Nyckeln samlar "HR-specialist" och "hR-specialist"
        # under samma post, och vilken av dem som visas ska inte bero
        # på sorteringsordningen i databasen.
        unambiguous_docs = {
            _doc_key(x) for x in rs if not x["ambiguous"]
        }
        # Ett FÖRSLAG är inte en tillsättning. Dokument vars enda
        # bidrag är ett förslag räknas separat: förslaget kan ha
        # bifallits i nästa punkt utan att namnet upprepas, men det
        # kan också ha fallit. Att räkna dem lika vore att belägga
        # något som kanske aldrig hänt.
        statuses = sorted({x["status"] for x in rs if x["status"]})
        scopes = sorted({x["scope"] for x in rs if x["scope"]})
        confirmed_docs = {
            _doc_key(x) for x in rs
            if not x["ambiguous"] and x["status"] != "föreslagen"
        }
        out.append(Candidate(
            subject=_most_common(x["subject"] for x in rs),
            object=_most_common(x["object"] for x in rs),
            documents=len(docs), observations=len(rs),
            unambiguous_documents=len(unambiguous_docs),
            statuses=statuses,
            scopes=scopes,
            confirmed_documents=len(confirmed_docs),
            # Bär SAMTLIGA belägg tvetydighet är bindningen inte
            # entydigt belagd, oavsett hur många de är. Ett enda
            # entydigt belägg väger tyngre än tio tvetydiga.
            ambiguous_only=all(x["ambiguous"] for x in rs),
            first_date=dates[0] if dates else None,
            last_date=dates[-1] if dates else None,
            constructions=sorted({x["construction"] for x in rs}),
            sentences=[x["sentence"] for x in rs if x["sentence"]],
            sources=sorted({x["file_name"] for x in rs if x["file_name"]}),
            locations=_locations(rs),
        ))
    compute_relevance(out)
    # En kandidat med ENBART tvetydiga belägg kan aldrig rankas överst,
    # oavsett hur många de är. Det är ett hårt villkor och inte bara en
    # vikt: en konstruktion som tillåter två läsningar får inte bära ett
    # svar när det finns entydiga alternativ.
    out.sort(key=lambda c: (not c.ambiguous_only, c.relevance), reverse=True)
    return out


# Markörer som inleder en rollavgränsning. SAMMA MÄNGD som
# grammar._SCOPE_MARKS, avsiktligt: frågan och källan ska beskrivas på
# samma sätt, annars matchar de aldrig varandra.
SCOPE_MARKS = {"för", "inom", "vid"}

# Markörer som räknas som avgränsning I EN FRÅGA. Snävare än
# SCOPE_MARKS, och revideringen är gjord på mätning:
#
# "vid IIT" namnger organisationen, inte uppdragets avgränsning. I ett
# bestånd från en enda institution skiljer den ingen bindning från en
# annan, medan ordet kan utesluta varje bindning vars text stavar ut
# institutionsnamnet i stället för förkortningen. Modellen har
# dessutom redan en egen relation för organisationstillhörighet
# (tillhorighet), så att låta "vid" styra rollavgränsning blandar två
# saker som lagras åtskilt.
#
# Uttaget ur KÄLLTEXTEN behåller alla tre markörerna (se
# grammar._SCOPE_MARKS). Asymmetrin är avsiktlig: en källa som skriver
# "studierektor vid IIT" har angett något, medan en fråga som skriver
# "vid IIT" bara säger var vi är.
_QUESTION_SCOPE_MARKS = {"för", "inom"}

# Sluten klass av ord som aldrig är en avgränsning trots att de följer
# en markör. "ansvarar FÖR ATT budget upprättas" är en bisats, inte en
# rollavgränsning.
_SCOPE_NONWORDS = {
    "att", "den", "det", "de", "denna", "detta", "dessa", "sin", "sitt",
    "sina", "alla", "varje", "vem", "vad", "vilka", "vilken", "vilket",
}


def scope_terms(text: str) -> list[str]:
    """
    Avgränsningsord i en fråga: innehållsorden efter för/inom/vid.

    "studierektor för grundutbildningen vid IIT" ger
    ["grundutbildningen", "iit"]. Uttaget är medvetet grovt — det ska
    hitta KANDIDATER till avgränsning, och matchningen mot lagrade
    scopes avgör sedan.
    """
    words = re.findall(r"[\wÅÄÖåäö-]+", text.lower())
    out: list[str] = []
    for i, w in enumerate(words[:-1]):
        if w in _QUESTION_SCOPE_MARKS:
            nxt = words[i + 1]
            if len(nxt) >= 3 and nxt not in SCOPE_MARKS and nxt not in _SCOPE_NONWORDS:
                out.append(nxt)
    return out


def _scope_compatible(candidate: Candidate, wanted: list[str]) -> bool:
    """
    Är kandidatens avgränsning förenlig med den frågan efterfrågar?

    TRE UTFALL, OCH DET MITTERSTA ÄR POÄNGEN:

      matchar        studierektor FÖR grundutbildningen  -> ja
      annan          studierektor FÖR forskarutbildning  -> NEJ
      ingen alls     studierektor (texten anger ingen)   -> ja

    Att en observation saknar scope betyder att TEXTEN inte angav
    någon, inte att rollen är oavgränsad. Frånvaro av uppgift är inte
    en motsägelse, och samma princip styr korpuskontrollen: systemet
    jämför belägg det faktiskt har, och avstår från att sluta sig till
    något ur en lucka.

    Att en observation har en ANNAN avgränsning är däremot en verklig
    oförenlighet. Elva programansvariga ansvarar för elva olika
    program, och studierektor för grundutbildningen är inte
    studierektor för forskarutbildningen.
    """
    if not wanted:
        return True
    if not candidate.scopes:
        return True
    for scope in candidate.scopes:
        for word in re.findall(r"[\wÅÄÖåäö-]+", scope.lower()):
            for w in wanted:
                if word == w or is_inflection_of(word, w) or is_inflection_of(w, word):
                    return True
    return False


def lookup_subject(
    conn, term: str, kind: str = "identitet", scope: list[str] | None = None
) -> list[Candidate]:
    """Vilka objekt binds till detta subjekt? ('Vad är X?')"""
    rows = _match(conn, "subject_key", term, kind)
    return _filter_scope(_rows_to_candidates(rows), scope)


def lookup_object(
    conn, term: str, kind: str = "identitet", scope: list[str] | None = None
) -> list[Candidate]:
    """
    Vilka subjekt binds till detta objekt? ('Vem är X?')

    scope avgränsar uppslaget: ["grundutbildningen"] utesluter
    bindningar som texten uttryckligen knutit till något annat, men
    behåller dem som saknar avgränsning. Se _scope_compatible.
    """
    rows = _match(conn, "object_key", term, kind)
    return _filter_scope(_rows_to_candidates(rows), scope)


def _scope_word_matches(candidate: Candidate, word: str) -> bool:
    for scope in candidate.scopes:
        for w in re.findall(r"[\wÅÄÖåäö-]+", scope.lower()):
            if w == word or is_inflection_of(w, word) or is_inflection_of(word, w):
                return True
    return False


def _filter_scope(
    candidates: list[Candidate], wanted: list[str] | None
) -> list[Candidate]:
    """
    EN SKILLNAD FILTRERAR BARA NÄR BESTÅNDET GÖR DEN — men "gör den"
    betyder att beståndet avgränsar rollen alls, inte att just den
    efterfrågade avgränsningen finns.

    Skillnaden mättes 2026-08-18. Uppslaget "studierektor" ger två
    kandidater, båda med UTSKRIVEN avgränsning
    (forskarutbildningsområdet respektive ett ämne). Frågan gällde
    grundutbildningen. En tidigare version krävde att
    avgränsningsordet matchade minst en kandidat för att få filtrera,
    och lät därför båda passera — vilket besvarade en fråga om
    grundutbildningen med forskarutbildningens studierektorer.

    Att ingen kandidat matchar är inte brist på information. Det är
    beskedet att beståndet inte binder den avgränsning frågan gäller,
    och rätt svar är då inget svar. Villkoret är därför omvänt: bara
    när INGEN kandidat har någon avgränsning alls avstår filtret.
    """
    if not wanted:
        return candidates

    # Gör beståndet någon avgränsningsskillnad alls för den här rollen?
    # Har ingen kandidat någon scope kan ordet inte diskriminera, och
    # då vore uteslutning ren gissning.
    if not any(c.scopes for c in candidates):
        return candidates

    return [c for c in candidates if _scope_compatible(c, wanted)]


def _matches_terms(value: str, key: str) -> bool:
    """
    Matchar nyckeln som hel term eller som rollens huvudord?

    Regeln: nyckeln ska vara värdets AVSLUTANDE ordföljd.

        "prefekt"  matchar  "prefekt", "tf prefekt"
        "prefekt"  matchar INTE  "proprefekt", "prefektbeslut"
        "lektor"   matchar  "biträdande lektor"

    Bestämningsord står först i svenska rolluttryck, så ett uppslag på
    grundrollen fångar även dess varianter — men de redovisas som
    EGNA kandidater med sin egen etikett, eftersom "tf proprefekt" och
    "proprefekt" är olika roller med olika innehavare. Användaren ser
    skillnaden i utdata; sammanvägningen håller dem isär.

    Att sammansättningar inte matchar är hela poängen: prefekt och
    proprefekt betecknar skilda uppdrag som per definition innehas av
    olika personer, och delsträngsmatchning gav tolv kandidater varav
    sju var brus.
    """
    vt, kt = value.split(), key.split()
    if not kt or len(kt) > len(vt):
        return False
    return vt[-len(kt):] == kt


def _match(conn, column: str, term: str, kind: str) -> list[dict]:
    """
    Hämta observationer vars nyckel matchar termen.

    MATCHNINGEN GÅR PÅ HELA ORD, INTE PÅ DELSTRÄNGAR.

    Uppmätt 2026-08-15: uppslaget "prefekt" träffade proprefekt,
    prefektbeslut och "prefekt godkänna", eftersom villkoret var
    LIKE '%prefekt%'. Tolv kandidater varav sju var brus, och rollen
    bedömdes felaktigt icke-unik.

    Det är samma fel som en gång fanns i predication._pair_match, och
    samma rättning: en delsträng är inte en term. "prefekt" ska matcha
    "prefekt" och "tf prefekt" men aldrig "proprefekt" — de betecknar
    olika roller och innehas per definition av olika personer.

    Kvar står tolerans i BÖJNING, som fallback när ordagrann matchning
    ger noll. Stavningsvarianter i källorna ska inte räknas som skilda
    entiteter vid sammanvägning, medan källans ordalydelse bevaras i
    utdata.
    """
    conn.row_factory = sqlite3.Row
    key = _key(term)
    rows = []
    for r in conn.execute(
        "SELECT * FROM observations WHERE kind = ?", (kind,)
    ):
        d = dict(r)
        if _matches_terms(d[column], key):
            rows.append(d)
    if rows:
        return rows
    # Böjningstolerant fallback, dyrare men bara när exakt uppslag gav noll.
    for r in conn.execute("SELECT * FROM observations WHERE kind = ?", (kind,)):
        d = dict(r)
        val = d[column]
        if any(is_inflection_of(w, key) or is_inflection_of(key, w)
               for w in val.split()):
            rows.append(d)
    return rows


# role_is_unique BORTTAGEN 2026-08-16.
#
# Funktionen byggde på antagandet att ett uppdrag har en innehavare i
# taget och slöt av överlappande intervall att rollen bars av flera.
# Antagandet är falskt: vid högskolan finns fyra prefekter, flera
# proprefekter och tiotals studierektorer, och antalet proprefekter per
# institution bestäms av prefekten — alltså utan regel att koda mot.
#
# Överlappande intervall kan därför lika gärna betyda två samtidiga
# innehavare vid samma enhet, två innehavare vid olika enheter, eller
# en felaktig extraktion. Systemet kan inte avgöra vilket, och en
# funktion som ändå uttalar sig lovar mer än underlaget bär.
#
# Att "proprefekt" bedömdes UNIK i mätningen var dessutom en artefakt
# av att indexet byggts ur en enda institutions protokoll.
#
# Ersatt av compute_relevance, som rangordnar utan att utesluta.


def observations_for_document(conn, source_path: str) -> list[dict]:
    """
    Observationer ur ETT dokument, i chunkordning.

    Uppslagen ovan går från termens håll: vilka subjekt binds till
    detta objekt, vilka objekt till detta subjekt. Ingen av dem svarar
    på vad ett enskilt dokument gav — trots att source_path är
    indexerad sedan schemat skrevs.

    conn=None öppnar en egen anslutning; samma mönster som
    corpus_guard.check_answer.
    """
    if conn is None:
        conn = connect()
    conn.row_factory = sqlite3.Row
    return [
        dict(r) for r in conn.execute(
            "SELECT * FROM observations WHERE source_path = ?"
            " ORDER BY chunk_index, id",
            (source_path,),
        )
    ]


def coverage(conn, source_paths: list[str]) -> dict:
    """
    Täckning per dokument: vilka av beståndets dokument har
    observationer, och vilka gav noll?

    NOLLFALLET ÄR POÄNGEN. Ett dokument utan observationer bidrar
    ingenting till entitetsuppslag, korpuskontroll eller uppräkning —
    men det syns inte någonstans idag, lika lite som evidenslösa
    dokument syntes innan ingest började leta efter dem. Frånvaro av
    belägg är inte ett fynd om något, men frånvaron av UTTAG är ett
    fynd om systemet: den skiljer ett dokument som inte säger något om
    roller från ett vars konstruktioner uttaget inte bär.

    source_paths är beståndets dokument enligt indexet, så att
    dokument som aldrig nått Attest räknas med — inte bara de som
    finns i observationstabellen.
    """
    with_obs = {
        row[0] for row in conn.execute(
            "SELECT DISTINCT source_path FROM observations"
        )
    }
    known = set(source_paths)
    covered = sorted(known & with_obs)
    empty = sorted(known - with_obs)
    # Dokument i Attest som inte längre finns i indexet: stale, och
    # ett tecken på att attest-build behöver köras om.
    stale = sorted(with_obs - known)
    return {
        "documents_indexed": len(known),
        "documents_with_observations": len(covered),
        "documents_without_observations": len(empty),
        "without_observations": empty,
        "stale_documents": stale,
    }


def stats(conn) -> dict:
    conn.row_factory = sqlite3.Row
    # Dokument räknas per innehåll, samma nyckel som _doc_key: samma
    # dokument i flera mappar är ett dokument. Skiljer sig siffran från
    # coverage() — som räknar indexposter per sökväg — är differensen
    # exakt antalet dubblettkopior, och duplicates() pekar ut dem.
    row = conn.execute(
        "SELECT COUNT(*) n,"
        " COUNT(DISTINCT COALESCE(content_hash, fingerprint, source_path)) d"
        " FROM observations"
    ).fetchone()
    kinds = {
        r["kind"]: r["c"]
        for r in conn.execute("SELECT kind, COUNT(*) c FROM observations GROUP BY kind")
    }
    return {"observations": row["n"], "documents": row["d"], "per_kind": kinds}


def duplicates(conn) -> list[dict]:
    """
    Samma innehåll under flera sökvägar.

    Läses ur dokumentregistret, inte observationstabellen, så att även
    dubbletter utan observationer syns — rapporten finns för
    dokumenthygienen i beståndet, inte bara för styrkeräkningen.

    Vilken kopia som ska bort är ett beslut om docs/, inte om Attest,
    och det är inte självklart att någon ska bort: sökvägen bestämmer
    dokumenttyp och normativ tyngd, så två kopior kan bära olika vikt.
    Rapporten gör dubbleringen synlig och räkningen korrekt; den städar
    inte.
    """
    conn.row_factory = sqlite3.Row
    by_hash: dict[str, set[str]] = {}
    for r in conn.execute(
        "SELECT content_hash, source_path FROM documents"
        " WHERE content_hash IS NOT NULL"
    ):
        by_hash.setdefault(r["content_hash"], set()).add(r["source_path"])
    return [
        {"content_hash": h, "paths": sorted(paths)}
        for h, paths in sorted(by_hash.items())
        if len(paths) > 1
    ]
