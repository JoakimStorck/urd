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
from dataclasses import dataclass
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
    strength     TEXT,
    sentence     TEXT,
    source_path  TEXT NOT NULL,
    file_name    TEXT,
    category     TEXT,
    document_date TEXT,
    fingerprint  TEXT,
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
    document_date TEXT,
    category     TEXT,
    num_obs      INTEGER,
    built_at     TEXT
);
"""


def _key(text: str) -> str:
    """
    Uppslagsnyckel: gemener, skiljetecken bort, ordordning bevarad.

    Tolerant vid sammanvägning, ordagrann vid återgivning. Beståndet
    stavar samma person både "Anna" och "Anna", och "Sara Lundquist"
    respektive "Sara Lundqvist". Nyckeln normaliserar för aggregering
    medan `subject`/`object` bevarar källans ordalydelse.
    """
    t = re.sub(r"[^\wÅÄÖåäö\s-]", " ", text.lower())
    return " ".join(t.split())


def connect(path: Path | None = None) -> sqlite3.Connection:
    p = path or DB_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p))
    conn.executescript(SCHEMA)
    return conn


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
_STORED_KINDS = {"identitet", "agens", "modalitet", "forkortning", "tillhorighet"}


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

        if rows:
            conn.executemany(
                "INSERT INTO observations (subject, subject_key, relation, object,"
                " object_key, kind, construction, ambiguous, strength, sentence,"
                " source_path, file_name, category, document_date, fingerprint,"
                " chunk_index) VALUES (:subject, :subject_key, :relation, :object,"
                " :object_key, :kind, :construction, :ambiguous, :strength,"
                " :sentence, :source_path, :file_name, :category, :document_date,"
                " :fingerprint, :chunk_index)",
                rows,
            )
        md = doc_chunks[0].metadata
        conn.execute(
            "INSERT OR REPLACE INTO documents (source_path, fingerprint,"
            " document_date, category, num_obs, built_at) VALUES (?,?,?,?,?,?)",
            (path, fp, md.document_date, md.category, len(rows),
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

    def as_dict(self) -> dict:
        return {
            "subject": self.subject, "object": self.object,
            "documents": self.documents, "observations": self.observations,
            "ambiguous_only": self.ambiguous_only,
            "first_date": self.first_date, "last_date": self.last_date,
            "constructions": self.constructions,
            "sentences": self.sentences[:3],
            "sources": self.sources[:5],
        }


def _rows_to_candidates(rows) -> list[Candidate]:
    groups: dict[tuple, list] = {}
    for r in rows:
        groups.setdefault((r["subject_key"], r["object_key"]), []).append(r)

    out: list[Candidate] = []
    for (_, _), rs in groups.items():
        dates = sorted(x["document_date"] for x in rs if x["document_date"])
        docs = {x["source_path"] for x in rs}
        out.append(Candidate(
            subject=rs[0]["subject"], object=rs[0]["object"],
            documents=len(docs), observations=len(rs),
            # Bär SAMTLIGA belägg tvetydighet är bindningen inte
            # entydigt belagd, oavsett hur många de är. Ett enda
            # entydigt belägg väger tyngre än tio tvetydiga.
            ambiguous_only=all(x["ambiguous"] for x in rs),
            first_date=dates[0] if dates else None,
            last_date=dates[-1] if dates else None,
            constructions=sorted({x["construction"] for x in rs}),
            sentences=[x["sentence"] for x in rs if x["sentence"]],
            sources=sorted({x["file_name"] for x in rs if x["file_name"]}),
        ))
    out.sort(key=lambda c: (not c.ambiguous_only, c.documents, c.last_date or ""),
             reverse=True)
    return out


def lookup_subject(conn, term: str, kind: str = "identitet") -> list[Candidate]:
    """Vilka objekt binds till detta subjekt? ('Vad är X?')"""
    rows = _match(conn, "subject_key", term, kind)
    return _rows_to_candidates(rows)


def lookup_object(conn, term: str, kind: str = "identitet") -> list[Candidate]:
    """Vilka subjekt binds till detta objekt? ('Vem är X?')"""
    rows = _match(conn, "object_key", term, kind)
    return _rows_to_candidates(rows)


def _match(conn, column: str, term: str, kind: str) -> list[dict]:
    """
    Hämta observationer vars nyckel matchar termen.

    Matchningen är tolerant: exakt, som delfras, eller som
    böjningsvariant. Stavningsvarianter i källorna ("Anna"/"Anna",
    "Lundquist"/"Lundqvist") får inte räknas som skilda entiteter vid
    sammanvägning — men källans ordalydelse bevaras i utdata.
    """
    conn.row_factory = sqlite3.Row
    key = _key(term)
    rows = []
    for r in conn.execute(
        f"SELECT * FROM observations WHERE kind = ? AND ({column} = ?"
        f" OR {column} LIKE ?)", (kind, key, f"%{key}%")
    ):
        rows.append(dict(r))
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


def role_is_unique(candidates: list[Candidate]) -> bool | None:
    """
    Bärs rollen av en person i taget? True / False / None (går ej att avgöra).

    Överlappar två personers intervall är rollen icke-unik (professor,
    HR-specialist). Följer de på varandra är den unik (prefekt,
    studierektor) — och då sluter ett nytt belägg det föregående
    intervallet. Detta MÄTS ur datat, det antas inte.

    OSÄKERHET REDOVISAS. Ett enda belägg per person ger punktintervall,
    och punkter kan aldrig överlappa — två professorer med var sitt
    omnämnande skulle då se ut som en följd av innehavare. Uppmätt
    2026-08-15. Går det inte att skilja följd från samtidighet
    returneras None, aldrig True: att stänga ett intervall är den
    konsekvensrika handlingen och kräver positivt stöd.

    Tvetydiga belägg utesluts helt — en konstruktion som tillåter två
    läsningar får inte avgöra rollens natur.
    """
    dated = [
        c for c in candidates
        if c.first_date and c.last_date and not c.ambiguous_only
    ]
    if len(dated) < 2:
        return None

    for i, a in enumerate(dated):
        for b in dated[i + 1:]:
            if a.first_date <= b.last_date and b.first_date <= a.last_date:
                return False

    # Inga överlapp — men bär ingen kandidat ett spann går följd inte
    # att skilja från samtidighet.
    if all(c.first_date == c.last_date for c in dated):
        return None
    return True


def stats(conn) -> dict:
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT COUNT(*) n, COUNT(DISTINCT source_path) d FROM observations"
    ).fetchone()
    kinds = {
        r["kind"]: r["c"]
        for r in conn.execute("SELECT kind, COUNT(*) c FROM observations GROUP BY kind")
    }
    return {"observations": row["n"], "documents": row["d"], "per_kind": kinds}
