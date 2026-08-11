"""
QUD-drift-skydd.

Problemet: klassificeraren kan ibland behandla en yttring som
related_to_qud trots att den i praktiken bryter ämnestråden helt.
När det händer används den aktiva QUD:n som retrieval-ankare mot
fel ämne, och (vid broadening) skrivs följdfrågan om med fel
kontext — vilket typiskt leder till fel underlag eller abstain.

Måttet har två nivåer:

1. **Dokumentbaserad drift (primär när material finns).** Yttringen
   jämförs mot texterna i de chunkar som bar de senaste svaren
   (state.active_hits). Med E5-prefix blir detta fråga-mot-passage —
   exakt den regim embeddingmodellen är tränad för. Frågan som mäts
   är den relevanta: "handlar yttringen fortfarande om det material
   vi läser?" Kalibreringen 2026-08-11 visade att fråga-mot-fråga-
   likhet INTE kan skilja ämnesbyten som delar institutionell
   vokabulär från äkta följdfrågor (ämnesbytet adjungerad lektor →
   IIT-rutiner fick högre likhet, 0,86, än samtliga besläktade par).

2. **QUD-baserad drift (fallback).** När inga aktiva chunkar finns
   (t.ex. efter abstain) jämförs yttringen mot QUD-texten som
   tidigare — fråga-mot-fråga.

Trösklarna är config-parametrar (qud_drift_threshold respektive
qud_drift_doc_threshold). Dokumenttröskeln är provisorisk tills den
kalibrerats mot verkliga körningar — båda likheterna loggas därför
alltid i debug/JSONL, så att tröskeln kan sättas från uppmätta
fördelningar i stället för gissningar.

Beslutet att överrida klassificeringen fattas av api-lagret, inte
här — denna modul levererar bara mätningen. Notera att en felaktig
drift-överridning numera är återhämtningsbar: den kontextuella
fallbacken i api.py kör om retrieval med kontext om det kontextlösa
försöket abstainar.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.embeddings import Embedder

logger = logging.getLogger(__name__)

# Antal aktiva chunkar som används i dokumentjämförelsen, och tak på
# hur många tecken per chunk som skickas till embeddingmodellen.
_MAX_ACTIVE_TEXTS = 4
_MAX_TEXT_CHARS = 1200


@dataclass
class DriftMeasurement:
    """Resultatet av en drift-kontroll."""
    similarity: float                # yttring ↔ QUD (fråga-mot-fråga)
    threshold: float
    drift_detected: bool
    # Dokumentbaserad mätning — None när inga aktiva chunkar fanns.
    doc_similarity: float | None = None
    doc_threshold: float | None = None
    decided_by: str = "qud"          # "doc" | "qud"


def _cosine(a: list[float], b: list[float]) -> float:
    """
    Cosinuslikhet mellan två normaliserade vektorer.

    Embedder normaliserar vektorerna (normalize_embeddings=True),
    så skalärprodukten ÄR cosinuslikheten.
    """
    if len(a) != len(b):
        raise ValueError(f"Olika dimensioner: {len(a)} vs {len(b)}")
    return sum(x * y for x, y in zip(a, b))


def measure_drift(
    utterance: str,
    qud_text: str | None,
    embedder: Embedder,
    threshold: float,
    active_hit_texts: list[str] | None = None,
    doc_threshold: float | None = None,
) -> DriftMeasurement | None:
    """
    Mät ämnesavstånd mellan aktuell yttring och samtalets aktiva
    kontext.

    Om active_hit_texts (texter ur chunkarna som bar senaste svaren)
    och doc_threshold anges avgörs driften av den HÖGSTA likheten
    mellan yttringen och någon av chunktexterna (fråga-mot-passage).
    Annars avgörs den av likheten mot QUD-texten (fråga-mot-fråga),
    som tidigare.

    Båda måtten beräknas och returneras när det går — även det som
    inte avgör beslutet — så att trösklar kan kalibreras ur JSONL-
    spåren i efterhand.

    Returnerar None om QUD saknas eller embeddings inte kan beräknas
    (konservativt: ingen drift antas).
    """
    if qud_text is None or not qud_text.strip():
        return None

    try:
        vec_utterance = embedder.embed_query(utterance)
        vec_qud = embedder.embed_query(qud_text)
    except Exception as e:
        logger.warning(
            "QUD-drift: kunde inte beräkna embeddings: %s. "
            "Antar ingen drift (konservativt).",
            e,
        )
        return None

    qud_similarity = round(_cosine(vec_utterance, vec_qud), 4)

    doc_similarity: float | None = None
    texts = [
        t[:_MAX_TEXT_CHARS]
        for t in (active_hit_texts or [])
        if t and t.strip()
    ][:_MAX_ACTIVE_TEXTS]

    if texts and doc_threshold is not None:
        try:
            passage_vecs = embedder.embed_texts(texts)
            doc_similarity = round(
                max(_cosine(vec_utterance, pv) for pv in passage_vecs), 4
            )
        except Exception as e:
            logger.warning(
                "QUD-drift: kunde inte beräkna dokumentlikhet: %s. "
                "Faller tillbaka på QUD-likhet.",
                e,
            )
            doc_similarity = None

    if doc_similarity is not None and doc_threshold is not None:
        return DriftMeasurement(
            similarity=qud_similarity,
            threshold=threshold,
            drift_detected=doc_similarity < doc_threshold,
            doc_similarity=doc_similarity,
            doc_threshold=doc_threshold,
            decided_by="doc",
        )

    return DriftMeasurement(
        similarity=qud_similarity,
        threshold=threshold,
        drift_detected=qud_similarity < threshold,
        doc_similarity=doc_similarity,
        doc_threshold=doc_threshold,
        decided_by="qud",
    )
