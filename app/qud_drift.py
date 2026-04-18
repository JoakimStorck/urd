"""
QUD-drift-skydd.

Problemet: klassificeraren kan ibland behandla en yttring som
related_to_qud trots att den i praktiken bryter ämnestråden helt.
När det händer används den aktiva QUD:n som retrieval-ankare mot
fel ämne, och cross-encodern får ett bullrigt underlag att bedöma
— vilket typiskt leder till abstain eller försämrade svar.

Skyddet är avsiktligt enkelt: vi beräknar cosinuslikhet mellan
embeddings av aktuell yttring och current_qud_text. Om likheten
är under en tröskel tolkar vi det som att yttringen har lämnat
QUD:ns ämnessfär, och rekommenderar att klassificeringen överrids
till new_main_question.

Beslutet att överrida fattas av api-lagret, inte här — denna modul
levererar bara mätningen. Det gör den testbar och begränsar
sidoeffekterna.

Tröskeln är kalibrerad för multilingual-e5-large, där besläktade
frågor typiskt hamnar ovanför 0.60 och orelaterade hamnar
markant under 0.55. Värdet är en config-parameter
(qud_drift_threshold) så det kan justeras utan kodändringar.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.embeddings import Embedder

logger = logging.getLogger(__name__)


@dataclass
class DriftMeasurement:
    """Resultatet av en drift-kontroll."""
    similarity: float
    threshold: float
    drift_detected: bool


def _cosine(a: list[float], b: list[float]) -> float:
    """
    Cosinuslikhet mellan två normaliserade vektorer.

    Embedder.embed_query normaliserar vektorerna (normalize_embeddings=True
    i embeddings.py), så skalärprodukten ÄR cosinuslikheten.
    """
    if len(a) != len(b):
        raise ValueError(f"Olika dimensioner: {len(a)} vs {len(b)}")
    return sum(x * y for x, y in zip(a, b))


def measure_drift(
    utterance: str,
    qud_text: str | None,
    embedder: Embedder,
    threshold: float,
) -> DriftMeasurement | None:
    """
    Mät ämnesavstånd mellan aktuell yttring och aktiv QUD.

    Returnerar None om QUD saknas — i det fallet finns ingen drift
    att mäta (klassificeraren kan inte ha klassat som related_to_qud
    ändå, men vi avstår från att returnera en mätning).

    Returnerar annars en DriftMeasurement med likhet, tröskel, och
    en boolsk flagga drift_detected = (similarity < threshold).
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

    similarity = _cosine(vec_utterance, vec_qud)
    drift_detected = similarity < threshold

    return DriftMeasurement(
        similarity=round(similarity, 4),
        threshold=threshold,
        drift_detected=drift_detected,
    )