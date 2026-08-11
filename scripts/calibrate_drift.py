"""
Kalibrera qud_drift_threshold mot den faktiska embeddingmodellen.

QUD-driftskyddet jämför embeddinglikhet mellan aktuell yttring och
den aktiva huvudfrågan. Tröskeln är bara meningsfull om den är
kalibrerad mot den likhetsfördelning modellen faktiskt producerar —
och den fördelningen ändras när t.ex. E5-prefix slås på eller
modellen byts.

Skriptet mäter likheten för ett antal frågepar — besläktade (samma
tråd, ska INTE trigga drift) och obesläktade (ämnesbyte, SKA trigga)
— och föreslår en tröskel mitt emellan fördelningarna.

Körs från projektroten (servern kan vara igång — bara embeddings
används, ingen Qdrant):

    python -m scripts.calibrate_drift

Egna par kan läggas till i en JSON-fil och anges med --file:

    [
      {"a": "fråga 1", "b": "fråga 2", "related": true},
      ...
    ]

Sätt sedan tröskeln:

    urd config set qud_drift_threshold <föreslaget värde>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Inbyggda kalibreringspar, hämtade från domänen (styrdokument vid
# lärosäte) och från testbatteriets faktiska sekvenser. "related"
# betyder: samma samtalstråd — driftskyddet ska INTE trigga.
DEFAULT_PAIRS: list[dict] = [
    # Besläktade — subquestions/broadening inom samma tråd
    {"a": "Vad behöver jag göra om jag ska skicka in en extern forskningsansökan?",
     "b": "Vad gäller för medfinansiering?", "related": True},
    {"a": "Vad behöver jag göra om jag ska skicka in en extern forskningsansökan?",
     "b": "Vem fattar beslut om inskickande?", "related": True},
    {"a": "Vilken process finns för att tillsätta en ny proprefekt?",
     "b": "Hur lång är mandatperioden?", "related": True},
    {"a": "Vad krävs för att antas till forskarutbildning?",
     "b": "Berätta mer om studiefinansieringen.", "related": True},
    {"a": "Vilka behörighetskrav gäller för att anställas som lektor?",
     "b": "Och för professorer då?", "related": True},
    {"a": "Vad gäller vid disputation?",
     "b": "Vilka regler gäller för licentiatseminarium?", "related": True},
    {"a": "Vilket arvode får en opponent vid en disputation?",
     "b": "Vilka anvisningar gäller för halvtidsseminarier?", "related": True},

    # Obesläktade — ämnesbyten; driftskyddet SKA trigga
    {"a": "Jag vill rekrytera en adjungerad lektor. Vilken rutin finns?",
     "b": "Vilka lokala rutiner gäller på IIT före beslut om extern forskningsansökan?",
     "related": False},
    {"a": "Vad krävs för att antas till forskarutbildning?",
     "b": "Vilka regler gäller för flaggning?", "related": False},
    {"a": "Vilken process finns för att tillsätta en ny proprefekt?",
     "b": "Vilka arvoden gäller för opponent och licentiatgranskare?", "related": False},
    {"a": "Vad gäller för medfinansiering av forskningsprojekt?",
     "b": "Vilka labbansvariga finns på IIT?", "related": False},
    {"a": "Vilka behörighetskrav gäller för universitetslektorer?",
     "b": "Vilka regler gäller för möten och resor?", "related": False},
    {"a": "Vad gäller vid disputation?",
     "b": "Hur hanteras rehabilitering vid sjukskrivning?", "related": False},
]


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Kalibrera qud_drift_threshold mot embeddingmodellen.",
    )
    parser.add_argument("--file", type=Path, help="JSON-fil med extra frågepar")
    args = parser.parse_args()

    pairs = list(DEFAULT_PAIRS)
    if args.file:
        with open(args.file, encoding="utf-8") as f:
            extra = json.load(f)
        pairs.extend(extra)

    print("Laddar embeddingmodellen...")
    from app.config import settings
    from app.embeddings import Embedder

    embedder = Embedder()
    print(f"Modell: {settings.embedding_model} "
          f"(E5-prefix: {'på' if embedder.use_e5_prefixes else 'av'})")
    print(f"Nuvarande qud_drift_threshold: {settings.qud_drift_threshold}")
    print()

    results: list[tuple[float, bool, dict]] = []
    for pair in pairs:
        va = embedder.embed_query(pair["a"])
        vb = embedder.embed_query(pair["b"])
        sim = _cosine(va, vb)
        results.append((sim, bool(pair["related"]), pair))

    results.sort(key=lambda r: r[0], reverse=True)
    print(f"{'likhet':>8}  {'typ':<12} par")
    print("-" * 72)
    for sim, related, pair in results:
        label = "besläktad" if related else "ÄMNESBYTE"
        print(f"{sim:>8.4f}  {label:<12} {pair['a'][:38]!r} ↔ {pair['b'][:38]!r}")
    print()

    related_sims = [s for s, r, _ in results if r]
    unrelated_sims = [s for s, r, _ in results if not r]

    if not related_sims or not unrelated_sims:
        print("Behöver både besläktade och obesläktade par för kalibrering.")
        raise SystemExit(1)

    min_related = min(related_sims)
    max_unrelated = max(unrelated_sims)

    print(f"Lägsta likhet bland besläktade par:   {min_related:.4f}")
    print(f"Högsta likhet bland ämnesbyten:       {max_unrelated:.4f}")
    print()

    if min_related > max_unrelated:
        suggested = round((min_related + max_unrelated) / 2, 3)
        margin = min_related - max_unrelated
        print(f"Fördelningarna är separerade (marginal {margin:.4f}).")
        print(f"Föreslagen tröskel: {suggested}")
        print()
        print(f"Sätt den med:  urd config set qud_drift_threshold {suggested}")
    else:
        # Överlapp — välj tröskel som minimerar felklassningar och
        # redovisa vilka par som hamnar fel.
        candidates = sorted({s for s, _, _ in results})
        best_t, best_errors = None, len(results) + 1
        for t in candidates:
            errors = sum(1 for s, r, _ in results if (s < t) == r)
            if errors < best_errors:
                best_t, best_errors = t, errors
        print("VARNING: fördelningarna överlappar — ingen tröskel skiljer dem rent.")
        print(f"Bästa kompromiss: {best_t:.3f} ({best_errors} felklassade par av {len(results)}).")
        print("Granska paren ovan; överväg att komplettera driftmåttet med")
        print("jämförelse mot aktiva dokument (fas 4 i åtgärdsplanen).")


if __name__ == "__main__":
    main()
