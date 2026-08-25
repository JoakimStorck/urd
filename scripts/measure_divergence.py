#!/usr/bin/env python3
"""
Mät divergensen mellan beslutstabellens dom och systemets faktiska svar.

Kör från instansens rot, mot ett körningsspår från urd test:

    python3 scripts/measure_divergence.py .urd/results/results_X.jsonl
    python3 scripts/measure_divergence.py --show 10 <spår.jsonl>

GRINDEN (white paper 3.0, införandevägen). Tabellen i
deliberation_table.yaml dömer i efterhand: vilket utfall borde turen
ha fått, givet operationens löfte och vad svaret faktiskt gjorde?

  Divergens NOLL          -> lagret är dekoration; bygg inte vidare
  Divergens på FEL ställen -> tabellen är felkalibrerad; skriv om den
  Divergens på RÄTT ställen (substitutionssvaren) -> klartecken att ge
  lagret makt, utfallsklass för utfallsklass

Domen är RETRODIKTIV: den läser spåret, inte servern. Det är avsiktligt
— åtagandet och bindningarna loggas redan per tur, så grinden kan mätas
utan att en rad i svarskedjan ändras. Först när mätningen sagt sitt
flyttar domen in i drift.

UTFALLSKLASSNING AV DET FAKTISKA SVARET, per tur:

  avstar     svaret är abstain-mallens (ingen syntes gjordes, eller
             svaret bär abstain-markören)
  namnger    en verifierbar bindning där något led är namnlikt
             (looks_like_person_name — predikatet som UTESLUTER namn
             används här för att KRÄVA dem, vilket är dess starka
             riktning)
  beskriver  övriga svar med innehåll

Klassningen är medvetet grov: den ska skilja substitutionssvar från
namngivande svar, inte förstå svaren. Det den inte kan avgöra — om en
beskrivning var UTTRYCKLIGEN markerad som "beståndet namnger ingen" —
finns inte som systemutfall ännu, och därför räknas varje beskrivning
på en innehavarfråga som divergent. Det är inte ett mätfel utan
mätobjektet.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import grammar  # noqa: E402

# Reservväg för äldre spår utan abstained-fält. Spåret bär sedan länge
# "abstained" som loggat FAKTUM, och det är det som läses först —
# första versionen av detta skript matchade strängar mot mallen och
# missade den ("inget TYDLIGT stöd"), med noll uppmätta avståenden
# över ett spår som bevisligen innehöll dem. Att gissa ur texten när
# sanningen ligger loggad bredvid är samma fel som strängassertionerna
# i batteriet.
_ABSTAIN_MARKERS = (
    "hittar inget tydligt stöd",
    "hittar inget stöd",
    "hittar ingen information",
)

# Personform kontra funktionsform, grammatiskt skild: "vem ÄR X" —
# kopula med rollpredikat — lovar en PERSON, medan "vem BESLUTAR om X"
# — agentverb — lovar en FUNKTION, och "prefekten beslutar" är då ett
# fullständigt svar. Uppmätt 2026-08-25: tre av sex divergenta var
# funktionsfrågor korrekt besvarade med funktion. Sluten kopulamängd,
# inte en verblista över domänen.
_COPULAS = {"är", "var", "blir", "heter", "vart"}
_HOLDER_PHRASES = ("uppdraget som", "rollen som", "utsedd till")

# Intenter vars svar prövar ett TIDIGARE påstående och inte ska dömas
# som nya innehavarfrågor.
_EXCLUDED_INTENTS = {"verification_or_challenge", "social_or_meta"}


def _is_person_form(question: str) -> bool:
    """Lovar frågan en person, inte bara en funktion?"""
    låg = question.lower()
    if any(f in låg for f in _HOLDER_PHRASES):
        return True
    ord_ = låg.replace("?", " ").split()
    for i, w in enumerate(ord_):
        if w in ("vem", "vilka") and i + 1 < len(ord_):
            return ord_[i + 1] in _COPULAS
    # "Vilken roll har <namn>" — omvänd innehavarfråga: given person,
    # efterfrågad roll. Namnet i frågan gör löftet personbundet.
    if låg.startswith("vilken roll") or "vilken roll har" in låg:
        return True
    return False

# Operationer vars löfte är en namngiven innehavare.
_NAMING_OPERATIONS = {"entity_lookup", "entity_aggregation"}


def _load_table(path: Path) -> dict:
    import yaml
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "operationer" not in data:
        raise SystemExit(f"Ogiltig beslutstabell: {path}")
    return data


def _turns(jsonl: Path):
    for rad in jsonl.read_text(encoding="utf-8").splitlines():
        try:
            post = json.loads(rad)
        except json.JSONDecodeError:
            continue
        debug = post.get("debug") or {}
        if "commitment" not in debug:
            continue  # metadatapost, inte en tur
        yield post


def _binder_namn(bindning: dict) -> bool:
    """Är något led i bindningen namnlikt?"""
    if not bindning.get("verifierbar"):
        return False
    return (grammar.looks_like_person_name(bindning.get("subjekt", ""))
            or grammar.looks_like_person_name(bindning.get("predikat", "")))


def classify_actual(post: dict) -> tuple[str, dict]:
    """Vad gjorde svaret? (utfall, underlag för rapporten)."""
    svar = (post.get("answer") or "").strip()
    debug = post.get("debug") or {}
    claims = ((debug.get("synthesis") or {}).get("answer_claims")) or {}
    bindningar = claims.get("bindningar", [])

    if debug.get("abstained") is True:
        return "avstar", {}
    låg = svar.lower()
    if not svar or any(m in låg for m in _ABSTAIN_MARKERS):
        return "avstar", {}

    namngivande = [b for b in bindningar if _binder_namn(b)]
    overifierbara = [b for b in bindningar if not b.get("verifierbar")]
    underlag = {
        "namngivande": len(namngivande),
        "overifierbara": len(overifierbara),
    }
    if namngivande:
        return "namnger", underlag
    return "beskriver", underlag


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("jsonl", help="Körningsspår från urd test.")
    ap.add_argument("--table", default="deliberation_table.yaml")
    ap.add_argument("--show", type=int, default=0,
                    help="Visa divergenta turer (frågetext kan bära namn).")
    args = ap.parse_args()

    tabell = _load_table(Path(args.table))
    operationer = tabell["operationer"]

    per_op: Counter = Counter()
    utfall_räknare: Counter = Counter()
    divergenta: list[dict] = []
    fynd_overifierbara = 0
    n = 0

    for post in _turns(Path(args.jsonl)):
        n += 1
        commitment = post["debug"]["commitment"]
        op = commitment.get("operation") or "direct_lookup"
        intent = commitment.get("intent")
        per_op[op] += 1

        faktiskt, underlag = classify_actual(post)
        utfall_räknare[faktiskt] += 1
        fynd_overifierbara += underlag.get("overifierbara", 0)

        if op not in operationer:
            divergenta.append({
                "sekvens": post.get("sequence"), "tur": post.get("turn"),
                "skäl": f"operationen {op!r} saknas i tabellen",
                "operation": op, "faktiskt": faktiskt,
            })
            continue

        # Substitutionsmåttet: en PERSONFORMAD innehavarfråga vars
        # svar beskriver utan att namnge och utan att avstå.
        # Funktionsfrågor ("vem beslutar om X") frias: funktionen är
        # där ett fullständigt svar. Verifikationsturer prövar ett
        # tidigare påstående och döms inte som nya innehavarfrågor.
        if (op in _NAMING_OPERATIONS
                and intent not in _EXCLUDED_INTENTS
                and _is_person_form(post.get("question", ""))
                and faktiskt == "beskriver"):
            divergenta.append({
                "sekvens": post.get("sequence"), "tur": post.get("turn"),
                "skäl": "innehavarfråga besvarad med beskrivning",
                "operation": op, "faktiskt": faktiskt,
                "fraga": post.get("question", ""),
            })

    print(f"Divergensmätning över {n} turer  ({Path(args.jsonl).name})")
    print("=" * 60)
    print("Turer per operation:")
    for op, c in per_op.most_common():
        print(f"  {op:<22} {c}")
    print("")
    print("Faktiska utfall:")
    for u, c in utfall_räknare.most_common():
        print(f"  {u:<22} {c}")
    print("")
    print(f"Divergenta turer:        {len(divergenta)}")
    print(f"Overifierbara bindningar: {fynd_overifierbara}  (prövningsfynd, "
          f"oavsett operation)")

    if args.show and divergenta:
        print("")
        print("Divergenta turer (frågetext kan bära namn — läs lokalt):")
        for d in divergenta[:args.show]:
            print(f"  {d.get('sekvens')} tur {d.get('tur')}: "
                  f"{d['skäl']}  [{d['operation']} -> {d['faktiskt']}]")
            if d.get("fraga"):
                print(f"    fråga: {d['fraga']}")

    if n and not divergenta:
        print("")
        print("Divergens noll: med denna tabell och detta spår skulle "
              "lagret inte ha ändrat något svar. Antingen är batteriet "
              "utan substitutionsfall — eller lagret dekoration. Avgör "
              "med ett spår som innehåller innehavarfrågor.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
