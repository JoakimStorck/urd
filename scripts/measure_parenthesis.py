#!/usr/bin/env python3
"""
Mät parentetiska appositioner: hur många är identiteter, hur många är
termekvivalenser?

Kör från instansens rot, med servern stoppad om den håller SQLite:

    python3 scripts/measure_parenthesis.py
    python3 scripts/measure_parenthesis.py --show 40
    python3 scripts/measure_parenthesis.py --term prefekt

BAKGRUND. Konstruktionen parentes:identitet blandar två slag av par:

    "Anna Andersson (prefekt)"      en PERSON i en roll
    "Head of School (prefekt)"      två NAMN på samma roll

Det andra slaget är beståndets egen tvåspråkiga termordlista och är
värdefullt — men det får inte kandidera på "vem är"-frågor. Uppmätt
2026-08-18: översättningsparet Head of School -> prefekt nådde relevans
0,312 och reserverade en passage med engelsk policytext i svaret på
"Vem är prefekt?".

VAD SKRIPTET GÖR. Det ändrar ingenting. Det tillämpar
grammar.looks_like_person_name på båda leden i varje parentetisk
apposition i Attest och redovisar hur klassningen SKULLE falla ut:

  - identitet   minst ett led ser ut som ett personnamn
  - översättning ingetdera ledet gör det

Syftet är att veta fördelningen och se falsklarmen INNAN regeln
införs. Predikatet är ortografiskt och passerar främmandespråkiga
titlar utan funktionsord ("Vice Chancellor"); just den restposten är
det mätningen ska visa storleken på.

NAMNPOLICY. Utdata innehåller personnamn ur beståndet. Den är avsedd
att läsas på maskinen, inte klistras in någon annanstans. Använd
--counts för att få enbart siffror.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import attest                      # noqa: E402
from app.grammar import looks_like_person_name  # noqa: E402


def klassa(subjekt: str, objekt: str) -> str:
    """Hur skulle paret falla ut med predikatet?"""
    if looks_like_person_name(subjekt) or looks_like_person_name(objekt):
        return "identitet"
    return "översättning"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    # NOLL SOM STANDARD. Utdata bär personnamn ur beståndet, och en
    # sammanställning av namn, roller och datum är känsligare än sina
    # delar. Exempel visas bara när de uttryckligen efterfrågas.
    ap.add_argument("--show", type=int, default=0,
                    help="Antal exempel per klass. Utdata bär PERSONNAMN.")
    ap.add_argument("--term", default=None,
                    help="Begränsa till par där något led matchar termen.")
    ap.add_argument("--counts", action="store_true",
                    help="Bara siffror, inga exempel (namnfri utdata).")
    args = ap.parse_args()

    conn = attest.connect()
    conn.row_factory = __import__("sqlite3").Row
    rader = list(conn.execute(
        "SELECT subject, object, construction, kind, source_path,"
        " COUNT(*) n"
        " FROM observations WHERE construction LIKE 'parentes:%'"
        " GROUP BY subject_key, object_key, construction"
        " ORDER BY n DESC"
    ))

    if args.term:
        t = args.term.lower()
        rader = [r for r in rader
                 if t in (r["subject"] or "").lower()
                 or t in (r["object"] or "").lower()]

    if not rader:
        print("Inga parentetiska appositioner i attest.db.")
        return 1

    per_konstruktion = Counter(r["construction"] for r in rader)
    utfall = Counter()
    ändrade: list = []
    behållna: list = []

    for r in rader:
        ny = klassa(r["subject"] or "", r["object"] or "")
        gammal = "identitet" if r["construction"].endswith("identitet") else "annat"
        utfall[(gammal, ny)] += 1
        if gammal == "identitet":
            (ändrade if ny != "identitet" else behållna).append(r)

    print("Parentetiska appositioner i Attest")
    print("==================================")
    print(f"Distinkta par: {len(rader)}")
    for k, n in per_konstruktion.most_common():
        print(f"  {k:<28} {n}")
    print("")

    id_totalt = sum(1 for r in rader if r["construction"].endswith("identitet"))
    if id_totalt:
        andel = 100.0 * len(ändrade) / id_totalt
        print(f"Av {id_totalt} par klassade som identitet skulle "
              f"{len(ändrade)} ({andel:.0f} %) bli översättning.")
        print(f"{len(behållna)} skulle förbli identitet.")
    print("")

    if args.counts or not args.show:
        return 0

    print("SKULLE BLI ÖVERSÄTTNING (inget led ser ut som personnamn)")
    print("---------------------------------------------------------")
    for r in ändrade[:args.show]:
        print(f"  {r['n']:>3}x  {r['subject']}  ->  {r['object']}")
    if len(ändrade) > args.show:
        print(f"  ... och {len(ändrade) - args.show} till")
    print("")

    print("SKULLE FÖRBLI IDENTITET (minst ett led ser ut som personnamn)")
    print("-------------------------------------------------------------")
    print("Läs denna lista efter FALSKLARM: främmandespråkiga titler utan")
    print("funktionsord passerar predikatet och hör egentligen till listan")
    print("ovan. Storleken på den resten avgör om regeln räcker.")
    for r in behållna[:args.show]:
        print(f"  {r['n']:>3}x  {r['subject']}  ->  {r['object']}")
    if len(behållna) > args.show:
        print(f"  ... och {len(behållna) - args.show} till")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
