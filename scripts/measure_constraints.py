#!/usr/bin/env python3
"""
Mät frågornas inskränkningar: vad avgränsar de belägg som får användas?

Kör från instansens rot:

    python3 scripts/measure_constraints.py
    python3 scripts/measure_constraints.py --jsonl .urd/results/results_X.jsonl
    python3 scripts/measure_constraints.py --show 30

BAKGRUND. Deliberationslagrets ram (white paper 3.0) ska avgöra vilka
belägg som är TILLÅTLIGA innan något vägs. Ramen får inte byggas mot de
inskränkningsslag som råkat förekomma i utvecklingsarbetet — årtal och
organisationsnamn — utan mot frågornas verkliga fördelning. Detta skript
mäter den fördelningen, innan mekanismen byggs.

VAD SOM MÄTS, per fråga:

  1. GRAMMATISKA INSKRÄNKNINGAR ur dependensparsen: prepositionsfraser
     och adverbial knutna till frågan (obl/nmod med case-barn, advmod).
     Detta är den strukturella definitionen — formen, inte en typlista.
  2. TIDSUTTRYCK på textnivå: årtal och tidsdeiktika ("då", "nuvarande",
     "senaste"). Deiktika kan inte upplösas ur frågan ensam — de pekar
     på samtalet, och räknas därför separat som ARVSBEROENDE.
  3. UNDERSPECIFICERADE TURER: följdfrågor med få innehållsord, vars ram
     måste ÄRVAS från QUD:n om den alls ska finnas.

Utan stanza körs bara textnivån, och det sägs i utdata — en mätning som
tyst mäter mindre än den påstår är värre än ingen mätning.

Utdata är aggregerad som standard. --show visar fraserna; frågor kan
bära personnamn, så visad utdata ska läsas på maskinen.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import grammar  # noqa: E402

# Årtal och intervall: "2024", "2023/2024", "2023-2025".
_YEAR = re.compile(r"\b(?:19|20)\d{2}(?:\s*[/–-]\s*(?:19|20)?\d{2})?\b")

# Tidsdeiktika: uttryck vars referent ligger i samtalet eller i
# yttrandeögonblicket, inte i frågetexten. Sluten funktionsordsklass.
_DEICTIC = re.compile(
    r"\b(då|nu|nuvarande|dåvarande|senaste?|tidigare|framöver|"
    r"i ?dag|idag|för närvarande|just nu|numera|längre)\b",
    re.IGNORECASE,
)

# Frågeord och funktionsord som inte räknas som innehåll när
# underspecificering bedöms.
_NONCONTENT = {
    "vem", "vad", "vilka", "vilken", "vilket", "hur", "när", "var",
    "varför", "är", "var", "blir", "finns", "gäller", "ska", "kan",
    "och", "eller", "men", "om", "att", "det", "den", "de", "har",
    "haft", "någon", "något", "några", "annan", "annat", "andra",
    "mer", "också", "berätta", "utveckla", "tack", "hej", "ja", "nej",
    "samma", "detta", "denna", "dessa",
}


def _questions_from_battery(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    seqs = data.get("sequences", data if isinstance(data, list) else [])
    for s in seqs:
        namn = s.get("name", "?")
        for i, turn in enumerate(s.get("turns", []), start=1):
            q = turn.get("question") or ""
            if q:
                yield namn, i, q


def _questions_from_jsonl(path: Path):
    for rad in path.read_text(encoding="utf-8").splitlines():
        try:
            post = json.loads(rad)
        except json.JSONDecodeError:
            continue
        q = post.get("question") or post.get("q") or ""
        namn = post.get("sequence") or post.get("seq") or "jsonl"
        tur = post.get("turn") or post.get("turn_index") or 0
        if q:
            yield str(namn), int(tur) if str(tur).isdigit() else 0, q


def _parse_constraints(q: str) -> list[tuple[str, str]]:
    """
    (huvudmarkör, fras) för varje grammatisk inskränkning i frågan.

    Prepositionsfraser hämtas som obl/nmod med case-barn; markören är
    prepositionens lemma. Adverbial som advmod med ADV. Formen avgör —
    ingen lista över vilka slag som "räknas".
    """
    pipeline = grammar._get_pipeline()
    if pipeline is None:
        return []
    ut: list[tuple[str, str]] = []
    try:
        doc = pipeline(q)
    except Exception:
        return []
    for mening in doc.sentences:
        ord_efter_id = {w.id: w for w in mening.words}
        barn: dict[int, list] = {}
        for w in mening.words:
            barn.setdefault(w.head, []).append(w)
        for w in mening.words:
            if w.deprel in ("obl", "obl:tmod", "nmod"):
                case = [b for b in barn.get(w.id, [])
                        if b.deprel == "case"]
                if not case:
                    continue
                markör = case[0].lemma.lower()
                # Frasen: markören + huvudordet + dess platta barn.
                led = [case[0].text, w.text] + [
                    b.text for b in barn.get(w.id, [])
                    if b.deprel in ("flat", "flat:name", "compound", "amod")
                ]
                ut.append((markör, " ".join(led)))
            elif w.deprel == "advmod" and w.upos == "ADV":
                if w.lemma.lower() not in ("inte", "också", "bara", "ju"):
                    ut.append(("adverbial", w.text))
    return ut


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--battery", default="test/questions.json",
                    help="Batterifil (default test/questions.json, "
                         "faller tillbaka på questions.example.json).")
    ap.add_argument("--jsonl", default=None,
                    help="Även frågor ur ett JSONL-körningsspår.")
    ap.add_argument("--show", type=int, default=0,
                    help="Visa fraser (kan bära personnamn).")
    args = ap.parse_args()

    bpath = Path(args.battery)
    if not bpath.exists():
        bpath = Path("test/questions.example.json")
    frågor = list(_questions_from_battery(bpath))
    if args.jsonl:
        frågor += list(_questions_from_jsonl(Path(args.jsonl)))
    if not frågor:
        print("Inga frågor funna.")
        return 1

    har_parse = grammar.is_available()
    if not har_parse:
        print("OBS: stanza saknas — ENDAST textnivån mäts. Kör på "
              "instansen för den grammatiska nivån.\n")

    markörer: Counter = Counter()
    fraser: Counter = Counter()
    n_år = n_deiktisk = n_underspec = n_med_inskr = 0
    underspec_lista: list[tuple[str, int, str]] = []
    deiktisk_lista: list[tuple[str, int, str]] = []

    for namn, tur, q in frågor:
        inskr = _parse_constraints(q) if har_parse else []
        år = _YEAR.findall(q)
        deikt = _DEICTIC.findall(q)
        innehåll = [
            w for w in re.findall(r"[\wåäöÅÄÖ-]+", q.lower())
            if w not in _NONCONTENT and len(w) > 2 and not w.isdigit()
        ]
        if inskr or år:
            n_med_inskr += 1
        if år:
            n_år += 1
        if deikt:
            n_deiktisk += 1
            deiktisk_lista.append((namn, tur, q))
        if tur > 1 and len(innehåll) <= 1:
            n_underspec += 1
            underspec_lista.append((namn, tur, q))
        for markör, fras in inskr:
            markörer[markör] += 1
            fraser[f"{markör}: {fras}"] += 1

    n = len(frågor)
    print(f"Inskränkningar i {n} frågor ({bpath.name}"
          f"{' + jsonl' if args.jsonl else ''})")
    print("=" * 56)
    print(f"Med uttryckt inskränkning (grammatisk eller årtal): "
          f"{n_med_inskr}/{n}")
    print(f"Med årtal:                    {n_år}")
    print(f"Med tidsdeiktikon (ARVSBEROENDE): {n_deiktisk}")
    print(f"Underspecificerade följdturer (ram måste ÄRVAS): {n_underspec}")
    print("")
    if har_parse:
        print("Markörfördelning (grammatisk nivå):")
        for m, c in markörer.most_common(25):
            print(f"  {m:<14} {c}")
    if args.show:
        print("")
        print("Underspecificerade följdturer:")
        for namn, tur, q in underspec_lista[:args.show]:
            print(f"  {namn} tur {tur}: {q}")
        print("")
        print("Tidsdeiktiska frågor:")
        for namn, tur, q in deiktisk_lista[:args.show]:
            print(f"  {namn} tur {tur}: {q}")
        if har_parse:
            print("")
            print("Vanligaste inskränkningsfraser:")
            for fras, c in fraser.most_common(args.show):
                print(f"  {c:>3}x  {fras}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
