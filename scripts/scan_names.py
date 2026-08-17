#!/usr/bin/env python3
"""
Kartlägg personnamn i kod, kommentarer och konfiguration.

Kör från instansens rot:

    python3 scripts/scan_names.py
    python3 scripts/scan_names.py --context 2
    python3 scripts/scan_names.py --json > /tmp/namn.json

Skriptet letar efter namn på två sätt:

1. KÄNDA NAMN — de som förekommit i arbetet och som vi vet ligger i
   kommentarer. Exakt matchning, hög träffsäkerhet.

2. MÖNSTER — två eller flera versalinledda ord i följd inuti en
   sträng eller kommentar. Fångar namn vi inte listat, till priset av
   falsklarm på saker som "Vice Chancellor" och "Head of School".

Endast spårade filer i git genomsöks, och binärformat hoppas över.
docs/ och .urd/ ignoreras — de är instansdata och ska inte vara i
versionshantering.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

# Namn som förekommit i arbetet 2026-08-14/15. Efternamn räcker som
# sökterm; förnamnen fångas av kontextutskriften.
KNOWN = [
    "Bodegrim", "Wäckelgård", "Storck", "Skogberg", "Skogbergs",
    "Karlsson", "Moudud", "Alam", "Rönnelid", "Forsman", "Englund",
    "Carling", "Olsmats", "Lenne", "Hägglund", "Augusto", "Sonne",
    "Rynoson", "Garman", "Tosteby", "Kallin", "Younes", "Nyberg",
    "Fleyeh", "Nordström", "Myhren", "Ersson", "Olsson", "Rapp",
    "Römsing", "Mosveen", "Saeed", "Sadeghian", "Psimopoulos",
    "Viklund", "Janols", "Zhang", "Zhao", "Fiedler",
    # "Person" utelämnat: vanligt substantiv, gav bara falsklarm på
    # "samma person", "per person" i kommentarer.
    "Lindgren", "Håkansson", "Bales", "Teledahl", "Johansson",
    "Rybarczyk", "Metselaar", "Höglund", "Haglund", "Öijwall",
    "Ranhagen", "Surreddi", "Jayamani", "Rappfors", "Hedback",
    "Stolpe", "Vixner", "Vänje", "Gradén", "Macuchova", "Kenger",
    "Walla", "Eklund", "Fjelkner", "Broman", "Dodou", "Mårtensson",
    "Vide", "Fors", "Södergren", "Bergman",
]

# Två eller fler versalinledda ord i följd. Ger falsklarm på
# institutionella termer, som filtreras bort nedan.
NAME_PATTERN = re.compile(
    r"\b([A-ZÅÄÖ][a-zåäöé\-]{1,}(?:\s+[A-ZÅÄÖ][a-zåäöé\-]{1,}){1,3})\b"
)

# Termer som ser ut som namn men inte är det.
NOT_NAMES = {
    "vice chancellor", "head of school", "deputy head", "high commissioner",
    "dalarna university", "högskolan dalarna", "information och",
    "vid protokollet", "region dalarna", "science park", "working papers",
    "data science", "academic data", "new history", "for the", "of the",
    "att göra", "det som", "den som", "vid ingest", "vid retrieval",
    "se ovan", "not found", "true false", "python dict", "type error",
}

SKIP_SUFFIXES = {
    ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".ico", ".woff", ".woff2",
    ".ttf", ".zip", ".gz", ".db", ".sqlite", ".pyc", ".patch",
}

SKIP_DIRS = {"docs", ".urd", ".git", ".venv", "__pycache__", "node_modules"}


def tracked_files() -> list[Path]:
    """Endast filer som git spårar — det är dem som riskerar spridning."""
    try:
        out = subprocess.run(
            ["git", "ls-files"], capture_output=True, text=True, check=True
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Kunde inte köra 'git ls-files' — kör från repots rot.",
              file=sys.stderr)
        raise SystemExit(2)

    files = []
    for line in out.splitlines():
        p = Path(line)
        if p.suffix.lower() in SKIP_SUFFIXES:
            continue
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if p.exists():
            files.append(p)
    return files


def scan(files: list[Path], context: int) -> list[dict]:
    hits: list[dict] = []
    # Ordgränser: utan dem matchar "Rapp" i "Rapportera" och
    # "Person" i "personnamn".
    known_re = re.compile(
        r"\b(?:" + "|".join(re.escape(n) for n in KNOWN) + r")\b", re.I
    )

    # Skriptets egen namnlista är inte en läcka. Jämförelse på FILNAMN
    # och inte på resolverad sökväg: git ls-files ger relativa vägar,
    # och __file__ kan vara absolut eller relativ beroende på hur
    # skriptet startats — uppmätt 2026-08-17 rapporterade skriptet sig
    # självt när det låg i scripts/.
    self_name = Path(__file__).name
    for path in files:
        if path.name == self_name:
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue

        for i, line in enumerate(lines):
            found: list[tuple[str, str]] = []

            for m in known_re.finditer(line):
                found.append(("känt", m.group(0)))

            for m in NAME_PATTERN.finditer(line):
                cand = m.group(1)
                if cand.lower() in NOT_NAMES:
                    continue
                if any(k.lower() in cand.lower() for k in KNOWN):
                    continue          # redan rapporterat som känt
                found.append(("mönster", cand))

            if not found:
                continue

            hits.append({
                "file": str(path),
                "line": i + 1,
                "kinds": sorted({k for k, _ in found}),
                "names": sorted({v for _, v in found}),
                "text": line.strip()[:160],
                "context": [
                    lines[j].strip()[:160]
                    for j in range(max(0, i - context), min(len(lines), i + context + 1))
                ] if context else [],
            })
    return hits


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--context", type=int, default=0,
                    help="Antal rader kontext runt varje träff.")
    ap.add_argument("--json", action="store_true",
                    help="Skriv ut som JSON i stället för text.")
    ap.add_argument("--only-known", action="store_true",
                    help="Rapportera bara kända namn, inte mönsterträffar.")
    args = ap.parse_args()

    files = tracked_files()
    hits = scan(files, args.context)
    if args.only_known:
        hits = [h for h in hits if "känt" in h["kinds"]]

    if args.json:
        print(json.dumps(hits, ensure_ascii=False, indent=2))
        return

    if not hits:
        print(f"Inga namnträffar i {len(files)} spårade filer.")
        return

    by_file: dict[str, list[dict]] = {}
    for h in hits:
        by_file.setdefault(h["file"], []).append(h)

    for path, rows in sorted(by_file.items()):
        print(f"\n=== {path}  ({len(rows)} rader) ===")
        for r in rows:
            marker = "!" if "känt" in r["kinds"] else "?"
            print(f"  {marker} rad {r['line']}: {', '.join(r['names'])}")
            print(f"      {r['text']}")
            for c in r["context"]:
                if c != r["text"]:
                    print(f"      | {c}")

    known = sum(1 for h in hits if "känt" in h["kinds"])
    print(f"\n{len(hits)} rader i {len(by_file)} filer.")
    print(f"  ! kända namn:      {known}")
    print(f"  ? mönsterträffar:  {len(hits) - known}  (kontrollera manuellt)")
    print(f"\nGenomsökta filer: {len(files)} (docs/ och .urd/ undantagna)")


if __name__ == "__main__":
    main()
