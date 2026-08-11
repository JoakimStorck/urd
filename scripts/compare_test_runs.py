#!/usr/bin/env python3
"""
Jämför två JSONL-diagnostikspår från `urd test`.

Användning:

    python -m scripts.compare_test_runs .urd/results/results_A.jsonl .urd/results/results_B.jsonl

Turer matchas på (sekvens, tur, fråga). Rapporten visar:

- flaggor som gått från fail till ok (förbättringar) och tvärtom
  (regressioner)
- ändrad klassificering (intent/substyle/operation)
- ändrade källdokument bakom svaret
- ändrat abstain-beteende
- tidsförändring totalt och per tur (större än tröskel)

Syftet är att göra fas 2-förändringarna ärligt utvärderbara: kör
baslinjen, gör EN förändring, kör igen, jämför. Utan detta blir
"det känns bättre" den enda metriken — och det var så systemet
hamnade i promptjusteringsspiralen.

Skriptet är avsiktligt beroendefritt (bara stdlib) så att det kan
köras var som helst, även utanför venv:en.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Tidsskillnad per tur (sekunder) som är värd att rapportera.
TIME_DELTA_THRESHOLD_S = 2.0


def load_run(path: Path) -> tuple[dict, dict[tuple, dict]]:
    """
    Läs ett JSONL-spår. Returnerar (run_meta, turer) där turer är
    en dict nycklad på (sekvens, turnummer, fråga).
    """
    meta: dict = {}
    turns: dict[tuple, dict] = {}

    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"{path}:{line_no}: ogiltig JSON hoppas över ({e})", file=sys.stderr)
                continue

            if record.get("type") == "run_meta":
                meta = record
            elif record.get("type") == "turn":
                key = (
                    record.get("sequence", "?"),
                    record.get("turn", 0),
                    record.get("question", ""),
                )
                turns[key] = record

    return meta, turns


def _flags_by_field(record: dict) -> dict[str, bool]:
    """Flaggutfall per fält för en tur: {fält: ok}."""
    return {
        f["field"]: bool(f["ok"])
        for f in record.get("flags", [])
        if "field" in f
    }


def _source_files(record: dict) -> list[str]:
    return [s.get("file_name", "?") for s in record.get("sources", [])]


def _classification(record: dict) -> str:
    cls = (record.get("debug") or {}).get("classification") or {}
    intent = cls.get("intent", "?")
    substyle = cls.get("substyle")
    operation = cls.get("question_operation")
    parts = [intent]
    if substyle:
        parts.append(substyle)
    if operation and operation != "direct_lookup":
        parts.append(operation)
    return "/".join(parts)


def _total_time(record: dict) -> float | None:
    timing = (record.get("debug") or {}).get("timing_s") or {}
    value = timing.get("total")
    return float(value) if isinstance(value, (int, float)) else None


def _abstained(record: dict) -> bool:
    return bool((record.get("debug") or {}).get("abstained", False))


def compare(old_path: Path, new_path: Path) -> int:
    old_meta, old_turns = load_run(old_path)
    new_meta, new_turns = load_run(new_path)

    print(f"Gammal körning: {old_path}")
    print(f"  tidpunkt: {old_meta.get('timestamp', '?')}  commit: {old_meta.get('git_commit', '?')}")
    print(f"Ny körning:     {new_path}")
    print(f"  tidpunkt: {new_meta.get('timestamp', '?')}  commit: {new_meta.get('git_commit', '?')}")
    print()

    common = sorted(set(old_turns) & set(new_turns))
    only_old = sorted(set(old_turns) - set(new_turns))
    only_new = sorted(set(new_turns) - set(old_turns))

    print(f"Gemensamma turer: {len(common)}")
    if only_old:
        print(f"Bara i gamla körningen: {len(only_old)}")
        for key in only_old:
            print(f"  - {key[0]} tur {key[1]}: {key[2][:60]}")
    if only_new:
        print(f"Bara i nya körningen: {len(only_new)} (nya testfall — ingår inte i jämförelsen)")
        for key in only_new:
            print(f"  + {key[0]} tur {key[1]}: {key[2][:60]}")
    print()

    improvements: list[str] = []
    regressions: list[str] = []
    class_changes: list[str] = []
    source_changes: list[str] = []
    abstain_changes: list[str] = []
    time_changes: list[str] = []

    old_times: list[float] = []
    new_times: list[float] = []

    for key in common:
        seq, turn, question = key
        label = f"{seq} tur {turn}: {question[:60]}"
        o, n = old_turns[key], new_turns[key]

        old_flags = _flags_by_field(o)
        new_flags = _flags_by_field(n)
        for field in sorted(set(old_flags) | set(new_flags)):
            was_ok = old_flags.get(field)
            is_ok = new_flags.get(field)
            if was_ok is None or is_ok is None:
                continue  # flaggan fanns bara i ena körningen
            if not was_ok and is_ok:
                improvements.append(f"{label}\n      {field}: fail → ok")
            elif was_ok and not is_ok:
                regressions.append(f"{label}\n      {field}: ok → FAIL")

        old_cls, new_cls = _classification(o), _classification(n)
        if old_cls != new_cls:
            class_changes.append(f"{label}\n      {old_cls} → {new_cls}")

        old_src, new_src = _source_files(o), _source_files(n)
        if old_src != new_src:
            source_changes.append(f"{label}\n      {old_src} → {new_src}")

        if _abstained(o) != _abstained(n):
            abstain_changes.append(
                f"{label}\n      abstain: {_abstained(o)} → {_abstained(n)}"
            )

        ot, nt = _total_time(o), _total_time(n)
        if ot is not None:
            old_times.append(ot)
        if nt is not None:
            new_times.append(nt)
        if ot is not None and nt is not None and abs(nt - ot) >= TIME_DELTA_THRESHOLD_S:
            time_changes.append(f"{label}\n      {ot:.1f}s → {nt:.1f}s")

    def _section(title: str, items: list[str]) -> None:
        print(f"{title} ({len(items)})")
        print("-" * len(f"{title} ({len(items)})"))
        for item in items:
            print(f"  {item}")
        if not items:
            print("  (inga)")
        print()

    _section("Förbättringar (flagga fail → ok)", improvements)
    _section("REGRESSIONER (flagga ok → fail)", regressions)
    _section("Ändrad klassificering", class_changes)
    _section("Ändrade källor bakom svaret", source_changes)
    _section("Ändrat abstain-beteende", abstain_changes)
    _section(f"Tidsförändring per tur (≥{TIME_DELTA_THRESHOLD_S:.0f}s)", time_changes)

    if old_times and new_times:
        print(
            f"Medeltid: {sum(old_times)/len(old_times):.1f}s → "
            f"{sum(new_times)/len(new_times):.1f}s"
        )

    # Exitkod 1 vid regressioner så att jämförelsen kan användas i skript.
    return 1 if regressions else 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Jämför två JSONL-diagnostikspår från 'urd test'.",
    )
    parser.add_argument("old", type=Path, help="Baslinjens JSONL-fil")
    parser.add_argument("new", type=Path, help="Den nya körningens JSONL-fil")
    args = parser.parse_args()

    for path in (args.old, args.new):
        if not path.exists():
            print(f"Filen finns inte: {path}", file=sys.stderr)
            raise SystemExit(2)

    raise SystemExit(compare(args.old, args.new))


if __name__ == "__main__":
    main()
