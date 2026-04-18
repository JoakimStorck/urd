"""
Kontrollera evidensobjektsindexet.

Visar:
- Totalt antal evidensobjekt i __evidence-collection
- Fördelning per typ (table, figure, numbered_list, bullet_list)
- Fördelning per dokument (topp 10)
- Exempel på varje typ

Körs från projektroten med URD-servern AVSTÄNGD:

    python scripts/inspect_evidence.py
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.retrieval import RagService


def main() -> None:
    print("Laddar RagService...")
    rag = RagService()
    print("Klart.\n")

    all_evidence = rag.store.iter_all_evidence()

    if not all_evidence:
        print("INGA EVIDENSOBJEKT i indexet.")
        print("Kör 'urd ingest' för att indexera evidensobjekt.")
        print("(Ingest raderar och återskapar evidens per dokument.)")
        return

    print(f"Totalt antal evidensobjekt: {len(all_evidence)}")
    print()

    # Fördelning per typ
    type_counts: Counter = Counter()
    for hit in all_evidence:
        evidence_type = hit.metadata.document_type or "unknown"
        type_counts[evidence_type] += 1

    print("Fördelning per typ:")
    for ev_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {ev_type:20s}  {count}")
    print()

    # Fördelning per dokument
    docs: Counter = Counter()
    for hit in all_evidence:
        docs[hit.metadata.file_name] += 1

    print(f"Antal dokument med evidens: {len(docs)}")
    print()
    print("Topp 10 dokument med flest evidensobjekt:")
    for file_name, count in docs.most_common(10):
        print(f"  {count:3d}  {file_name}")
    print()

    # Exempel per typ
    by_type: dict[str, list] = defaultdict(list)
    for hit in all_evidence:
        ev_type = hit.metadata.document_type or "unknown"
        by_type[ev_type].append(hit)

    print("=" * 70)
    print("EXEMPEL PER TYP")
    print("=" * 70)

    for ev_type in sorted(by_type.keys()):
        examples = by_type[ev_type]
        print()
        print(f"--- {ev_type} ({len(examples)} st) ---")
        # Visa upp till 2 exempel
        for i, hit in enumerate(examples[:2], start=1):
            print(f"\n  Exempel {i}/{min(2, len(examples))}:")
            print(f"    Dokument: {hit.metadata.file_name}")
            print(f"    Sektion:  {hit.metadata.section_title}")
            # Begränsa textlängden för läsbarhet
            text_preview = hit.text[:500]
            if len(hit.text) > 500:
                text_preview += " ... [avkortat]"
            # Indentera för läsbarhet
            for line in text_preview.splitlines():
                print(f"    | {line}")

    print()
    print("=" * 70)
    print("KLART")
    print("=" * 70)


if __name__ == "__main__":
    main()