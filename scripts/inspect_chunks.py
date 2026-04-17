"""
Inspektera chunkarna för proprefekt-dokumentet.

Vi vill veta om chunken 'Process för tillsättning' innehåller alla
16 steg eller bara en delmängd, och om 'Tidsramar' finns som
egen chunk alls. Det avgör om B1-syntesens brister beror på prompt
eller på chunkning.

Körs från projektroten med URD-servern AVSTÄNGD:

    python scripts/inspect_chunks.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.retrieval import RagService


def main() -> None:
    print("Laddar RagService...")
    rag = RagService()
    print("Klart.\n")

    # Leta efter proprefekt-dokumentet — prova några vanliga path-varianter
    candidate_paths = [
        "./docs/regler-for-tillsattning-av-proprefekt.pdf",
        "docs/regler-for-tillsattning-av-proprefekt.pdf",
        "regler-for-tillsattning-av-proprefekt.pdf",
    ]

    chunks = []
    found_path = None
    for p in candidate_paths:
        chunks = rag.bm25_index.get_chunks_by_source(p)
        if chunks:
            found_path = p
            break

    if not chunks:
        # Fallback: bred sökning i BM25-indexet
        print("Hittade inga chunkar via direkta path-varianter. Söker brett...")
        hits = rag.bm25_index.top_k(
            "proprefekt tillsättning kollegial kommitté",
            k=100,
        )
        seen = set()
        for h in hits:
            if (
                "proprefekt" in h.metadata.file_name.lower()
                and h.chunk_id not in seen
            ):
                chunks.append(h)
                seen.add(h.chunk_id)
        if chunks:
            found_path = chunks[0].metadata.source_path
            print(f"Hittade {len(chunks)} chunkar via sökning. Path: {found_path}")

    if not chunks:
        print("KUNDE INTE HITTA DOKUMENTET. Sluta.")
        return

    print(f"Path: {found_path}")
    print(f"Antal chunkar: {len(chunks)}")
    print()

    # Visa varje chunk i sin helhet
    for i, c in enumerate(chunks, start=1):
        print("=" * 70)
        print(f"CHUNK {i}/{len(chunks)}")
        print(f"  section_title:   {c.metadata.section_title}")
        print(f"  document_title:  {c.metadata.document_title}")
        print(f"  document_type:   {c.metadata.document_type}")
        print(f"  chunk_id:        {c.chunk_id}")
        print(f"  text_len:        {len(c.text)} tecken")
        print("-" * 70)
        print(c.text)
        print()


if __name__ == "__main__":
    main()