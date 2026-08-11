"""
Inspektera vad indexet innehåller för ett dokument — och diagnostisera
varför en fråga inte når det.

Ersätter det hårdkodade inspect_chunks.py (som bara kände till
proprefekt-dokumentet) med ett generellt verktyg. Detta är ett första
steg mot 'urd docs inspect' ur white paper.

Körs från projektroten med URD-servern AVSTÄNGD (inbäddad Qdrant
tillåter bara en process):

    python -m scripts.inspect_doc "reviderad rutin för beslut"
    python -m scripts.inspect_doc "reviderad rutin" --full
    python -m scripts.inspect_doc "reviderad rutin" --evidence
    python -m scripts.inspect_doc "reviderad rutin" \
        --question "Vilka lokala rutiner gäller på IIT före beslut om extern forskningsansökan?"

Med --question körs en retrievaldiagnos som svarar på tre frågor:

  1. Kommer dokumentet in i den SEMANTISKA kandidatpoolen (topp-15)?
  2. Kommer det in via BM25 (topp-10, med synonymexpansion)?
  3. Hur bedömer CROSS-ENCODERN dokumentets chunkar mot frågan?

Ett dokument som saknas i 1 och 2 når aldrig rerankern — felet sitter
i kandidatinsamlingen (t.ex. språkgap eller embeddingkvalitet). Ett
dokument som kommer in men får negativa CE-scores filtreras bort i
rerankingsteget. De två felen har olika åtgärder.

Mönstermatchningen unicode-normaliserar (NFC) både mönster och
filnamn — macOS/vissa verktyg lagrar 'ö' dekomponerat, vilket annars
ger tysta mismatchar.
"""

from __future__ import annotations

import argparse
import sys
import unicodedata
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _norm(s: str) -> str:
    """NFC-normalisera och casefolda för robust substrängsmatchning."""
    return unicodedata.normalize("NFC", s).casefold()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspektera indexerade chunkar/evidens för ett dokument.",
    )
    parser.add_argument("pattern", help="Substräng av filnamn eller sökväg (case-okänslig, NFC-normaliserad)")
    parser.add_argument("--question", help="Kör retrievaldiagnos mot denna fråga")
    parser.add_argument("--full", action="store_true", help="Visa chunktexter i sin helhet (annars 300 tecken)")
    parser.add_argument("--evidence", action="store_true", help="Visa även evidensobjekt för dokumentet")
    args = parser.parse_args()

    # Tunga imports först här, så att --help är snabb.
    print("Laddar RagService (kräver att servern är avstängd)...")
    from app.retrieval import RagService

    rag = RagService()
    print("Klart.\n")

    pattern = _norm(args.pattern)

    # ---- Hitta dokument i BM25-indexet (speglar chunk-collectionen) ----
    all_paths = sorted(rag.bm25_index._by_source.keys())
    matches = [p for p in all_paths if pattern in _norm(p)]

    if not matches:
        print(f"Inget indexerat dokument matchar {args.pattern!r}.")
        print(f"Indexet innehåller {len(all_paths)} dokument. Närliggande kandidater:")
        tokens = [t for t in pattern.split() if len(t) >= 4]
        near = [p for p in all_paths if any(t in _norm(p) for t in tokens)]
        for p in near[:10]:
            print(f"  - {p}")
        raise SystemExit(1)

    if len(matches) > 1:
        print(f"{len(matches)} dokument matchar — visar alla. Precisera mönstret om det är för många.\n")

    for source_path in matches:
        chunks = rag.bm25_index.get_chunks_by_source(source_path)
        print("=" * 72)
        print(f"DOKUMENT: {source_path}")
        print(f"Antal chunkar: {len(chunks)}")
        total_chars = sum(len(c.text) for c in chunks)
        print(f"Total textmängd: {total_chars} tecken")
        print()

        for i, c in enumerate(chunks, start=1):
            text = c.text if args.full else c.text[:300] + ("..." if len(c.text) > 300 else "")
            print(f"--- chunk {i}/{len(chunks)}  [{c.metadata.section_title}]  {len(c.text)} tecken")
            print(text)
            print()

        if args.evidence:
            evidence = [
                h for h in rag.store.iter_all_evidence()
                if h.metadata.source_path == source_path
            ]
            print(f"Evidensobjekt: {len(evidence)}")
            for i, e in enumerate(evidence, start=1):
                text = e.text if args.full else e.text[:300] + ("..." if len(e.text) > 300 else "")
                print(f"--- evidens {i}/{len(evidence)}  [{e.metadata.section_title}]")
                print(text)
                print()

    # ---- Retrievaldiagnos ----
    if not args.question:
        return

    question = args.question
    match_set = set(matches)
    print("=" * 72)
    print(f"RETRIEVALDIAGNOS för frågan: {question!r}")
    print()

    # 1. Semantisk kandidatpool (samma limit som RagService.answer)
    query_vector = rag.embedder.embed_query(question)
    semantic_hits = rag.store.search(query_vector, limit=15)
    sem_ranks = [
        (rank, h) for rank, h in enumerate(semantic_hits, start=1)
        if h.metadata.source_path in match_set
    ]
    print(f"1. SEMANTISK sökning (topp-{len(semantic_hits)}):")
    if sem_ranks:
        for rank, h in sem_ranks:
            print(f"   TRÄFF på plats {rank}: [{h.metadata.section_title}] score={h.score:.4f}")
    else:
        print("   INTE i kandidatpoolen. Dokumentet kan aldrig nå rerankern den vägen.")
        top = semantic_hits[0] if semantic_hits else None
        if top:
            print(f"   (plats 1 gick till: {top.metadata.file_name} score={top.score:.4f})")
    print()

    # 2. BM25 med samma synonymexpansion som den riktiga vägen
    synonym_additions = rag.synonyms.expand_terms(question)
    bm25_text = question + (" " + " ".join(synonym_additions) if synonym_additions else "")
    bm25_hits = rag.bm25_index.top_k(bm25_text, k=10)
    bm25_ranks = [
        (rank, h) for rank, h in enumerate(bm25_hits, start=1)
        if h.metadata.source_path in match_set
    ]
    print(f"2. BM25 (topp-{len(bm25_hits)}"
          + (f", synonymtillägg: {synonym_additions}" if synonym_additions else ", inga synonymtillägg")
          + "):")
    if bm25_ranks:
        for rank, h in bm25_ranks:
            print(f"   TRÄFF på plats {rank}: [{h.metadata.section_title}]")
    else:
        print("   INTE bland BM25-kandidaterna (ordagrann matchning saknas — "
              "t.ex. språkgap eller annan terminologi).")
    print()

    # 3. Cross-encoderns bedömning av dokumentets egna chunkar.
    # filter_floor=-999 så att ALLA scores visas, även de som den
    # riktiga vägen skulle filtrera bort (< 0).
    doc_chunks = [c for p in matches for c in rag.bm25_index.get_chunks_by_source(p)]
    reranked, _ = rag.reranker.rerank(question, doc_chunks, filter_floor=-999.0)
    print(f"3. CROSS-ENCODER: dokumentets {len(doc_chunks)} chunkar mot frågan "
          "(negativ score = filtreras bort i den riktiga kedjan):")
    for h in reranked[:10]:
        marker = "  " if h.score >= 0 else "✗ "
        print(f"   {marker}score={h.score:+.4f}  [{h.metadata.section_title}]")
    print()

    # Sammanfattande slutsats
    in_pool = bool(sem_ranks or bm25_ranks)
    best_ce = reranked[0].score if reranked else float("-inf")
    print("SLUTSATS:")
    if not in_pool:
        print("  Dokumentet kommer inte in i kandidatpoolen — felet sitter i")
        print("  kandidatinsamlingen (embedding/BM25), inte i rerankern.")
    elif best_ce < 0:
        print("  Dokumentet når rerankern men alla chunkar får negativ score —")
        print("  cross-encodern bedömer dem som irrelevanta för frågan.")
    else:
        print("  Dokumentet når rerankern och har minst en chunk med positiv")
        print("  score. Om det ändå inte bär svaret: kontrollera urvalssteget")
        print("  (score-gap-regler, dokumentexpansion) i JSONL-spåret.")


if __name__ == "__main__":
    main()
