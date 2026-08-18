"""
Inspektera vad indexet innehåller för ett dokument — och diagnostisera
varför en fråga inte når det.

Skriptet är numera ett SKAL. Analysen bor i app/inspect.py som typade
operationer, så att samma rapport kan bäras av 'urd docs inspect', av
ett punktkommando i interaktivt läge och av en endpoint utan att
logiken finns i tre exemplar. Kvar här: argumenthantering, uppstart av
RagService och retrievaldiagnosen (som ännu reproducerar framkedjan
för hand — den ska bli en projektion av ett verkligt anrop).

Körs från projektroten med URD-servern AVSTÄNGD (inbäddad Qdrant
tillåter bara en process):

    python -m scripts.inspect_doc "reviderad rutin för beslut"
    python -m scripts.inspect_doc "reviderad rutin" --chunks
    python -m scripts.inspect_doc "reviderad rutin" --evidence
    python -m scripts.inspect_doc "anstallningsordning" --sections
    python -m scripts.inspect_doc "reviderad rutin" \
        --question "Vilka lokala rutiner gäller på IIT före beslut om extern forskningsansökan?"

Med --sections visas dokumentets rubrikträd i dokumentordning, samt
vilka rubriker som INTE är unika inom dokumentet. Det senare är den
avgörande siffran: en chunk indexeras idag med dokumenttitel plus
NÄRMASTE rubrik, så två sektioner som heter samma sak blir oskiljbara
för embedding, BM25 och cross-encoder — även när föräldrarubriken
skiljer dem åt. Rapporten visar hur många av dessa kollisioner som
skulle upplösas av en full rubrikkedja.

Med --question körs en retrievaldiagnos som svarar på tre frågor:

  1. Kommer dokumentet in i den SEMANTISKA kandidatpoolen (topp-15)?
  2. Kommer det in via BM25 (topp-10, med synonymexpansion)?
  3. Hur bedömer CROSS-ENCODERN dokumentets chunkar mot frågan?

Ett dokument som saknas i 1 och 2 når aldrig rerankern — felet sitter
i kandidatinsamlingen (t.ex. språkgap eller embeddingkvalitet). Ett
dokument som kommer in men får negativa CE-scores filtreras bort i
rerankingsteget. De två felen har olika åtgärder.

Mönstermatchningen (i app/inspect.py) unicode-normaliserar NFC både
mönster och filnamn — macOS/vissa verktyg lagrar 'ö' dekomponerat,
vilket annars ger tysta mismatchar.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspektera indexerade chunkar/evidens för ett dokument.",
    )
    parser.add_argument("pattern", help="Substräng av filnamn eller sökväg (case-okänslig, NFC-normaliserad)")
    parser.add_argument("--question", help="Kör retrievaldiagnos mot denna fråga")
    # VOKABULÄREN ÄR GEMENSAM. Flaggan hette --full medan operationen
    # tar parametern chunks — samma sak, två namn, i samma kodbas.
    # Fyra ytor ska så småningom bära samma ord (skript, CLI, HTTP,
    # interaktivt läge), och driften börjar alltid så här.
    # --full behålls som odokumenterat alias.
    parser.add_argument("--chunks", "--full", dest="chunks", action="store_true",
                        help="Visa chunktexter i sin helhet (annars 300 tecken)")
    parser.add_argument("--evidence", action="store_true", help="Visa även evidensobjekt för dokumentet")
    parser.add_argument(
        "--attest",
        action="store_true",
        help="Visa Attests observationer ur dokumentet",
    )
    parser.add_argument(
        "--sections",
        action="store_true",
        help="Visa rubrikträd och rapport över icke-unika rubriker (utan chunktexter)",
    )
    args = parser.parse_args()

    # Tunga imports först här, så att --help är snabb.
    print("Laddar RagService (kräver att servern är avstängd)...")
    from app.retrieval import RagService
    from app.qdrant_store import StorageLockedError

    try:
        rag = RagService()
    except StorageLockedError as e:
        print(f"\n{e}", file=sys.stderr)
        raise SystemExit(1)
    print("Klart.\n")

    from app import inspect as ins

    all_paths = sorted(rag.bm25_index._by_source.keys())
    resolution = ins.resolve_document(args.pattern, all_paths)
    print(ins.format_resolution(resolution))
    if not resolution.found:
        # Grep-konventionen: 1 = lyckades men gav ingen träff.
        raise SystemExit(1)
    print()

    matches = resolution.matched
    for source_path in matches:
        report = ins.inspect_document(
            source_path,
            rag.store,
            sections=args.sections,
            chunks=not args.sections,
            evidence=args.evidence,
            attest=args.attest,
            all_chunks=rag.bm25_index.hits,
        )
        if report is None:
            continue
        if report.chunks and not args.chunks:
            for ch in report.chunks:
                if len(ch.text) > 300:
                    ch.text = ch.text[:300] + "..."
        if report.evidence and not args.chunks:
            for e in report.evidence:
                if len(e.text) > 300:
                    e.text = e.text[:300] + "..."
        print(ins.format_document_report(report))
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
    # filter_floor=0.0 så att ALLA sannolikheter visas, även de som
    # den riktiga vägen skulle filtrera bort (< 0.5).
    doc_chunks = [c for p in matches for c in rag.bm25_index.get_chunks_by_source(p)]
    reranked, _ = rag.reranker.rerank(question, doc_chunks, filter_floor=0.0)
    print(f"3. CROSS-ENCODER: dokumentets {len(doc_chunks)} chunkar mot frågan "
          "(sannolikhet < 0.5 = filtreras bort i den riktiga kedjan):")
    for h in reranked[:10]:
        marker = "  " if h.score >= 0.5 else "✗ "
        print(f"   {marker}prob={h.score:.4f}  [{h.metadata.section_title}]")
    print()

    # Sammanfattande slutsats
    in_pool = bool(sem_ranks or bm25_ranks)
    best_ce = reranked[0].score if reranked else 0.0
    print("SLUTSATS:")
    if not in_pool:
        print("  Dokumentet kommer inte in i kandidatpoolen — felet sitter i")
        print("  kandidatinsamlingen (embedding/BM25), inte i rerankern.")
    elif best_ce < 0.5:
        print("  Dokumentet når rerankern men ingen chunk når sannolikhet 0.5 —")
        print("  cross-encodern bedömer dem som irrelevanta för frågan.")
    else:
        print("  Dokumentet når rerankern och har minst en chunk med positiv")
        print("  score. Om det ändå inte bär svaret: kontrollera urvalssteget")
        print("  (score-gap-regler, dokumentexpansion) i JSONL-spåret.")


if __name__ == "__main__":
    main()
