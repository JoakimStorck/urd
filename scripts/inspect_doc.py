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

Mönstermatchningen unicode-normaliserar (NFC) både mönster och
filnamn — macOS/vissa verktyg lagrar 'ö' dekomponerat, vilket annars
ger tysta mismatchar.
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _norm(s: str) -> str:
    """NFC-normalisera och casefolda för robust substrängsmatchning."""
    return unicodedata.normalize("NFC", s).casefold()


_NUMBERING_RE = re.compile(r"^\s*\d+(?:[.:]\d+)*[.:)]?\s+")


def _strip_numbering(title: str) -> str:
    """
    Ta bort ledande avsnittsnumrering ("7.3.4 ", "12 ", "3) ").

    Numreringen är en ordningsmarkör, inte innehåll. För embedding
    och cross-encoder är "7.2 Behörighet" och "8.2 Behörighet"
    praktiskt taget samma sträng — det är den skillnaden mätningen
    ska fånga.
    """
    return _NUMBERING_RE.sub("", title).strip()


def _sections_in_order(chunks) -> list[dict]:
    """
    Rekonstruera dokumentets sektioner ur de indexerade chunkarna.

    chunk_index är ett löpande index i dokumentordning, så chunkar som
    tillhör samma sektion ligger sammanhängande. Qdrants scroll-ordning
    är däremot inte garanterad — därför sorteras det explicit.
    """
    ordered = sorted(chunks, key=lambda c: c.metadata.chunk_index)
    sections: list[dict] = []
    for c in ordered:
        title = c.metadata.section_title
        level = c.metadata.section_level
        if sections and sections[-1]["title"] == title and sections[-1]["level"] == level:
            sections[-1]["chunks"] += 1
            sections[-1]["chars"] += len(c.text)
        else:
            sections.append({
                "title": title,
                "level": level,
                "chunks": 1,
                "chars": len(c.text),
                "first_index": c.metadata.chunk_index,
            })
    return sections


def _attach_chains(sections: list[dict]) -> list[dict]:
    """
    Bygg full rubrikkedja per sektion med en nivåstack.

    Sektioner utan nivå (text före första rubriken, eller dokument där
    Docling inte hittat rubrikstruktur) får inte störa stacken — de
    bär ingen hierarkisk information.
    """
    stack: list[tuple[int, str]] = []
    for s in sections:
        level = s["level"]
        title = s["title"]
        if level is None:
            s["chain"] = [title] if title else []
            continue
        while stack and stack[-1][0] >= level:
            stack.pop()
        s["chain"] = [t for _, t in stack] + ([title] if title else [])
        if title:
            stack.append((level, title))
    return sections


def _print_sections(chunks) -> None:
    """Skriv rubrikträd i dokumentordning plus rapport över kollisioner."""
    sections = _attach_chains(_sections_in_order(chunks))
    no_level = sum(1 for s in sections if s["level"] is None)

    print(f"STRUKTUR: {len(sections)} sektioner, {len(chunks)} chunkar")
    if no_level:
        print(f"  Varning: sektioner utan nivå: {no_level} — rubrikstrukturen är "
              "ofullständig för detta dokument.")
    print()

    for s in sections:
        level = s["level"]
        indent = "  " * ((level - 1) if level else 0)
        title = s["title"] or "(ingen rubrik)"
        print(f"  {indent}{title}"
              f"   [nivå {level if level is not None else '-'}, "
              f"{s['chunks']} chunkar, {s['chars']} tecken]")
    print()

    # Kollisionsrapport.
    #
    # Mätningen görs på AVNUMRERAD rubrik, inte på exakt sträng.
    # "7.2 Behörighet" och "8.2 Behörighet" är olika strängar men
    # bär samma betydelse: siffrorna är ordningsmarkörer utan
    # semantiskt innehåll för en embedding eller cross-encoder.
    # Det är alltså den avnumrerade formen som avgör om två
    # sektioner är särskiljbara för modellerna.
    by_key: dict[str, list[dict]] = {}
    for s in sections:
        if s["title"]:
            by_key.setdefault(_strip_numbering(s["title"]), []).append(s)
    collisions = {t: ss for t, ss in by_key.items() if len(ss) > 1}

    print("SEMANTISKT ICKE-UNIKA RUBRIKER (numrering bortstädad):")
    if not collisions:
        print("  Inga — varje rubrik är särskiljbar även utan sin numrering.")
        return

    resolved = 0
    total = 0
    for key, group in sorted(collisions.items(), key=lambda kv: -len(kv[1])):
        print(f"  {key!r} förekommer {len(group)} gånger:")
        chains = []
        for s in group:
            chain = " > ".join(_strip_numbering(t) for t in s["chain"]) or key
            chains.append(chain)
            print(f"     {chain}   [{s['chunks']} chunkar]")
        total += len(group)
        if len(set(chains)) == len(chains):
            resolved += len(group)
        print()

    print(f"  {total} sektioner bär en semantiskt icke-unik rubrik "
          f"av totalt {len(sections)}.")
    print(f"  {resolved} av dem blir särskiljbara med full rubrikkedja "
          f"({total - resolved} förblir tvetydiga även då).")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspektera indexerade chunkar/evidens för ett dokument.",
    )
    parser.add_argument("pattern", help="Substräng av filnamn eller sökväg (case-okänslig, NFC-normaliserad)")
    parser.add_argument("--question", help="Kör retrievaldiagnos mot denna fråga")
    parser.add_argument("--full", action="store_true", help="Visa chunktexter i sin helhet (annars 300 tecken)")
    parser.add_argument("--evidence", action="store_true", help="Visa även evidensobjekt för dokumentet")
    parser.add_argument(
        "--sections",
        action="store_true",
        help="Visa rubrikträd och rapport över icke-unika rubriker (utan chunktexter)",
    )
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

        if args.sections:
            _print_sections(chunks)
        else:
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
