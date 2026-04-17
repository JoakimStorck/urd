"""
Retrieval med hybrid sökning (semantisk + BM25) och cross-encoder-reranking.

Ersätter tidigare heuristisk omrankning med en neural reranker som
generaliserar över frågetyper utan handskrivna bonusar.
"""

import re
import time
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from app.config import settings
from app.embeddings import Embedder
from app.qdrant_store import QdrantStore
from app.llm import LocalLLM
from app.prompting import build_prompt
from app.synthesis import synthesize, rework as synthesis_rework
from app.schemas import ChatResponse, SourceHit


# ---------------------------------------------------------------------------
# Boilerplate-filter (behålls – detta är dokumentspecifikt, inte heuristisk
# ranking, och filtrerar bort sektioner som aldrig bär meningsfullt innehåll)
# ---------------------------------------------------------------------------

_BOILERPLATE_SECTION_TITLES = {
    "bilaga",
    "delges",
    "sändlista",
    "sändlista:",
    "protokoll",
    "b e s l u t",
}


def _normalize_text(s: str | None) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", " ", s.strip()).casefold()


def _tokenize_bm25(text: str) -> list[str]:
    """Enkel tokenisering för BM25."""
    return [
        tok
        for tok in re.findall(r"\w+", text.casefold(), flags=re.UNICODE)
        if len(tok) >= 2
    ]


def _is_boilerplate(title: str | None, text: str) -> bool:
    norm_title = _normalize_text(title)
    if norm_title in _BOILERPLATE_SECTION_TITLES:
        return True

    stripped = text.strip()
    if stripped == "<!-- image -->":
        return True

    tokens = _tokenize_bm25(stripped)
    if len(tokens) <= 2 and len(stripped) < 40:
        return True

    return False


# ---------------------------------------------------------------------------
# BM25-index (byggs vid uppstart från alla chunks i Qdrant)
# ---------------------------------------------------------------------------

class BM25Index:
    """Lättviktigt BM25-index som byggs från befintliga Qdrant-chunks."""

    def __init__(self, hits: list[SourceHit]) -> None:
        self.hits = hits
        self._id_to_idx = {h.chunk_id: i for i, h in enumerate(hits)}

        # Dokumentindex för snabb expansion
        self._by_source: dict[str, list[SourceHit]] = {}
        for h in hits:
            self._by_source.setdefault(h.metadata.source_path, []).append(h)

        # BM25 kraschar på tomt corpus — skjut upp skapandet
        if hits:
            corpus = [_tokenize_bm25(h.text) for h in hits]
            self.bm25 = BM25Okapi(corpus)
        else:
            self.bm25 = None

    def top_k(self, question: str, k: int = 10) -> list[SourceHit]:
        """Returnera de k bästa BM25-träffarna som SourceHit."""
        if self.bm25 is None:
            return []
        tokens = _tokenize_bm25(question)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [self.hits[idx] for idx, score in ranked[:k] if score > 0]

    def get_chunks_by_source(self, source_path: str) -> list[SourceHit]:
        """Hämta alla chunkar från ett dokument."""
        return self._by_source.get(source_path, [])


# ---------------------------------------------------------------------------
# Kandidatpool – slå ihop semantisk sökning och BM25
# ---------------------------------------------------------------------------

def _merge_candidates(
    semantic_hits: list[SourceHit],
    bm25_hits: list[SourceHit],
) -> list[SourceHit]:
    """
    Slå ihop kandidater från semantisk sökning och BM25 till en unik pool.
    BM25:s roll är att tillföra kandidater som vektorsökningen missade,
    t.ex. chunkar med exakt terminologimatchning.
    Cross-encodern avgör sedan rankingen.
    """
    seen: set[str] = set()
    merged: list[SourceHit] = []

    for hit in semantic_hits:
        if hit.chunk_id not in seen:
            seen.add(hit.chunk_id)
            merged.append(hit)

    for hit in bm25_hits:
        if hit.chunk_id not in seen:
            seen.add(hit.chunk_id)
            merged.append(hit)

    return merged


# ---------------------------------------------------------------------------
# Dedup – undvik dubbletter från samma sektion
# ---------------------------------------------------------------------------

def _dedup_and_select(
    ranked: list[SourceHit],
    top_k: int,
) -> list[SourceHit]:
    """
    Välj topp-K med dedup per (source_path, section_title).

    Flera sektioner från samma dokument tillåts — cross-encoderns
    ranking avgör vilka som når topp-K. Ingen godtycklig gräns per
    dokument: om ett dokument har 6 relevanta sektioner och top_k=5
    får det 5 av dem.
    """
    selected: list[SourceHit] = []
    seen_keys: set[tuple[str, str | None]] = set()

    for hit in ranked:
        key = (hit.metadata.source_path, hit.metadata.section_title)
        if key in seen_keys:
            continue

        seen_keys.add(key)
        selected.append(hit)

        if len(selected) >= top_k:
            break

    return selected


# ---------------------------------------------------------------------------
# Cross-encoder reranking
# ---------------------------------------------------------------------------

class Reranker:
    def __init__(self) -> None:
        try:
            self.model = CrossEncoder(settings.reranker_model)
        except Exception as e:
            raise RuntimeError(
                f"Kunde inte ladda reranker-modellen '{settings.reranker_model}'. "
                f"URD använder endast standardladdning utan remote code. "
                f"Ursprungligt fel: {type(e).__name__}: {e}"
            ) from e
            
    def rerank(
        self,
        question: str,
        hits: list[SourceHit],
        filter_floor: float = 0.0,
    ) -> tuple[list[SourceHit], list[dict]]:
        """
        Rerankar kandidater med cross-encoder.

        Returnerar (sorterade hits, debug-info). Kandidater med score
        under filter_floor filtreras bort.

        filter_floor default är 0.0: cross-encodern bedömer chunkar
        med negativ score som irrelevanta. För chunkar som kommer
        från dokument som redan visat sig starkt relevanta (via
        expansion) kan en lägre floor användas — t.ex. -1.0 — så
        att borderline-relevanta sektioner från ett relevant dokument
        inte filtreras bort trots att cross-encodern är osäker på dem
        individuellt.
        """
        if not hits:
            return [], []

        # Filtrera boilerplate före reranking
        filtered = [
            h for h in hits
            if not _is_boilerplate(h.metadata.section_title, h.text)
        ]

        if not filtered:
            filtered = hits  # fallback om allt filtreras

        # Cross-encoder: varje (fråga, chunk) bedöms som ett par
        pairs = [(question, h.text) for h in filtered]
        scores = self.model.predict(pairs)

        scored = list(zip(scores, filtered))
        scored.sort(key=lambda x: x[0], reverse=True)

        reranked: list[SourceHit] = []
        debug: list[dict] = []

        for ce_score, hit in scored:
            debug.append({
                "file_name": hit.metadata.file_name,
                "section_title": hit.metadata.section_title,
                "document_title": hit.metadata.document_title,
                "cross_encoder_score": round(float(ce_score), 4),
                "document_type": hit.metadata.document_type,
                "filtered": float(ce_score) < filter_floor,
            })

            if float(ce_score) < filter_floor:
                continue

            reranked.append(SourceHit(
                chunk_id=hit.chunk_id,
                score=float(ce_score),
                text=hit.text,
                metadata=hit.metadata,
            ))

        return reranked, debug


# ---------------------------------------------------------------------------
# RagService
# ---------------------------------------------------------------------------

class RagService:
    def __init__(self) -> None:
        self.embedder = Embedder()
        test_vec = self.embedder.embed_query("test")
        self.store = QdrantStore(vector_size=len(test_vec))
        self.llm = LocalLLM()
        self.reranker = Reranker()

        # Bygg BM25-index från alla chunks i Qdrant
        self._build_bm25_index()

    def _build_bm25_index(self) -> None:
        """Bygg eller återbygg BM25-indexet från Qdrant."""
        all_chunks = self.store.iter_all_chunks()
        self.bm25_index = BM25Index(all_chunks)

    def refresh_index(self) -> int:
        """
        Återbygg BM25-indexet från Qdrant.
        Anropas efter ingest för att synka retrieval med nytt innehåll.
        Returnerar antal chunks i det nya indexet.
        """
        self._build_bm25_index()
        return len(self.bm25_index.hits)

    def answer(
        self,
        question: str,
        qud_anchor: str | None = None,
        background_turns: list[dict] | None = None,
        background_max_turns: int = 0,
        style: str | None = None,
    ) -> ChatResponse:
        """
        Kör retrieval och syntes.

        Parametrar:
        - question: originalfrågan som användaren ställde. Det är den
          som syntesen refererar till.
        - qud_anchor: om satt, en QUD-text som konkateneras med question
          för att bilda söktexten som används i semantisk sökning, BM25
          och cross-encoder-reranking. Används för related_to_qud-fall
          där den aktiva huvudfrågan ska påverka retrieval utan att
          förvränga originalfrågan i syntesen.
        - background_turns, background_max_turns: samtalsbakgrund som
          skickas med till evidensextraktionen.
        - style: valfri stilmarkör som styr svarsformuleringen. Giltiga
          värden hanteras i synthesis.py. None = standardstil.
        """
        t0 = time.perf_counter()

        # Bygg söktexten. Om QUD-ankare finns konkateneras det med
        # originalfrågan så att både embedding-modell, BM25 och
        # cross-encoder får samma utökade kontext.
        if qud_anchor:
            search_text = f"{question}\n\n(Huvudfråga i samtalet: {qud_anchor})"
        else:
            search_text = question

        # 1. Semantisk sökning via Qdrant
        query_vector = self.embedder.embed_query(search_text)
        t1 = time.perf_counter()

        semantic_hits = self.store.search(query_vector, limit=15)
        t2 = time.perf_counter()

        # 2. BM25-sökning – tillför kandidater med exakt ordmatchning
        bm25_hits = self.bm25_index.top_k(search_text, k=10)

        # 3. Slå ihop till en unik kandidatpool
        candidates = _merge_candidates(semantic_hits, bm25_hits)
        t3 = time.perf_counter()

        # 4. Första reranking
        reranked, rerank_debug = self.reranker.rerank(search_text, candidates)
        t4 = time.perf_counter()

        # 5. Dokumentexpansion: för dokument med högt rankade chunkar,
        #    hämta övriga chunkar och låt cross-encodern bedöma dem
        expanded_new = self._expand_from_top_docs(reranked, candidates)
        num_expanded = len(expanded_new)

        if expanded_new:
            # Andra rerankingpasset använder en lägre filtreringströskel
            # eftersom chunkarna kommer från dokument som redan visat
            # sig starkt relevanta. Se expanded_filter_floor i config.
            exp_reranked, exp_debug = self.reranker.rerank(
                search_text,
                expanded_new,
                filter_floor=settings.expanded_filter_floor,
            )
            # Slå ihop med första rankingen och sortera om
            all_reranked = reranked + exp_reranked
            all_reranked.sort(key=lambda h: h.score, reverse=True)
            rerank_debug = rerank_debug + exp_debug
        else:
            all_reranked = reranked

        expanded_doc_paths = sorted({
            hit.metadata.source_path for hit in expanded_new
        })        
        
        t5 = time.perf_counter()

        # 6. Dedup och välj topp-K
        hits = _dedup_and_select(all_reranked, settings.top_k)

        if not hits:
            return ChatResponse(
                answer=(
                    "Jag hittar inget tydligt stöd i de indexerade dokumenten "
                    "för att besvara frågan."
                ),
                sources=[],
                debug={
                    "top_k": settings.top_k,
                    "num_semantic": len(semantic_hits),
                    "num_bm25": len(bm25_hits),
                    "num_candidates": len(candidates),
                    "num_expanded": num_expanded,
                    "expanded_docs": expanded_doc_paths,
                    "abstained": True,
                    "qud_anchor_used": qud_anchor is not None,
                    "timing_s": {
                        "embed_query": round(t1 - t0, 3),
                        "search": round(t2 - t1, 3),
                        "bm25_and_merge": round(t3 - t2, 3),
                        "rerank_1": round(t4 - t3, 3),
                        "expand_and_rerank_2": round(t5 - t4, 3),
                        "total": round(t5 - t0, 3),
                    },
                },
            )

        # 7. Tvåstegssyntes: evidensextraktion → svarsformulering
        t6 = time.perf_counter()

        synthesis_result = synthesize(
            question,
            hits,
            self.llm,
            background_turns=background_turns,
            background_max_turns=background_max_turns,
            style=style,
        )
        t7 = time.perf_counter()

        # Bygg debug-info för syntesen
        synthesis_debug = {
            "used_fallback": synthesis_result.used_fallback,
        }
        if synthesis_result.fallback_reason:
            synthesis_debug["fallback_reason"] = synthesis_result.fallback_reason
        if synthesis_result.evidence is not None:
            synthesis_debug["num_extracted"] = len(synthesis_result.evidence.extracted)
            synthesis_debug["not_found"] = synthesis_result.evidence.not_found
            if synthesis_result.evidence.raw_json:
                synthesis_debug["evidence_json"] = synthesis_result.evidence.raw_json
        if synthesis_result.timing_s:
            synthesis_debug["timing_s"] = synthesis_result.timing_s

        return ChatResponse(
            answer=synthesis_result.answer,
            sources=hits,
            debug={
                "top_k": settings.top_k,
                "num_semantic": len(semantic_hits),
                "num_bm25": len(bm25_hits),
                "num_candidates": len(candidates),
                "num_expanded": num_expanded,
                "num_reranked": len(all_reranked),
                "num_hits": len(hits),
                "abstained": False,
                "qud_anchor_used": qud_anchor is not None,
                "synthesis": synthesis_debug,
                "timing_s": {
                    "embed_query": round(t1 - t0, 3),
                    "search": round(t2 - t1, 3),
                    "bm25_and_merge": round(t3 - t2, 3),
                    "rerank_1": round(t4 - t3, 3),
                    "expand_and_rerank_2": round(t5 - t4, 3),
                    "synthesize": round(t7 - t6, 3),
                    "total": round(t7 - t0, 3),
                },
                "rerank_top": sorted(
                    rerank_debug,
                    key=lambda d: d.get("cross_encoder_score", -999),
                    reverse=True,
                )[: settings.top_k + 5],
            },
        )

    def rework(
        self,
        question: str,
        hits: list[SourceHit],
        previous_answer: str,
        mode: str,
    ) -> ChatResponse:
        """
        Arbeta mot föregående turs källor utan ny retrieval.

        Används av elaboration och verification. mode är en av:
        - "elaboration": lyft fram vad som prioriterades bort
        - "verification": strikt granskning av tidigare svar

        Returnerar en ChatResponse där sources är samma hits som
        bar det tidigare svaret (så UI:t fortfarande visar dem som
        källor och debug är sammanhängande).
        """
        t0 = time.perf_counter()

        synthesis_result = synthesis_rework(
            question,
            hits,
            previous_answer,
            self.llm,
            mode=mode,
        )

        t1 = time.perf_counter()

        synthesis_debug = {
            "used_fallback": synthesis_result.used_fallback,
            "mode": mode,
        }
        if synthesis_result.fallback_reason:
            synthesis_debug["fallback_reason"] = synthesis_result.fallback_reason
        if synthesis_result.evidence is not None:
            synthesis_debug["num_extracted"] = len(synthesis_result.evidence.extracted)
            synthesis_debug["not_found"] = synthesis_result.evidence.not_found
            if synthesis_result.evidence.raw_json:
                synthesis_debug["evidence_json"] = synthesis_result.evidence.raw_json
        if synthesis_result.timing_s:
            synthesis_debug["timing_s"] = synthesis_result.timing_s

        return ChatResponse(
            answer=synthesis_result.answer,
            sources=hits,
            debug={
                "rework_mode": mode,
                "num_hits_reused": len(hits),
                "abstained": not (synthesis_result.evidence and synthesis_result.evidence.extracted),
                "synthesis": synthesis_debug,
                "timing_s": {
                    "rework": round(t1 - t0, 3),
                    "total": round(t1 - t0, 3),
                },
            },
        )

    def _expand_from_top_docs(
        self,
        reranked: list[SourceHit],
        already_seen: list[SourceHit],
        score_threshold: float | None = None,
    ) -> list[SourceHit]:
        """
        Expandera alla dokument som redan visat tydlig relevans.
    
        Om ett dokument har minst en chunk med score >= score_threshold,
        hämtas övriga chunkar från samma dokument som ännu inte finns i
        kandidatpoolen. Dessa får sedan bedömas i en andra rerankingrunda.
    
        Detta gör expansionen dokumentdriven i stället för att begränsa
        den till ett fast antal toppdokument.

        score_threshold default läses från settings.expansion_score_threshold.
        """
        if not reranked:
            return []

        if score_threshold is None:
            score_threshold = settings.expansion_score_threshold
    
        seen_ids = {h.chunk_id for h in already_seen} | {h.chunk_id for h in reranked}
    
        # Alla dokument som visat tydlig relevans får expanderas
        docs_to_expand: list[str] = []
        seen_docs: set[str] = set()
    
        for hit in reranked:
            source_path = hit.metadata.source_path
            if hit.score >= score_threshold and source_path not in seen_docs:
                docs_to_expand.append(source_path)
                seen_docs.add(source_path)
    
        if not docs_to_expand:
            return []
    
        new_candidates: list[SourceHit] = []
    
        for source_path in docs_to_expand:
            doc_chunks = self.bm25_index.get_chunks_by_source(source_path)
            for chunk in doc_chunks:
                if chunk.chunk_id not in seen_ids:
                    new_candidates.append(chunk)
                    seen_ids.add(chunk.chunk_id)
    
        return new_candidates