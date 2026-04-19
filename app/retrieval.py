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
from app.synthesis import synthesize
from app.rework import elaborate, verify
from app.schemas import ChatResponse, SourceHit
from app.synonyms import load_synonyms
from app.concepts import load_concepts

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

def _contains_label(text: str, label: str) -> bool:
    if not text or not label:
        return False
    return label.casefold() in text.casefold()

def _ordered_unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out

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




def _top_document_paths(ranked: list[SourceHit], max_docs: int = 3) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    for hit in ranked:
        path = hit.metadata.source_path
        if not path or path in seen:
            continue
        seen.add(path)
        paths.append(path)
        if len(paths) >= max_docs:
            break
    return paths


def _boost_evidence_from_text_support(
    evidence_hits: list[SourceHit],
    text_hits: list[SourceHit],
    section_boost: float,
    document_boost: float,
) -> tuple[list[SourceHit], list[dict]]:
    """
    Höj evidensobjektens score när vanliga textchunkar från samma
    sektion eller samma dokument redan rankat högt.

    Tanken: ett evidensobjekt (tabell, lista, figur) är ofta språkligt
    svagt och får låg individuell rankscore trots att det bär central
    information. Men om den FÖRKLARANDE TEXT som omger objektet har
    rankat högt som vanlig textchunk, är objektet sannolikt relevant
    för frågan — texten introducerar eller refererar till det.

    Två nivåer:
    - Sektion-match: samma (source_path, section_title) som någon
      textchunk. Stark indikation. Full boost.
    - Dokument-match (men inte sektion): samma source_path men annan
      sektion. Svagare. Mindre boost.

    Endast text_hits med positiv score räknas som stödjande — vi vill
    inte boosta evidens baserat på chunkar som cross-encodern dömt ut.

    Returnerar (boostade hits sorterade fallande, debug-rader).
    """
    if not evidence_hits:
        return [], []

    supporting_sections: set[tuple[str, str | None]] = set()
    supporting_documents: set[str] = set()

    for hit in text_hits:
        if hit.score <= 0:
            continue
        path = hit.metadata.source_path
        if not path:
            continue
        supporting_documents.add(path)
        supporting_sections.add((path, hit.metadata.section_title))

    boosted: list[SourceHit] = []
    debug: list[dict] = []

    for hit in evidence_hits:
        path = hit.metadata.source_path
        section = hit.metadata.section_title
        original = hit.score

        applied = 0.0
        reason = None
        if (path, section) in supporting_sections:
            applied = section_boost
            reason = "section_match"
        elif path in supporting_documents:
            applied = document_boost
            reason = "document_match"

        new_score = original + applied
        debug.append({
            "file_name": hit.metadata.file_name,
            "section_title": section,
            "evidence_type": hit.metadata.document_type,
            "original_score": round(original, 4),
            "applied_boost": round(applied, 4),
            "boosted_score": round(new_score, 4),
            "boost_reason": reason,
        })

        if applied > 0:
            boosted.append(
                SourceHit(
                    chunk_id=hit.chunk_id,
                    score=new_score,
                    text=hit.text,
                    metadata=hit.metadata,
                )
            )
        else:
            boosted.append(hit)

    boosted.sort(key=lambda h: h.score, reverse=True)
    return boosted, debug


def _select_evidence_hits(ranked: list[SourceHit], max_hits: int) -> list[SourceHit]:
    """
    Välj evidensobjekt relativt sin egen toppscore, utan absolut golv.

    Evidensobjekt inom redan valda dokument ska få företräde även om de
    individuellt är språkligt svagare än vanliga textchunkar. Därför
    används en mild relativ cutoff och ett litet maxantal.
    """
    if not ranked:
        return []

    top_score = ranked[0].score
    cutoff = top_score * 0.35

    selected: list[SourceHit] = []
    seen_keys: set[tuple[str, str | None, str]] = set()

    for hit in ranked:
        if hit.score < cutoff and selected:
            break

        key = (
            hit.metadata.source_path,
            hit.metadata.section_title,
            hit.metadata.document_type or "",
        )
        if key in seen_keys:
            continue

        seen_keys.add(key)
        selected.append(hit)
        if len(selected) >= max_hits:
            break

    return selected


def _merge_with_evidence_precedence(
    text_hits: list[SourceHit],
    evidence_hits: list[SourceHit],
    max_hits: int,
) -> list[SourceHit]:
    """
    Ge evidensobjekt företräde inom redan valda dokument.

    Strategin är enkel:
    1. ta först utvalda evidensobjekt
    2. fyll sedan på med textträffar från samma dokument
    3. fyll därefter på med övriga textträffar

    Detta håller fast vid principen "evidensobjekt först, stödtext sedan"
    utan att kasta bort den vanliga textretrievalen.
    """
    selected: list[SourceHit] = []
    seen_ids: set[str] = set()

    evidence_doc_paths = {
        hit.metadata.source_path for hit in evidence_hits if hit.metadata.source_path
    }

    def add(hit: SourceHit) -> None:
        if hit.chunk_id in seen_ids:
            return
        seen_ids.add(hit.chunk_id)
        selected.append(hit)

    for hit in evidence_hits:
        add(hit)
        if len(selected) >= max_hits:
            return selected

    for hit in text_hits:
        if hit.metadata.source_path in evidence_doc_paths:
            add(hit)
            if len(selected) >= max_hits:
                return selected

    for hit in text_hits:
        add(hit)
        if len(selected) >= max_hits:
            return selected

    return selected

def _select_hits_for_synthesis(hits: list[SourceHit]) -> list[SourceHit]:
    """
    Välj en snävare delmängd av hits för huvudsyntesen baserat på score-gap.

    Tanken:
    - UI kan visa flera källor
    - huvudsvaret ska i första hand bäras av de tydligast relevanta
      träffarna, så att LLM:n inte glider till svagare källor

    Policy:
    - 1 hit om toppträffen sticker ut mycket tydligt
    - 2 hits om toppträffen är tydlig men inte ensam
    - annars 3 hits
    """
    if not hits:
        return []

    if len(hits) == 1:
        return hits[:1]

    score_1 = hits[0].score
    score_2 = hits[1].score

    if score_2 <= 0:
        return hits[:1]

    if score_1 >= score_2 * 2.0:
        return hits[:1]

    if score_1 >= score_2 * 1.35:
        return hits[:2]

    return hits[:3]
    
# ---------------------------------------------------------------------------
# Dedup – undvik dubbletter från samma sektion
# ---------------------------------------------------------------------------

def _dedup_and_select(ranked: list[SourceHit]) -> list[SourceHit]:
    """
    Välj hits baserat på relevans (cross-encoder-score), med dedup
    per (source_path, section_title).

    Urvalet görs i två passes:

    Pass 1 (primärt): ta alla hits med score ≥ max(min_relevance_floor,
    top_score × relevance_ratio), upp till max_hits.

    Pass 2 (kompletterande): om färre än min_desired_hits valdes i
    pass 1 men det finns fler positiva hits kvar, fyll på upp till
    min_desired_hits totalt. Detta ger elaboration och liknande
    vägar tillräckligt material att arbeta mot, utan att införa
    ett godtyckligt absolut golv. Endast positiva scores används
    — vi vill inte släppa in chunkar som cross-encodern aktivt
    dömt ut.
    """
    if not ranked:
        return []

    top_score = ranked[0].score
    cutoff = max(
        settings.min_relevance_floor,
        top_score * settings.relevance_ratio,
    )

    selected: list[SourceHit] = []
    seen_keys: set[tuple[str, str | None]] = set()

    # Pass 1: välj allt över cutoff
    for hit in ranked:
        if hit.score < cutoff:
            break  # listan är sorterad fallande

        key = (hit.metadata.source_path, hit.metadata.section_title)
        if key in seen_keys:
            continue

        seen_keys.add(key)
        selected.append(hit)

        if len(selected) >= settings.max_hits:
            break

    # Pass 2: fyll på till min_desired_hits om tillräckligt svaga
    # men positiva hits finns. Hoppar över de redan valda via dedup.
    if len(selected) < settings.min_desired_hits:
        for hit in ranked:
            if len(selected) >= settings.min_desired_hits:
                break
            if hit.score <= 0:
                break  # inga fler positiva kvar
            if hit.score >= cutoff:
                continue  # redan övervägd i pass 1
            key = (hit.metadata.source_path, hit.metadata.section_title)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            selected.append(hit)

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

        # Ladda instansens synonymlista om den finns. Tyst fallback
        # till tomt index om filen saknas eller är felaktig.
        self.synonyms = load_synonyms(settings.synonyms_path)
        
        # Ladda instansens begreppsmodell om den finns. Används ännu
        # inte i retrievalpolicyn, men hålls laddad så att strukturen
        # kan testas och byggas ut stegvis.
        self.concepts = load_concepts(settings.concepts_path)

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

    def _evidence_candidates_for_documents(
        self,
        rerank_text: str,
        query_vector: list[float],
        text_hits: list[SourceHit],
    ) -> tuple[list[SourceHit], list[dict], list[str], list[dict]]:
        """
        Hämta evidensobjekt från ett litet antal redan utvalda dokument,
        reranka dem mot frågan, boosta dem baserat på textstöd och välj
        ut de starkaste.

        rerank_text ska vara den rena frågan (inte en QUD-konkatenerad
        sökvariant), eftersom cross-encodern är känslig för meta-text
        i frågesträngen. Se answer() för resonemanget.

        query_vector tas emot färdigberäknat från den vanliga retrieval-
        vägen så att vi inte kör samma embedding-anrop två gånger per
        request.

        text_hits används både för att välja ut kandidatdokument (de tre
        högst rankade dokumenten) och för att identifiera vilka sektioner
        som har starkt textstöd. Evidensobjekt från en sektion där text
        också rankat högt får en kraftig boost; evidens från bara samma
        dokument får en svag boost.

        Returnerar (utvalda hits, rerank-debug, source_paths, boost-debug).
        """
        source_paths = _top_document_paths(text_hits, max_docs=3)
        if not source_paths:
            return [], [], [], []

        evidence_candidates = self.store.search_evidence(
            query_vector,
            source_paths=source_paths,
            limit=12,
        )
        if not evidence_candidates:
            return [], [], source_paths, []

        reranked, debug = self.reranker.rerank(
            rerank_text,
            evidence_candidates,
            filter_floor=-1.0,
        )

        boosted, boost_debug = _boost_evidence_from_text_support(
            reranked,
            text_hits,
            section_boost=settings.evidence_section_boost,
            document_boost=settings.evidence_document_boost,
        )

        selected = _select_evidence_hits(boosted, max_hits=min(4, settings.max_hits))
        return selected, debug, source_paths, boost_debug

    def _related_concepts_in_selected_docs(
        self,
        question: str,
        hits: list[SourceHit],
        max_items: int = 5,
    ) -> list[str]:
        """
        Hitta relaterade begrepp som faktiskt förekommer i de dokument som
        bar svaret.
    
        Relationer:
        - broader-begrepp till de begrepp som matchar frågan
        - syskonbegrepp som delar samma broader-begrepp
    
        Endast begrepp som faktiskt kan beläggas i samma dokument returneras.
        """
        if not hits:
            return []
    
        matched_ids = self.concepts.find_matching_concept_ids(question)
        if not matched_ids:
            return []
    
        matched_set = set(matched_ids)
    
        # Kandidatbegrepp: broader + syskon
        candidate_ids: list[str] = []
        broader_ids: list[str] = []
    
        for concept_id in matched_ids:
            concept = self.concepts.concepts.get(concept_id)
            if concept is None:
                continue
            for broader_id in concept.broader:
                broader_ids.append(broader_id)
                candidate_ids.append(broader_id)
    
        for concept in self.concepts.concepts.values():
            if concept.concept_id in matched_set:
                continue
            if set(concept.broader) & set(broader_ids):
                candidate_ids.append(concept.concept_id)
    
        candidate_ids = _ordered_unique(candidate_ids)
    
        source_paths = {
            hit.metadata.source_path
            for hit in hits
            if hit.metadata.source_path
        }
        if not source_paths:
            return []
    
        found_labels: list[str] = []
    
        for candidate_id in candidate_ids:
            concept = self.concepts.concepts.get(candidate_id)
            if concept is None or not concept.labels:
                continue
    
            found = False
            for path in source_paths:
                for chunk in self.bm25_index.get_chunks_by_source(path):
                    haystack_title = chunk.metadata.section_title or ""
                    haystack_text = chunk.text or ""
    
                    if any(
                        _contains_label(haystack_title, label)
                        or _contains_label(haystack_text, label)
                        for label in concept.labels
                    ):
                        found = True
                        break
                if found:
                    break
    
            if found:
                found_labels.append(concept.labels[0])
    
            if len(found_labels) >= max_items:
                break
    
        return found_labels
        
    def answer(
        self,
        question: str,
        qud_anchor: str | None = None,
        background_turns: list[dict] | None = None,
        background_max_turns: int = 0,
        retrieval_question: str | None = None,
        preferred_source_paths: list[str] | None = None,
    ) -> ChatResponse:
        """
        Kör retrieval och syntes.

        Parametrar:
        - question: originalfrågan som användaren ställde. Det är den
          som syntesen refererar till, och det är den som cross-encodern
          bedömer chunkar mot.
        - qud_anchor: om satt, en QUD-text som konkateneras med question
          för att bredda kandidatpoolen i semantisk sökning och BM25.
          Cross-encoder-rerankingen använder däremot alltid den rena
          frågan (question eller retrieval_question), eftersom mMARCO-
          tränade rerankers är känsliga för konkatenerade söktexter och
          ger kollapsade scores när frågan innehåller meta-formulering
          som "Huvudfråga i samtalet: ...". QUD-ankaret påverkar alltså
          *vad som tas upp i poolen*, inte *hur det bedöms*.
        - background_turns, background_max_turns: samtalsbakgrund som
          skickas med till syntesen.
        - retrieval_question: omskriven fråga för retrieval. Om satt
          används den i stället för question i embedding, BM25 och
          cross-encoder-reranking. question används fortfarande i syntesen.
        - preferred_source_paths: dokument som bör prioriteras i
          retrieval, t.ex. aktiva dokument från föregående svar.
        """
        t0 = time.perf_counter()

        # Bygg söktexten. Två varianter:
        #
        # - search_text används i kandidatinsamling (embedding + BM25).
        #   Här är QUD-konkatenering mindre skadlig — embeddings klarar
        #   längre texter, och BM25 kan dra nytta av bredare ordmängd.
        #
        # - rerank_text används i cross-encoder-reranking och är alltid
        #   den rena frågan. Cross-encodern (mMARCO-tränad) är känslig
        #   för konkatenerade fråge-strängar som innehåller meta-text
        #   av formen "(Huvudfråga i samtalet: ...)" och ger kollapsade
        #   scores i det fallet. Att hålla den rena bevarar rerankerns
        #   precision även när QUD-ankaret breddar kandidatpoolen.
        if retrieval_question:
            search_text = retrieval_question
            rerank_text = retrieval_question
        elif qud_anchor:
            search_text = f"{question}\n\n(Huvudfråga i samtalet: {qud_anchor})"
            rerank_text = question
        else:
            search_text = question
            rerank_text = question

        # 1. Semantisk sökning via Qdrant. För broadening kan retrievaln
        # ankras lokalt till tidigare aktiva dokument.
        query_vector = self.embedder.embed_query(search_text)
        t1 = time.perf_counter()

        semantic_hits = self.store.search(
            query_vector,
            limit=15,
            source_paths=preferred_source_paths,
        )
        t2 = time.perf_counter()

        # 2. BM25-sökning – tillför kandidater med exakt ordmatchning.
        # Den hålls global i första versionen av broadening-fixen.
        # Synonymexpansion breddar söktexten med kända termvarianter
        # (se app/synonyms.py). Det påverkar bara BM25 — embedding
        # och cross-encoder-rerank arbetar på den ursprungliga frågan.
        synonym_additions = self.synonyms.expand_terms(search_text)
        if synonym_additions:
            bm25_search_text = search_text + " " + " ".join(synonym_additions)
        else:
            bm25_search_text = search_text
        bm25_hits = self.bm25_index.top_k(bm25_search_text, k=10)

        # 3. Slå ihop till en unik kandidatpool
        candidates = _merge_candidates(semantic_hits, bm25_hits)
        t3 = time.perf_counter()

        # 4. Första reranking – använder den rena frågan, inte QUD-ankaret
        reranked, rerank_debug = self.reranker.rerank(rerank_text, candidates)
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
                rerank_text,
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

        # 6. Texturval efter vanlig retrieval
        text_hits = _dedup_and_select(all_reranked)

        # 7. Evidensobjekt inom valda dokument får företräde
        evidence_hits: list[SourceHit] = []
        evidence_debug: list[dict] = []
        evidence_source_paths: list[str] = []
        evidence_boost_debug: list[dict] = []
        if text_hits:
            evidence_hits, evidence_debug, evidence_source_paths, evidence_boost_debug = self._evidence_candidates_for_documents(
                rerank_text,
                query_vector,
                text_hits,
            )

        hits = _merge_with_evidence_precedence(
            text_hits,
            evidence_hits,
            max_hits=settings.max_hits,
        )
        hits.sort(key=lambda h: h.score, reverse=True)

        t5 = time.perf_counter()

        if not hits:
            return ChatResponse(
                answer=(
                    "Jag hittar inget tydligt stöd i de indexerade dokumenten "
                    "för att besvara frågan."
                ),
                sources=[],
                debug={
                    "selection": {
                        "min_relevance_floor": settings.min_relevance_floor,
                        "relevance_ratio": settings.relevance_ratio,
                        "max_hits": settings.max_hits,
                        "top_score": round(all_reranked[0].score, 3) if all_reranked else None,
                    },
                    "num_semantic": len(semantic_hits),
                    "num_bm25": len(bm25_hits),
                    "num_candidates": len(candidates),
                    "num_expanded": num_expanded,
                    "expanded_docs": expanded_doc_paths,
                    "num_evidence_candidates": len(evidence_hits),
                    "evidence_docs": evidence_source_paths,
                    "abstained": True,
                    "qud_anchor_used": qud_anchor is not None,
                    "retrieval_question_used": retrieval_question is not None,
                    "synonym_additions": synonym_additions,
                    "preferred_source_paths": preferred_source_paths,
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

        # 8. Syntes: enstegsformulering direkt från källorna
        t6 = time.perf_counter()

        hits_for_synthesis = _select_hits_for_synthesis(hits)
        
        synthesis_result = synthesize(
            question,
            hits_for_synthesis,
            self.llm,
            background_turns=background_turns,
            background_max_turns=background_max_turns,
        )
        t7 = time.perf_counter()
        
        related_concepts = self._related_concepts_in_selected_docs(question, hits)
        if related_concepts:
            synthesis_result.answer = (
                synthesis_result.answer.rstrip()
                + "\n\nRelaterade begrepp: "
                + ", ".join(related_concepts)
                + "."
            )
        t8 = time.perf_counter()

        # Bygg debug-info för syntesen
        synthesis_debug = {
            "used_fallback": synthesis_result.used_fallback,
        }
        if synthesis_result.fallback_reason:
            synthesis_debug["fallback_reason"] = synthesis_result.fallback_reason
        if synthesis_result.verification is not None:
            synthesis_debug["num_findings"] = len(synthesis_result.verification.findings)
            if synthesis_result.verification.raw_json:
                synthesis_debug["verification_json"] = synthesis_result.verification.raw_json
        if synthesis_result.timing_s:
            synthesis_debug["timing_s"] = synthesis_result.timing_s

        return ChatResponse(
            answer=synthesis_result.answer,
            sources=hits_for_synthesis,
            debug={
                "selection": {
                    "min_relevance_floor": settings.min_relevance_floor,
                    "relevance_ratio": settings.relevance_ratio,
                    "max_hits": settings.max_hits,
                    "top_score": round(all_reranked[0].score, 3) if all_reranked else None,
                    "cutoff_used": round(
                        max(
                            settings.min_relevance_floor,
                            all_reranked[0].score * settings.relevance_ratio,
                        ),
                        3,
                    ) if all_reranked else None,
                },
                "synthesis_input": {
                    "num_hits_for_synthesis": len(hits_for_synthesis),
                    "synthesis_source_sections": [
                        {
                            "file_name": h.metadata.file_name,
                            "section_title": h.metadata.section_title,
                            "score": round(h.score, 4),
                        }
                        for h in hits_for_synthesis
                    ],
                },                
                "num_semantic": len(semantic_hits),
                "num_bm25": len(bm25_hits),
                "num_candidates": len(candidates),
                "num_expanded": num_expanded,
                "num_evidence_candidates": len(evidence_hits),
                "evidence_docs": evidence_source_paths,
                "num_reranked": len(all_reranked),
                "num_hits": len(hits),
                "abstained": False,
                "synonym_additions": synonym_additions,
                "related_concepts": related_concepts,
                
                "synthesis": synthesis_debug,
                "timing_s": {
                    "embed_query": round(t1 - t0, 3),
                    "search": round(t2 - t1, 3),
                    "bm25_and_merge": round(t3 - t2, 3),
                    "rerank_1": round(t4 - t3, 3),
                    "expand_and_rerank_2": round(t5 - t4, 3),
                    "synthesize": round(t7 - t6, 3),
                    "related_concepts":  round(t8 - t7, 3),
                    "total": round(t8 - t0, 3),
                },
                "rerank_top": sorted(
                    rerank_debug,
                    key=lambda d: d.get("cross_encoder_score", -999),
                    reverse=True,
                )[: settings.max_hits + 5],
                "evidence_top": sorted(
                    evidence_debug,
                    key=lambda d: d.get("cross_encoder_score", -999),
                    reverse=True,
                )[: 6],
                "evidence_boost": sorted(
                    evidence_boost_debug,
                    key=lambda d: d.get("boosted_score", -999),
                    reverse=True,
                )[: 6],
            },
        )

    def rework(
        self,
        question: str,
        hits: list[SourceHit],
        previous_answer: str,
        mode: str,
        qud_question: str | None = None,
        consumed_hit_ids: set[str] | None = None,
    ) -> ChatResponse:
        """
        Arbeta mot föregående turs källor utan ny huvudretrieval.

        mode styr vilken rework-funktion som används:
        - "elaboration": hämta ny retrieval inom samma dokument som bar
          föregående svar och formulera ett tillägg. Kräver qud_question
          (originalfrågan) som söktext för den nya rankningen.
        - "verification": strikt granskning av tidigare svar mot
          föregående källor, ingen ny retrieval.

        För elaboration visas de NYA hits (de som faktiskt bar tillägget)
        som sources i svaret, så att UI:ts källhänvisningar stämmer med
        [Källa N] i svarstexten. För verification visas de ursprungliga
        hits eftersom granskningen arbetar mot dem.
        """
        t0 = time.perf_counter()

        if mode == "elaboration":
            search_question = qud_question or question
            new_hits = self.retrieve_for_elaboration(
                search_question,
                hits,
                consumed_hit_ids=consumed_hit_ids or set(),
            )

            t1 = time.perf_counter()

            synthesis_result = elaborate(
                question,
                new_hits,
                previous_answer,
                self.llm,
            )
            sources_to_show = new_hits if new_hits else hits
            num_new = len(new_hits)

        elif mode == "verification":
            new_hits = []
            t1 = time.perf_counter()

            synthesis_result = verify(
                question,
                hits,
                previous_answer,
                self.llm,
            )
            sources_to_show = hits
            num_new = 0

        else:
            raise ValueError(f"Okänt rework-läge: {mode!r}")

        t2 = time.perf_counter()

        synthesis_debug = {
            "used_fallback": synthesis_result.used_fallback,
            "mode": mode,
        }
        if synthesis_result.fallback_reason:
            synthesis_debug["fallback_reason"] = synthesis_result.fallback_reason
        if synthesis_result.verification is not None:
            report = synthesis_result.verification
            synthesis_debug["num_findings"] = len(report.findings)
            synthesis_debug["status_counts"] = {
                status: sum(1 for f in report.findings if f.status == status)
                for status in ("supported", "unclear", "unsupported")
            }
            if report.raw_json:
                synthesis_debug["verification_json"] = report.raw_json
        if synthesis_result.timing_s:
            synthesis_debug["timing_s"] = synthesis_result.timing_s

        # Abstain-bedömning skiljer sig mellan elaboration och verification.
        # Elaboration: om ingen ny retrieval gav något eller om elaborate()
        #   själv returnerade den tomma-nya-hits-formuleringen.
        # Verification: om parsningen misslyckades eller inga findings.
        if mode == "elaboration":
            abstained = not new_hits
        else:
            abstained = (
                synthesis_result.used_fallback
                or synthesis_result.verification is None
                or not synthesis_result.verification.findings
            )

        return ChatResponse(
            answer=synthesis_result.answer,
            sources=sources_to_show,
            debug={
                "rework_mode": mode,
                "num_hits_reused": len(hits),
                "num_new_hits": num_new,
                "num_consumed_hits": len(consumed_hit_ids or set()),
                "abstained": abstained,
                "synthesis": synthesis_debug,
                "timing_s": {
                    "retrieve_for_elaboration": round(t1 - t0, 3) if mode == "elaboration" else 0.0,
                    "rework": round(t2 - t1, 3),
                    "total": round(t2 - t0, 3),
                },
            },
        )

    def retrieve_for_elaboration(
        self,
        question: str,
        active_hits: list[SourceHit],
        consumed_hit_ids: set[str] | None = None,
    ) -> list[SourceHit]:
        """
        Hämta material som kan bära en elaboration av föregående svar.

        Hämtar alla chunks från de dokument som bar föregående svar,
        filtrerar bort de som redan finns i active_hits, och rerankar
        resten mot original-frågan (typiskt state.current_qud_text).

        Returnerar de rerankade hits som passerar cross-encoderns
        standardfilter (positiv score, ej boilerplate). Returnerar
        tom lista om inga nya relevanta chunks hittas — elaborate()
        hanterar då tomfallet med en ärlig abstain.
        """
        if not active_hits:
            return []

        active_doc_paths = {
            h.metadata.source_path for h in active_hits if h.metadata.source_path
        }
        if not active_doc_paths:
            return []

        active_ids = {h.chunk_id for h in active_hits}
        blocked_ids = active_ids | set(consumed_hit_ids or set())
        
        # Samla alla chunks från aktiva dokument som inte redan användes.
        # bm25_index innehåller bara textchunks, inte evidensobjekt.
        # Om active_hits är rent evidensobjekt kommer active_ids inte att
        # matcha något i candidates — då återanvänds ingen textchunk som
        # aktivt filter, men det är OK: elaboration ska lyfta fram
        # förklarande text omkring tabeller/listor som redan visades.
        candidates: list[SourceHit] = []
        for path in active_doc_paths:
            for chunk in self.bm25_index.get_chunks_by_source(path):
                if chunk.chunk_id in blocked_ids:
                    continue
                candidates.append(chunk)

        if not candidates:
            return []

        # Reranka mot originalfrågan med cross-encoderns standardfilter.
        reranked, _debug = self.reranker.rerank(question, candidates)

        # Välj hits med samma relevansbaserade urval som huvudvägen,
        # men utan sektionsdedup — elaboration ska lyfta fram just
        # det som föll bort, så vi vill inte sålla på samma nyckel igen.
        if not reranked:
            return []

        top_score = reranked[0].score
        cutoff = max(
            settings.min_relevance_floor,
            top_score * settings.relevance_ratio,
        )

        selected = [h for h in reranked if h.score >= cutoff][: settings.max_hits]
        return selected

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