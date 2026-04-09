import re
import time
from collections import Counter

from app.config import settings
from app.embeddings import Embedder
from app.qdrant_store import QdrantStore
from app.llm import LocalLLM
from app.prompting import build_prompt
from app.schemas import ChatResponse, SourceHit


_BOILERPLATE_SECTION_TITLES = {
    "bilaga",
    "delges",
    "sändlista",
    "sändlista:",
    "protokoll",
    "b e s l u t",
}

_INFORMATIONAL_SECTION_HINTS = {
    "rapport",
    "information",
    "övriga frågor",
    "övrigt",
    "sändlista",
    "delges",
    "bilaga",
}

_RULE_HINTS = {
    "delegation",
    "delegera",
    "beslut om delegation",
    "får besluta",
    "besluta om",
    "delegationsordning",
    "ansvarig för beslut",
    "tillgodoräknande",
    "rutin",
    "rutiner",
    "handläggning",
    "handläggningsordning",
    "instruktion",
    "anvisning",
    "anvisningar",
    "gäller för",
}

_ROLE_DESCRIPTION_HINTS = {
    "uppdrag",
    "arbetsuppgifter",
    "ansvarar för",
    "bereda ärenden",
    "granska dokument",
    "granska individuella studieplaner",
    "följa upp individuella studieplaner",
    "följa upp hur handledning fungerar",
    "koordinera utbudet",
    "administrera doktorandärenden",
    "föredragande i fun",
    "strategiskt stöd",
    "kvalitetssäkring avseende utbildning",
    "utbildningsmiljön",
    "stödja institutionsledning",
    "svara för information om forskarutbildningen",
    "bereda utlysning",
    "behörighetsprövning",
    "urval och antagning av nya doktorander",
}

_APPOINTMENT_CASE_HINTS = {
    "ta fram kandidat",
    "förslag på kandidat",
    "presentera kandidat",
    "ny studierektor",
    "får uppdrag",
    "utses",
    "utse",
    "förankrat inom avdelningen",
    "presenterat förslaget på institutionens ledningsråd",
    "ledningsråd",
    "ställer sig bakom förslaget",
    "kandidat till rollen",
    "återkomma med förslag",
}

_CASE_SPECIFIC_HINTS = {
    "för doktorand",
    "doktorand",
    "student",
    "i detta ärende",
    "denna doktorand",
    "tillgodoräknande för",
    "antagning av",
    "paria sadeghian",
}

_PRIMARY_RULE_SECTION_HINTS = {
    "beslut",
}

_PRIMARY_RULE_PREFIX_HINTS = {
    "anvisningar",
    "instruktion",
    "instruktioner",
    "rutin",
    "rutiner",
    "handläggningsordning",
}

_SECONDARY_RULE_SECTION_HINTS = {
    "beskrivning av ärendet",
    "beredning av ärendet",
    "uppföljningsmöte",
    "delges",
    "bilaga",
}

_PRIMARY_ROLE_SECTION_HINTS = {
    "uppdrag",
    "ansvar",
    "arbetsuppgifter",
}

_SECONDARY_ROLE_SECTION_HINTS = {
    "diskussion",
    "ta fram kandidat",
    "förslag på kandidat",
    "ny studierektor",
    "kandidat till rollen",
}

_RULE_QUESTION_TERMS = {
    "vem", "ansvar", "ansvarar", "besluta", "beslutar", "beslut",
    "delegation", "delegera", "regel", "regler", "rutin", "rutiner",
    "instruktion", "instruktioner", "gäller", "handläggning",
    "handläggningsordning", "får", "tillgodoräknande",
}

_ROLE_QUESTION_TERMS = {
    "uppdrag", "uppgifter", "roll", "roller", "arbetsuppgifter",
    "studierektor", "vilka uppdrag", "vilka arbetsuppgifter", "vad gör",
}

_PROCESS_QUESTION_TERMS = {
    "hur", "genomförs", "process", "förfarande", "steg",
    "förbereds", "anmäls", "planeras",
}


def _normalize_text(s: str | None) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", " ", s.strip()).casefold()


def _tokenize(text: str) -> set[str]:
    return {
        tok
        for tok in re.findall(r"\w+", text.casefold(), flags=re.UNICODE)
        if len(tok) >= 3
    }


def _contains_any_phrase(text: str, phrases: set[str]) -> bool:
    norm = _normalize_text(text)
    return any(p in norm for p in phrases)


def _looks_like_person_name(text: str) -> bool:
    return bool(re.search(r"\b[A-ZÅÄÖ][a-zåäö]+ [A-ZÅÄÖ][a-zåäö]+\b", text))


def _overlap_score(query_terms: set[str], values: list[str] | None) -> int:
    if not query_terms or not values:
        return 0
    score = 0
    for value in values:
        value_terms = _tokenize(value)
        if query_terms & value_terms:
            score += 1
    return score


def _title_overlap_bonus(query_terms: set[str], title: str | None, unit_bonus: float, cap: int = 3) -> tuple[float, int]:
    if not query_terms or not title:
        return 0.0, 0
    title_terms = _tokenize(title)
    overlap = len(query_terms & title_terms)
    if overlap <= 0:
        return 0.0, 0
    bonus = min(overlap, cap) * unit_bonus
    return bonus, overlap


def _document_type_matches(question_terms: set[str], document_type: str | None) -> bool:
    dt = _normalize_text(document_type)
    if not dt:
        return False

    if dt in question_terms:
        return True

    if "beslut" in dt and {"beslut", "besluta", "delegera", "delegation"} & question_terms:
        return True
    if ("instruktion" in dt or "anvisning" in dt) and {"instruktion", "anvisning", "rutiner", "gäller"} & question_terms:
        return True
    if "rutin" in dt and {"rutin", "rutiner", "handläggning"} & question_terms:
        return True
    if "protokoll" in dt and {"protokoll", "möte", "sammanträde"} & question_terms:
        return True

    return False


def _detect_question_intent(question: str) -> str:
    qn = _normalize_text(question)
    q_terms = _tokenize(question)

    if _contains_any_phrase(qn, _ROLE_QUESTION_TERMS) or (
        "studierektor" in q_terms and {"uppdrag", "uppgifter", "arbetsuppgifter"} & q_terms
    ):
        return "role_description"

    if _contains_any_phrase(qn, _PROCESS_QUESTION_TERMS):
        return "process"

    if q_terms & _RULE_QUESTION_TERMS:
        return "rule"

    return "generic"


def _is_boilerplate_section(title: str | None, text: str) -> bool:
    norm_title = _normalize_text(title)
    if norm_title in _BOILERPLATE_SECTION_TITLES:
        return True

    stripped = text.strip()
    if stripped == "<!-- image -->":
        return True

    if len(_tokenize(stripped)) <= 2 and len(stripped) < 40:
        return True

    return False


def _classify_section_function(hit: SourceHit) -> str:
    meta = hit.metadata
    title = meta.section_title or ""
    summary = meta.section_summary or ""
    text = hit.text or ""
    document_title = meta.document_title or ""
    document_type = meta.document_type or ""

    title_norm = _normalize_text(title)
    doc_title_norm = _normalize_text(document_title)
    doc_type_norm = _normalize_text(document_type)

    joined = " | ".join([
        title_norm,
        _normalize_text(summary),
        doc_title_norm,
        doc_type_norm,
        _normalize_text(" ".join(meta.keywords)),
        _normalize_text(" ".join(meta.roles)),
        _normalize_text(" ".join(meta.actions)),
        _normalize_text(" ".join(meta.applies_to)),
        _normalize_text(text[:800]),
    ])

    if _contains_any_phrase(title, _INFORMATIONAL_SECTION_HINTS) or _contains_any_phrase(summary, _INFORMATIONAL_SECTION_HINTS):
        return "informational"

    # 1. Exakta normativa sektioner först
    if title_norm == "beslut":
        return "rule"

    if any(title_norm.startswith(h) for h in _PRIMARY_RULE_PREFIX_HINTS):
        return "rule"

    # 2. Tydliga uppdrags-/rollbeskrivningar före personbaserade heuristiker
    if (
        "uppdrag" in doc_title_norm
        or "studierektorer fu uppdrag" in doc_title_norm
        or _contains_any_phrase(joined, _ROLE_DESCRIPTION_HINTS)
    ):
        # kandidat-/utnämningsspår ska fortfarande särskiljas
        if _contains_any_phrase(joined, _APPOINTMENT_CASE_HINTS):
            return "appointment_case"
        return "role_description"

    # 3. Tydliga delegations-/regelspår
    if _contains_any_phrase(joined, _RULE_HINTS):
        return "rule"

    # 4. Kandidat-/utnämningsärenden
    if _contains_any_phrase(joined, _APPOINTMENT_CASE_HINTS):
        return "appointment_case"

    # 5. Enskilda fall sist
    if _contains_any_phrase(joined, _CASE_SPECIFIC_HINTS):
        return "case_specific"

    if _looks_like_person_name(text):
        return "case_specific"

    return "unknown"


def _dedup_and_select(
    rescored_rows: list[tuple[float, SourceHit, dict, str]],
    top_k: int,
) -> tuple[list[SourceHit], list[dict]]:
    """
    1. Endast en träff per (source_path, section_title)
    2. Försök få variation i topp-K så att samma dokument inte dominerar
    """
    selected_hits: list[SourceHit] = []
    selected_debug: list[dict] = []

    seen_section_keys: set[tuple[str, str | None]] = set()
    doc_counter: Counter[str] = Counter()

    for _, hit, dbg, _ in rescored_rows:
        key = (hit.metadata.source_path, hit.metadata.section_title)
        if key in seen_section_keys:
            continue
        if doc_counter[hit.metadata.source_path] >= 1 and len(selected_hits) < top_k:
            continue

        seen_section_keys.add(key)
        doc_counter[hit.metadata.source_path] += 1
        selected_hits.append(hit)
        selected_debug.append(dbg)

        if len(selected_hits) >= top_k:
            return selected_hits, selected_debug

    for _, hit, dbg, _ in rescored_rows:
        key = (hit.metadata.source_path, hit.metadata.section_title)
        if key in seen_section_keys:
            continue

        seen_section_keys.add(key)
        selected_hits.append(hit)
        selected_debug.append(dbg)

        if len(selected_hits) >= top_k:
            break

    return selected_hits, selected_debug

def _section_title_type(title: str | None) -> str:
    norm = _normalize_text(title)

    if not norm:
        return "none"

    # Exakta normativa rubriker
    if norm in _PRIMARY_RULE_SECTION_HINTS:
        return "primary_rule"

    # Rubriker som börjar med normativa ord
    if any(norm.startswith(h) for h in _PRIMARY_RULE_PREFIX_HINTS):
        return "primary_rule"

    # Diskussions-/kandidatspår ska fångas tidigt
    if any(h in norm for h in _SECONDARY_ROLE_SECTION_HINTS):
        return "secondary_role"

    # "uppdrag" etc. ska bara vara primary_role om det inte är diskussion/kandidat
    if any(h in norm for h in _PRIMARY_ROLE_SECTION_HINTS):
        return "primary_role"

    if norm in _SECONDARY_RULE_SECTION_HINTS:
        return "secondary_rule"

    return "other"


def _section_title_priority_bonus(question_intent: str, hit: SourceHit) -> tuple[float, str | None]:
    title_type = _section_title_type(hit.metadata.section_title)
    norm_title = _normalize_text(hit.metadata.section_title)

    if question_intent == "rule":
        if title_type == "primary_rule":
            # Exakt "Beslut" ska väga extra tungt
            if norm_title == "beslut":
                return 0.18, "exact_beslut_section+0.18"
            return 0.12, "primary_rule_section+0.12"

        if title_type == "secondary_rule":
            # Beskrivning/beredning ska ned i regel-frågor
            if norm_title == "beskrivning av ärendet":
                return -0.12, "beskrivning_av_arendet-0.12"
            return -0.06, "secondary_rule_section-0.06"

        if title_type == "secondary_role":
            return -0.10, "secondary_role_section-0.10"

    if question_intent == "process":
        if title_type == "primary_rule":
            return 0.08, "primary_rule_section+0.08"
        if title_type == "secondary_rule":
            return -0.04, "secondary_rule_section-0.04"

    if question_intent == "role_description":
        if title_type == "primary_role":
            return 0.06, "primary_role_section+0.06"
        if title_type == "secondary_role":
            return -0.14, "secondary_role_section-0.14"
        if title_type == "secondary_rule":
            return -0.04, "secondary_rule_section-0.04"

    return 0.0, None


def _document_intro_bonus(question_intent: str, hit: SourceHit) -> tuple[float, str | None]:
    """
    Liten bonus för dokumentets tidiga chunkar när frågan är bred/generaliserande.
    Det approximera 'huvudsektion' utan att kräva nytt metadatafält.
    """
    idx = hit.metadata.chunk_index

    if idx > 2:
        return 0.0, None

    if question_intent == "rule":
        return 0.04, "early_chunk+0.04"
    if question_intent == "process":
        return 0.03, "early_chunk+0.03"
    if question_intent == "role_description":
        return 0.02, "early_chunk+0.02"

    return 0.0, None
    
def _rerank_hits(question: str, hits: list[SourceHit]) -> tuple[list[SourceHit], list[dict]]:
    question_terms = _tokenize(question)
    question_intent = _detect_question_intent(question)

    rescored: list[tuple[float, SourceHit, dict, str]] = []

    for hit in hits:
        meta = hit.metadata
        score = float(hit.score)
        reasons: list[str] = []

        keyword_hits = _overlap_score(question_terms, meta.keywords)
        if keyword_hits:
            bonus = min(keyword_hits, 3) * settings.rerank_keyword_bonus
            score += bonus
            reasons.append(f"keywords+{bonus:.2f}")

        role_hits = _overlap_score(question_terms, meta.roles)
        if role_hits:
            bonus = min(role_hits, 2) * settings.rerank_role_bonus
            score += bonus
            reasons.append(f"roles+{bonus:.2f}")

        action_hits = _overlap_score(question_terms, meta.actions)
        if action_hits:
            bonus = min(action_hits, 2) * settings.rerank_action_bonus
            score += bonus
            reasons.append(f"actions+{bonus:.2f}")

        summary_match = False
        if meta.section_summary:
            summary_terms = _tokenize(meta.section_summary)
            if question_terms & summary_terms:
                summary_match = True
                score += settings.rerank_summary_bonus
                reasons.append(f"summary+{settings.rerank_summary_bonus:.2f}")

        if _document_type_matches(question_terms, meta.document_type):
            score += settings.rerank_document_type_bonus
            reasons.append(f"document_type+{settings.rerank_document_type_bonus:.2f}")

        section_title_bonus, section_title_overlap = _title_overlap_bonus(
            question_terms,
            meta.section_title,
            settings.rerank_section_title_bonus,
            cap=3,
        )
        if section_title_bonus:
            score += section_title_bonus
            reasons.append(f"section_title+{section_title_bonus:.2f}")

        document_title_bonus, document_title_overlap = _title_overlap_bonus(
            question_terms,
            meta.document_title,
            settings.rerank_document_title_bonus,
            cap=3,
        )
        if document_title_bonus:
            score += document_title_bonus
            reasons.append(f"document_title+{document_title_bonus:.2f}")

        section_priority_bonus, section_priority_reason = _section_title_priority_bonus(question_intent, hit)
        if section_priority_bonus:
            score += section_priority_bonus
            reasons.append(section_priority_reason)

        intro_bonus, intro_reason = _document_intro_bonus(question_intent, hit)
        if intro_bonus:
            score += intro_bonus
            reasons.append(intro_reason)
            
        if _is_boilerplate_section(meta.section_title, hit.text):
            score -= settings.rerank_boilerplate_penalty
            reasons.append(f"boilerplate-{settings.rerank_boilerplate_penalty:.2f}")

        section_function = _classify_section_function(hit)

        if question_intent == "rule":
            if section_function == "rule":
                score += 0.14
                reasons.append("rule+0.14")
            elif section_function == "role_description":
                score -= 0.04
                reasons.append("role_description-0.04")
            elif section_function == "appointment_case":
                score -= 0.12
                reasons.append("appointment_case-0.12")
            elif section_function == "case_specific":
                score -= 0.14
                reasons.append("case_specific-0.14")
            elif section_function == "informational":
                score -= 0.05
                reasons.append("informational-0.05")

        elif question_intent == "role_description":
            if section_function == "role_description":
                score += 0.16
                reasons.append("role_description+0.16")
            elif section_function == "rule":
                score += 0.01
                reasons.append("rule_support+0.01")
            elif section_function == "appointment_case":
                score -= 0.18
                reasons.append("appointment_case-0.18")
            elif section_function == "case_specific":
                score -= 0.10
                reasons.append("case_specific-0.10")
            elif section_function == "informational":
                score -= 0.05
                reasons.append("informational-0.05")

        elif question_intent == "process":
            if section_function in {"rule", "role_description"}:
                score += 0.05
                reasons.append("process_support+0.05")
            elif section_function in {"appointment_case", "case_specific"}:
                score -= 0.04
                reasons.append("case_like-0.04")
            elif section_function == "informational":
                score -= 0.04
                reasons.append("informational-0.04")

        topical_signals = (
            section_title_overlap
            + document_title_overlap
            + keyword_hits
            + role_hits
            + action_hits
            + (1 if summary_match else 0)
        )

        if question_intent in {"rule", "role_description"} and section_function in {"rule", "role_description"} and topical_signals == 0:
            score -= 0.08
            reasons.append("weak_topical_match-0.08")

        debug_row = {
            "file_name": meta.file_name,
            "section_title": meta.section_title,
            "base_score": round(hit.score, 4),
            "reranked_score": round(score, 4),
            "document_type": meta.document_type,
            "semantic_enriched": meta.semantic_enriched,
            "question_intent": question_intent,
            "section_function": section_function,
            "section_title_overlap": section_title_overlap,
            "section_title_type": _section_title_type(meta.section_title),            
            "document_title_overlap": document_title_overlap,
            "keyword_hits": keyword_hits,
            "role_hits": role_hits,
            "action_hits": action_hits,
            "summary_match": summary_match,
            "reasons": reasons,
        }

        rescored.append((score, hit, debug_row, section_function))

    rescored.sort(
        key=lambda row: (
            row[0],
            len(row[1].metadata.keywords),
            1 if row[1].metadata.section_title else 0,
        ),
        reverse=True,
    )

    selected_hits, selected_debug = _dedup_and_select(rescored, settings.top_k)
    return selected_hits, selected_debug

_GENERIC_SECTION_TITLES = {
    "beskrivning av ärendet",
    "beredning av ärendet",
    "beslut",
    "delges",
    "bilaga",
    "protokoll",
}


def _normalize_text_for_guard(s: str | None) -> str:
    if not s:
        return ""
    s = s.casefold()
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _is_generic_section_title(title: str | None) -> bool:
    norm = _normalize_text_for_guard(title)
    if not norm:
        return True
    if norm in _GENERIC_SECTION_TITLES:
        return True
    # fånga "Beslut:" och liknande
    if norm.startswith("beslut"):
        return True
    return False


def _question_terms(question: str) -> set[str]:
    q = _normalize_text_for_guard(question)
    parts = re.findall(r"[a-zA-ZåäöÅÄÖ]{4,}", q)
    stop = {
        "vilken", "vilka", "gäller", "finns", "detta", "denna",
        "inom", "eller", "enligt", "fråga", "process", "rutinen",
        "rutin", "beslut", "ärende", "gäller", "gällde",
        "tillsättande", "tillsättning",
    }
    return {p for p in parts if p not in stop}


def _document_topic_overlap(question: str, hit) -> int:
    terms = _question_terms(question)
    if not terms:
        return 0

    meta = hit.metadata
    hay = " ".join(
        [
            getattr(meta, "file_name", "") or "",
            getattr(meta, "document_title", "") or "",
            getattr(meta, "section_title", "") or "",
            getattr(meta, "section_summary", "") or "",
            " ".join(getattr(meta, "keywords", []) or []),
            " ".join(getattr(meta, "roles", []) or []),
            " ".join(getattr(meta, "actions", []) or []),
            " ".join(getattr(meta, "applies_to", []) or []),
        ]
    )
    hay_norm = _normalize_text_for_guard(hay)
    return sum(1 for t in terms if t in hay_norm)


def _has_sufficient_support(question: str, hits, rerank_debug: list[dict]) -> tuple[bool, dict]:
    if not hits or not rerank_debug:
        return False, {"reason": "no_hits"}

    top = rerank_debug[: min(3, len(rerank_debug))]

    strong_topic_match = False
    any_semantic_support = False
    all_generic = True

    for hit, dbg in zip(hits[: len(top)], top):
        topic_overlap = _document_topic_overlap(question, hit)
        keyword_hits = dbg.get("keyword_hits", 0) or 0
        role_hits = dbg.get("role_hits", 0) or 0
        action_hits = dbg.get("action_hits", 0) or 0
        summary_match = bool(dbg.get("summary_match", False))
        section_title_overlap = dbg.get("section_title_overlap", 0) or 0
        document_title_overlap = dbg.get("document_title_overlap", 0) or 0

        if topic_overlap >= 1:
            strong_topic_match = True
        elif section_title_overlap >= 2:
            strong_topic_match = True
        elif document_title_overlap >= 2:
            strong_topic_match = True

        if keyword_hits > 0 or role_hits > 0 or action_hits > 0 or summary_match:
            any_semantic_support = True

        if not _is_generic_section_title(hit.metadata.section_title):
            all_generic = False

    qnorm = _normalize_text_for_guard(question)
    is_process_question = "process" in qnorm or "hur" in qnorm or "tillsätt" in qnorm

    if not strong_topic_match and not any_semantic_support:
        return False, {
            "reason": "no_topic_support",
            "strong_topic_match": strong_topic_match,
            "any_semantic_support": any_semantic_support,
            "all_generic_sections": all_generic,
        }

    if is_process_question and not any_semantic_support:
        return False, {
            "reason": "process_without_semantic_support",
            "strong_topic_match": strong_topic_match,
            "any_semantic_support": any_semantic_support,
            "all_generic_sections": all_generic,
        }

    if all_generic and not any_semantic_support:
        return False, {
            "reason": "generic_hits_only",
            "strong_topic_match": strong_topic_match,
            "any_semantic_support": any_semantic_support,
            "all_generic_sections": all_generic,
        }

    return True, {
        "reason": "sufficient_support",
        "strong_topic_match": strong_topic_match,
        "any_semantic_support": any_semantic_support,
        "all_generic_sections": all_generic,
    }

class RagService:
    def __init__(self) -> None:
        self.embedder = Embedder()
        test_vec = self.embedder.embed_query("test")
        self.store = QdrantStore(vector_size=len(test_vec))
        self.llm = LocalLLM()

    def answer(self, question: str) -> ChatResponse:
        t0 = time.perf_counter()
        query_vector = self.embedder.embed_query(question)
        t1 = time.perf_counter()
    
        candidate_hits = self.store.search(
            query_vector,
            limit=max(settings.top_k, settings.retrieval_candidate_k),
        )
        t2 = time.perf_counter()
    
        hits, rerank_debug = _rerank_hits(question, candidate_hits)
        t3 = time.perf_counter()
    
        support_ok, support_debug = _has_sufficient_support(question, hits, rerank_debug)
    
        if not support_ok:
            t4 = time.perf_counter()
            answer = (
                "Jag hittar inget tydligt stöd i de indexerade dokumenten för att besvara frågan. "
                "De närmaste träffarna verkar handla om andra beslut, rutiner eller uppdrag än det du frågar om."
            )
            return ChatResponse(
                answer=answer,
                sources=hits,
                debug={
                    "top_k": settings.top_k,
                    "candidate_k": max(settings.top_k, settings.retrieval_candidate_k),
                    "num_candidates": len(candidate_hits),
                    "num_hits": len(hits),
                    "abstained": True,
                    "support_check": support_debug,
                    "timing_s": {
                        "embed_query": round(t1 - t0, 3),
                        "search": round(t2 - t1, 3),
                        "rerank": round(t3 - t2, 3),
                        "generate": 0.0,
                        "total": round(t4 - t0, 3),
                    },
                    "rerank_top": rerank_debug[: settings.top_k + 3],
                },
            )
    
        prompt = build_prompt(question, hits)
        t4 = time.perf_counter()
    
        answer = self.llm.generate(prompt)
        t5 = time.perf_counter()
    
        return ChatResponse(
            answer=answer,
            sources=hits,
            debug={
                "top_k": settings.top_k,
                "candidate_k": max(settings.top_k, settings.retrieval_candidate_k),
                "num_candidates": len(candidate_hits),
                "num_hits": len(hits),
                "abstained": False,
                "support_check": support_debug,
                "timing_s": {
                    "embed_query": round(t1 - t0, 3),
                    "search": round(t2 - t1, 3),
                    "rerank": round(t3 - t2, 3),
                    "build_prompt": round(t4 - t3, 3),
                    "generate": round(t5 - t4, 3),
                    "total": round(t5 - t0, 3),
                },
                "prompt_chars": len(prompt),
                "rerank_top": rerank_debug[: settings.top_k + 3],
            },
        )