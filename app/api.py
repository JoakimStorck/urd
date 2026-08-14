import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.config_validation import validate_config_files, format_report_lines
from app.schemas import ChatRequest, ChatResponse, SourceHit
from app.retrieval import RagService
from app.session_state import SessionStore
from app.intent import classify_utterance, Classification
from app.social import handle_social
from app.qud_drift import measure_drift
from app.followup import rewrite_followup
from app.question_rules import rule_based_operation

logger = logging.getLogger(__name__)

# Uvicorn konfigurerar bara sina egna loggers. Appens loggers (den här
# modulens, loadernas i synonyms/concepts/question_operations, llm:s
# trunkeringsvarning) hamnar på rotloggern som saknar handler — och
# försvinner då tyst. Det underminerar hela poängen med synlig
# konfigstatus. Konfigurera därför rotloggern här, men bara om ingen
# handler redan finns (så att en inbäddande process kan styra själv).
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:     %(message)s",
    )

# Tredjepartsbibliotek som loggar varje HTTP-anrop på INFO-nivå
# dränker appens egna rader. Dämpa dem till WARNING — deras fel
# ska fortfarande synas, men inte deras vardagsprat.
for _noisy in ("httpx", "huggingface_hub", "urllib3", "filelock"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

app = FastAPI(title="Local IIT URD")
rag = RagService()
sessions = SessionStore()

# Validera instansens konfigurationsfiler vid uppstart. Trasiga filer
# stoppar inte servern (tyst fallback i loaders gäller fortfarande),
# men status ska alltid vara synlig: en rad per fil i startloggen och
# samma bild i /health. Se app/config_validation.py för bakgrund.
_config_report = validate_config_files(
    synonyms_path=settings.synonyms_path,
    concepts_path=settings.concepts_path,
    question_operations_path=settings.question_operations_path,
)
for _line in format_report_lines(_config_report):
    if _line.startswith("[FEL]") or _line.strip().startswith("fel:"):
        logger.warning("konfig: %s", _line)
    else:
        logger.info("konfig: %s", _line)
if not _config_report.ok:
    logger.warning(
        "konfig: en eller flera konfigurationsfiler har FEL — kör "
        "'urd config validate' för detaljer. Berörda funktioner är avstängda."
    )

# Aktiv modell och LLM-inställningar i startloggen.
#
# Modellen sätts i .urd/config.json men kan överstyras av OLLAMA_MODEL,
# och think/num_ctx avgör både svarskvalitet och svarstid. Utan den här
# raden går det inte att se vad som faktiskt kör: 2026-08-14 kostade det
# en hel testkörning att upptäcka att gemma4:12b hade resonemang påslaget
# (107,6 s median per tur mot 9,0 s med det avstängt), och en körning
# till att reda ut om en modellväxling ens hade slagit igenom.
logger.info(
    "modell: %s | think=%s | num_ctx=%d | enrich-modell=%s",
    settings.ollama_model,
    settings.llm_think,
    settings.llm_num_ctx,
    settings.preprocess_ollama_model,
)

static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Resolve docs root en gång vid uppstart
_docs_root = settings.docs_path.resolve()


def select_active_hits(hits: list[SourceHit], max_hits: int = 3) -> list[SourceHit]:
    """
    Välj de träffar som blir samtalets aktiva kontext (QUD-underlag,
    rework-material, driftmätningens dokumentreferens).

    Åtgärd 4.1: inget toppdokumentlås. Tidigare kastades alla träffar
    från andra dokument än topphiten, vilket gjorde att flerdokumentsvar
    (jämförelser, aggregeringar) fick en aktiv kontext som bara täckte
    ett av dokumenten — följdfrågor mot de andra dokumenten tappade då
    sitt underlag. Sannolikhetsgolvet 0.5 behålls: bara träffar som är
    mer sannolikt relevanta än inte får ingå.
    """
    if not hits:
        return []

    selected = [hits[0]]

    for hit in hits[1:]:
        if len(selected) >= max_hits:
            break
        # Sannolikhetsskala: bara träffar som är mer sannolikt
        # relevanta än inte får ingå i rework-underlaget.
        if hit.score < 0.5:
            continue
        selected.append(hit)

    return selected
    
@app.get("/")
def index():
    return FileResponse(static_dir / "index.html")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "config_files": _config_report.as_dict(),
        # Klienten (och urd connect) ska kunna se vad servern faktiskt
        # kör utan att läsa serverloggen.
        "llm": {
            "model": settings.ollama_model,
            "think": settings.llm_think,
            "num_ctx": settings.llm_num_ctx,
            "enrich_model": settings.preprocess_ollama_model,
        },
    }


@app.post("/refresh")
def refresh() -> dict[str, int]:
    """Återbygg BM25-index efter ingest. Anropas av CLI."""
    num_chunks = rag.refresh_index()
    return {"status": "ok", "num_chunks": num_chunks}


@app.get("/document")
def get_document(path: str = Query(..., description="Relativ sökväg under docs/")):
    """
    Servera ett originaldokument. Validerar att sökvägen pekar
    in i docs-katalogen för att förhindra path traversal.
    """
    resolved = (_docs_root / path).resolve()

    if not resolved.is_relative_to(_docs_root):
        raise HTTPException(status_code=404, detail="Dokumentet hittades inte.")

    if not resolved.is_file():
        raise HTTPException(status_code=404, detail="Dokumentet hittades inte.")

    return FileResponse(resolved, filename=resolved.name)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    try:
        state = sessions.get_or_create(req.session_id)

        # 1. Klassificera yttringen inom QUD-modellen.
        classification = classify_utterance(req.question, state, rag.llm)

        # 1a. Regelbaserad föroperation: för frågeoperationer med
        # entydiga språkliga markörer (comparison, aggregation) avgör
        # deterministiska regler över LLM-klassificeringen. Intent
        # berörs inte. Se question_rules.py.
        operation_source = "llm"
        rule_operation = rule_based_operation(req.question)
        if rule_operation is not None:
            if rule_operation != classification.question_operation:
                classification.question_operation = rule_operation  # type: ignore[assignment]
                operation_source = "rule_override"
            else:
                operation_source = "rule_confirmed"

        # 1b. QUD-drift-skydd: om klassificeraren säger related_to_qud
        # men aktuell yttring ligger semantiskt långt från aktiv QUD,
        # tolka om till new_main_question. Detta fångar fall där
        # samtalet bytt ämne utan att klassificeraren märkt det, vilket
        # annars skulle leda till kontaminerad retrieval (QUD-ankare mot
        # fel ämne) och typiskt till abstain.
        drift: object | None = None
        if classification.intent == "related_to_qud" and state.current_qud_text:
            drift = measure_drift(
                req.question,
                state.current_qud_text,
                rag.embedder,
                threshold=settings.qud_drift_threshold,
                # Dokumentbaserad drift: jämför yttringen även mot
                # texterna som bar de senaste svaren (fråga-mot-
                # passage). När sådana finns avgör de beslutet —
                # se qud_drift.py för motiveringen.
                active_hit_texts=[h.text for h in state.active_hits],
                doc_threshold=settings.qud_drift_doc_threshold,
            )
            if drift is not None and drift.drift_detected:
                classification = Classification(
                    intent="new_main_question",
                    substyle=None,
                    reason=(
                        f"qud_drift_detected (similarity={drift.similarity} "
                        f"< threshold={drift.threshold})"
                    ),
                    question_operation=classification.question_operation,
                    raw=classification.raw,
                    used_fallback=False,
                )

        matched_concept_ids = rag.concepts.find_matching_concept_ids(req.question)
        matched_concept_labels = rag.concepts.labels_for_concept_ids(matched_concept_ids)
        relation_pair_ids = rag.concepts.first_two_matching_concept_ids(req.question)
        relation_pair_labels = rag.concepts.labels_for_concept_ids(relation_pair_ids)
        
        # Grund-debug som alla vägar lägger till
        base_debug = {
            "session_id": state.session_id,
            "classification": {
                "intent": classification.intent,
                "substyle": classification.substyle,
                "question_operation": classification.question_operation,
                "operation_source": operation_source,
                "reason": classification.reason,
                "used_fallback": classification.used_fallback,
            },
            "concepts": {
                "matched_ids": matched_concept_ids,
                "matched_labels": matched_concept_labels,
                "relation_pair_ids": relation_pair_ids,
                "relation_pair_labels": relation_pair_labels,
            },            
            "qud": {
                "text": state.current_qud_text,
                "age_turns": state.qud_age_turns,
            },
            "rework_state": {
                "num_active_hits": len(state.active_hits),
                "num_consumed_hits": len(state.consumed_hit_ids),
            },            
        }

        if drift is not None:
            base_debug["qud_drift"] = {
                "similarity": drift.similarity,
                "threshold": drift.threshold,
                "doc_similarity": drift.doc_similarity,
                "doc_threshold": drift.doc_threshold,
                "decided_by": drift.decided_by,
                "drift_detected": drift.drift_detected,
            }

        # 2. Dispatcha baserat på intent.

        # 2a. Social/meta: inget retrieval, inget QUD-påverkan.
        if classification.intent == "social_or_meta":
            answer_text = handle_social(req.question, state, rag.llm)
            state.add_social_turn(req.question, answer_text)

            return ChatResponse(
                answer=answer_text,
                sources=[],
                session_id=state.session_id,
                debug={
                    **base_debug,
                    "path": "social_or_meta",
                },
            )

        # 2b. Elaboration och verification: arbetar mot active_hits från
        # föregående tur. Elaboration gör ny reranking inom aktiva
        # dokument för att hitta material som inte användes första
        # gången; verification arbetar direkt mot active_hits.
        # Skyddsregeln i intent.py har redan garanterat att active_hits
        # inte är tom här.
        if classification.intent in ("elaboration", "verification_or_challenge"):
            mode = (
                "elaboration"
                if classification.intent == "elaboration"
                else "verification"
            )
            previous_answer = state.last_answer or ""

            response = rag.rework(
                req.question,
                hits=state.active_hits,
                previous_answer=previous_answer,
                mode=mode,
                qud_question=state.current_qud_text,
                consumed_hit_ids=state.consumed_hit_ids,
            )

            # Rework-tur: ersätt INTE active_hits — samma material bär
            # fortfarande tråden. Bara last_answer och snippets uppdateras.
            state.add_rework_turn(
                req.question,
                response.answer,
                mode=mode,
                hits=response.sources,
            )

            if response.debug is None:
                response.debug = {}
            response.debug.update(base_debug)
            response.debug["path"] = classification.intent

            response.session_id = state.session_id
            return response

        # Spara föregående QUD innan den ev. skrivs över — den behövs
        # av den kontextuella fallbacken nedan, som ger en tur som
        # berövats sin kontext (falsk drift, klassificerarflipp) en
        # andra chans MED kontexten innan systemet abstainar.
        prev_qud_text = state.current_qud_text
        prev_qud_index = state.current_qud_turn_index

        # 2c. Ny huvudfråga: sätt QUD till ordagrann originaltext FÖRE
        # retrieval, så att den registreras även om den här turen
        # inte använder QUD-ankaret.
        if classification.intent == "new_main_question":
            state.set_qud(req.question)
            base_debug["qud"] = {
                "text": state.current_qud_text,
                "age_turns": state.qud_age_turns,
            }

        # 2d. Bestäm retrieval- och syntesparametrar för de två
        # kvarvarande klasserna (new_main_question, related_to_qud).
        qud_anchor: str | None = None
        background_turns = None
        background_max_turns = 0
        retrieval_question: str | None = None
        preferred_source_paths: list[str] | None = None

        if classification.intent == "new_main_question":
            # Standard retrieval, ingen bakgrund.
            path_label = "new_main_question"

        elif classification.intent == "related_to_qud":
            # QUD-ankare i retrieval + bakgrund i syntes
            qud_anchor = state.current_qud_text
            background_turns = list(state.turns)
            background_max_turns = settings.qud_background_turns
            path_label = "related_to_qud"

            # Broadening: skriv om den korta följdfrågan till en
            # fristående retrievalfråga. De dokument som bar
            # föregående svar skickas med som PREFERENS — retrieval
            # söker globalt och kompletterar med en ankrad pool
            # (se RagService.answer), så att broadening kan nå
            # dokument utanför den aktiva kontexten.
            if classification.substyle == "broadening":
                retrieval_question, was_rewritten = rewrite_followup(
                    req.question,
                    state,
                    rag.llm,
                )
                if not was_rewritten:
                    retrieval_question = None

                if state.active_doc_paths:
                    preferred_source_paths = list(state.active_doc_paths)

        else:
            # Skulle inte hända — alla klasser är hanterade ovan.
            path_label = "new_main_question"

        response = rag.answer(
            req.question,
            qud_anchor=qud_anchor,
            background_turns=background_turns,
            background_max_turns=background_max_turns,
            retrieval_question=retrieval_question,
            preferred_source_paths=preferred_source_paths,
            question_operation=classification.question_operation,
            matched_concept_ids=matched_concept_ids,
        )

        # Kontextuell fallback vid abstain. En elliptisk följdfråga
        # ("Vad gäller för medfinansiering?") är per definition bara
        # begriplig mot samtalets aktiva huvudfråga. Om en sådan tur
        # har berövats sin kontext — genom drift-överridning eller en
        # klassificerarflipp till new_main_question — och det
        # kontextlösa försöket abstainar, körs retrieval om EN gång
        # med föregående QUD som ankare och samtalsbakgrund, innan
        # systemet ger upp. Fallbacken kan aldrig göra utfallet sämre
        # (den aktiveras bara när alternativet är ett tomt svar) och
        # cross-encodern bedömer fortfarande mot den rena frågan, så
        # kontexten breddar kandidatpoolen utan att förvränga
        # relevansbedömningen.
        #
        # Villkor: första försöket abstainade, det finns en tidigare
        # QUD att ankra mot, och första försöket saknade antingen
        # QUD-ankare eller körde med omskriven retrievalfråga (vars
        # omskrivning kan ha varit problemet).
        context_fallback: dict | None = None
        if (
            (response.debug or {}).get("abstained")
            and prev_qud_text
            and (qud_anchor is None or retrieval_question is not None)
        ):
            retry = rag.answer(
                req.question,
                qud_anchor=prev_qud_text,
                background_turns=list(state.turns),
                background_max_turns=settings.qud_background_turns,
                question_operation=classification.question_operation,
                matched_concept_ids=matched_concept_ids,
            )
            rescued = not (retry.debug or {}).get("abstained", False)
            context_fallback = {
                "triggered": True,
                "rescued": rescued,
                "prev_qud": prev_qud_text,
            }
            if rescued:
                response = retry
                # Turen visade sig vara kontextberoende — den föregående
                # huvudfrågan är fortfarande samtalets QUD. Återställ den
                # så att nästa tur ankras rätt.
                if classification.intent == "new_main_question":
                    state.current_qud_text = prev_qud_text
                    state.current_qud_turn_index = prev_qud_index
                    base_debug["qud"] = {
                        "text": state.current_qud_text,
                        "age_turns": state.qud_age_turns,
                    }

        # Uppdatera sessionsstate med dokumentkällorna OCH de faktiska
        # hits som bar svaret — så att nästa elaboration/verification
        # kan återanvända dem.
        active_hits = select_active_hits(response.sources)
        
        doc_paths = list({
            hit.metadata.source_path
            for hit in active_hits
        })
        
        state.add_turn(
            req.question,
            response.answer,
            doc_paths,
            hits=active_hits,
        )

        # Merga debug-info från retrieval/syntes med vår dispatch-info
        if response.debug is None:
            response.debug = {}
        response.debug.update(base_debug)
        response.debug["path"] = path_label
        if context_fallback is not None:
            response.debug["context_fallback"] = context_fallback
        if background_max_turns > 0:
            response.debug["background_max_turns"] = background_max_turns
        if retrieval_question is not None:
            response.debug["retrieval_question"] = retrieval_question
        if preferred_source_paths is not None:
            response.debug["preferred_source_paths"] = preferred_source_paths

        response.session_id = state.session_id

        return response
    except Exception:
        import traceback
        tb = traceback.format_exc()
        print(tb)
        raise HTTPException(status_code=500, detail=tb)