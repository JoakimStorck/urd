from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.schemas import ChatRequest, ChatResponse
from app.retrieval import RagService
from app.session_state import SessionStore
from app.intent import classify_utterance, Classification
from app.social import handle_social
from app.qud_drift import measure_drift
from app.followup import rewrite_followup

app = FastAPI(title="Local IIT URD")
rag = RagService()
sessions = SessionStore()

static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Resolve docs root en gång vid uppstart
_docs_root = settings.docs_path.resolve()


@app.get("/")
def index():
    return FileResponse(static_dir / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


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
            )
            if drift is not None and drift.drift_detected:
                classification = Classification(
                    intent="new_main_question",
                    substyle=None,
                    reason=(
                        f"qud_drift_detected (similarity={drift.similarity} "
                        f"< threshold={drift.threshold})"
                    ),
                    raw=classification.raw,
                    used_fallback=False,
                )

        # Grund-debug som alla vägar lägger till
        base_debug = {
            "session_id": state.session_id,
            "classification": {
                "intent": classification.intent,
                "substyle": classification.substyle,
                "reason": classification.reason,
                "used_fallback": classification.used_fallback,
            },
            "qud": {
                "text": state.current_qud_text,
                "age_turns": state.qud_age_turns,
            },
        }

        if drift is not None:
            base_debug["qud_drift"] = {
                "similarity": drift.similarity,
                "threshold": drift.threshold,
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
            )

            # Rework-tur: ersätt INTE active_hits — samma material bär
            # fortfarande tråden. Bara last_answer och snippets uppdateras.
            state.add_rework_turn(req.question, response.answer)

            if response.debug is None:
                response.debug = {}
            response.debug.update(base_debug)
            response.debug["path"] = classification.intent

            response.session_id = state.session_id
            return response

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
            # fristående retrievalfråga och ankra retrieval lokalt
            # i de dokument som bar föregående svar.
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
        )

        # Uppdatera sessionsstate med dokumentkällorna OCH de faktiska
        # hits som bar svaret — så att nästa elaboration/verification
        # kan återanvända dem.
        doc_paths = list({
            hit.metadata.source_path
            for hit in response.sources
        })
        state.add_turn(
            req.question,
            response.answer,
            doc_paths,
            hits=response.sources,
        )

        # Merga debug-info från retrieval/syntes med vår dispatch-info
        if response.debug is None:
            response.debug = {}
        response.debug.update(base_debug)
        response.debug["path"] = path_label
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