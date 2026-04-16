from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.schemas import ChatRequest, ChatResponse
from app.retrieval import RagService
from app.session_state import SessionStore
from app.intent import classify_utterance
from app.social import handle_social
from app.llm import LLMUnavailableError

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

        # 1. Klassificera yttringen för att avgöra vilken väg som tas.
        classification = classify_utterance(req.question, state, rag.llm)

        # Grund-debug som alla vägar lägger till
        base_debug = {
            "session_id": state.session_id,
            "classification": {
                "intent": classification.intent,
                "reason": classification.reason,
                "used_fallback": classification.used_fallback,
            },
        }

        # 2. Dispatcha baserat på intent.
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

        # Från och med här är det document_question eller followup —
        # båda går via retrieval + syntes, men followup får bakgrund.
        if classification.intent == "followup":
            background_turns = list(state.turns)
            background_max_turns = settings.followup_background_turns
            path_label = "followup"
        else:
            background_turns = None
            background_max_turns = 0
            path_label = "document_question"

        response = rag.answer(
            req.question,
            background_turns=background_turns,
            background_max_turns=background_max_turns,
        )

        # Uppdatera sessionsstate med dokumentkällorna
        doc_paths = list({
            hit.metadata.source_path
            for hit in response.sources
        })
        state.add_turn(req.question, response.answer, doc_paths)

        # Merga debug-info från retrieval/syntes med vår dispatch-info
        if response.debug is None:
            response.debug = {}
        response.debug.update(base_debug)
        response.debug["path"] = path_label
        if path_label == "followup":
            response.debug["background_max_turns"] = background_max_turns

        response.session_id = state.session_id

        return response
    except Exception:
        import traceback
        tb = traceback.format_exc()
        print(tb)
        raise HTTPException(status_code=500, detail=tb)