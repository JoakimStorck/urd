from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.schemas import ChatRequest, ChatResponse
from app.retrieval import RagService
from app.session_state import SessionStore
from app.followup import rewrite_followup
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

        # Skriv om följdfrågor till fristående frågor
        effective_question, was_rewritten = rewrite_followup(
            req.question, state, rag.llm
        )

        response = rag.answer(effective_question)

        # Uppdatera sessionsstate
        doc_paths = list({
            hit.metadata.source_path
            for hit in response.sources
        })
        state.add_turn(req.question, response.answer, doc_paths)

        # Lägg till session- och rewrite-info i response
        if response.debug is None:
            response.debug = {}
        response.debug["session_id"] = state.session_id
        response.debug["was_rewritten"] = was_rewritten
        if was_rewritten:
            response.debug["original_question"] = req.question
            response.debug["rewritten_question"] = effective_question

        response.session_id = state.session_id

        return response
    except Exception:
        import traceback
        tb = traceback.format_exc()
        print(tb)
        raise HTTPException(status_code=500, detail=tb)