from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.schemas import ChatRequest, ChatResponse
from app.retrieval import RagService
from app.session_state import SessionStore
from app.followup import rewrite_followup
from app.llm import LLMUnavailableError

app = FastAPI(title="Local IIT DocChat")
rag = RagService()
sessions = SessionStore()

static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def index():
    return FileResponse(static_dir / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


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