"""
Följdfrågeupplösning.

Skriver om en följdfråga till en fristående fråga med hjälp av
samtalshistoriken. Använder ett LLM-anrop via Ollama.
"""

from app.llm import LocalLLM
from app.session_state import ConversationState


def _summarize_recent_turns(turns: list[dict], max_turns: int = 4) -> str:
    """Sammanfatta de senaste turerna till en kort kontextbeskrivning."""
    recent = turns[-max_turns:]
    parts = []
    for turn in recent:
        role = "Användare" if turn["role"] == "user" else "Assistent"
        content = turn["content"]
        # Korta ned långa svar
        if len(content) > 200:
            content = content[:200] + "..."
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


def rewrite_followup(
    question: str,
    state: ConversationState,
    llm: LocalLLM,
) -> tuple[str, bool]:
    """
    Om frågan verkar vara en följdfråga, skriv om den till en fristående
    fråga. Returnerar (omskriven fråga, om den skrevs om).

    Om det inte finns någon historik returneras frågan oförändrad.
    """
    if not state.has_history:
        return question, False

    context_summary = _summarize_recent_turns(state.turns)

    doc_titles = ", ".join(state.active_doc_paths[-3:]) if state.active_doc_paths else "inga"
    snippet_text = " | ".join(state.active_answer_snippets[-3:]) if state.active_answer_snippets else "inget"

    prompt = f"""Du hjälper till att skriva om en följdfråga till en fristående fråga
som kan sökas utan konversationshistorik. Om frågan redan är fristående,
returnera den oförändrad.

Samtalshistorik:
{context_summary}

Aktiva dokument: {doc_titles}
Senaste svaret innehöll: {snippet_text}

Följdfråga: {question}

Fristående fråga:"""

    rewritten = llm.generate(prompt).strip()

    # Enkel sanitetskontroll — om LLM:en returnerar något orimligt,
    # använd originalet
    if not rewritten or len(rewritten) < 3 or len(rewritten) > 500:
        return question, False

    return rewritten, True