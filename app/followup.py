"""
Följdfrågeupplösning.

Skriver om en följdfråga till en fristående fråga med hjälp av
samtalskontexten. Använder ett LLM-anrop via Ollama.

Två hårda lärdomar från mätningarna styr designen (körning
20260812_112439):

1. **Omskrivningskontexten innehåller ALDRIG assistentens svarstext.**
   En omskrivning som byggdes på tur 2-svaret ärvde nonsensordet
   'beslutskonstanta' (en Nemo-hallucination) och metareferensen
   '[Källa 1]' — cross-encodern dömde sedan varje chunk mot
   rappakaljan och turen abstainade. Frågans referenter finns i
   användarens egna ord, huvudfrågan och de aktiva dokumentens namn;
   svarstexten är den enda plats hallucinerade ord kan komma ifrån.

2. **Resultatet saneras och valideras deterministiskt.** Kända
   prefix och källreferenser strippas, och omskrivningar som
   innehåller innehållsord som varken förekommer i originalfrågan,
   QUD:n, användarens tidigare frågor eller dokumenttitlarna
   förkastas (retrieval kör då vidare med originalfrågan och
   QUD-ankare). Förkastanden loggas med både rå och sanerad text
   så att rewrite-kvaliteten är granskbar i serverloggen.
"""

from __future__ import annotations

import logging
import re

from app.llm import LocalLLM
from app.session_state import ConversationState

logger = logging.getLogger(__name__)

_CITATION_RE = re.compile(r"\[?\s*[Kk]älla\s*\d+\s*\]?")
_KNOWN_PREFIXES = (
    "fristående fråga:",
    "omskriven fråga:",
    "fråga:",
    "svar:",
)

# Innehållsord = tokens med minst så här många tecken. Kortare ord
# (vad, hur, för, inför ...) är grammatik och valideras inte.
_CONTENT_WORD_MIN_LEN = 5

# Böjningstolerans i vokabulärkontrollen: ett omskrivningsord räknas
# som känt om det delar prefix med ett känt ord och längdskillnaden
# är högst så här stor ("forskningsansökan"/"forskningsansökningar"
# passerar; "beslut"/"beslutskonstanta" gör det inte).
_MAX_INFLECTION_DIFF = 4


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)


def _sanitize_rewrite(text: str) -> str:
    """Strippa källreferenser, kända prefix och citattecken."""
    cleaned = _CITATION_RE.sub("", text).strip()
    lowered = cleaned.casefold()
    for prefix in _KNOWN_PREFIXES:
        if lowered.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
            lowered = cleaned.casefold()
    cleaned = cleaned.strip('"“”’ ')
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned


def _is_known_word(word: str, vocabulary: set[str]) -> bool:
    if word in vocabulary:
        return True
    for known in vocabulary:
        if len(known) < _CONTENT_WORD_MIN_LEN:
            continue
        shorter, longer = sorted((word, known), key=len)
        if longer.startswith(shorter) and len(longer) - len(shorter) <= _MAX_INFLECTION_DIFF:
            return True
    return False


def _novel_content_words(rewritten: str, vocabulary: set[str]) -> list[str]:
    """Innehållsord i omskrivningen som saknas i den tillåtna vokabulären."""
    return [
        tok for tok in _tokenize(rewritten)
        if len(tok) >= _CONTENT_WORD_MIN_LEN and not _is_known_word(tok, vocabulary)
    ]


def _user_questions(turns: list[dict], max_questions: int = 3) -> list[str]:
    """De senaste användarturerna — aldrig assistentens svar."""
    questions = [t["content"] for t in turns if t.get("role") == "user"]
    return questions[-max_questions:]


def _doc_display_names(doc_paths: list[str]) -> list[str]:
    """Filnamn utan katalogväg — läsbara dokumentnamn för prompten."""
    names = []
    for path in doc_paths:
        name = path.rsplit("/", 1)[-1]
        names.append(name)
    return names


def rewrite_followup(
    question: str,
    state: ConversationState,
    llm: LocalLLM,
) -> tuple[str, bool]:
    """
    Skriv om en följdfråga till en fristående fråga.

    Returnerar (fråga, om den skrevs om). Vid saknad historik,
    misslyckad sanering eller förkastad validering returneras
    originalfrågan oförändrad — anroparen kör då QUD-ankarvägen.
    """
    if not state.has_history:
        return question, False

    user_questions = _user_questions(state.turns)
    doc_names = _doc_display_names(state.active_doc_paths[-3:])

    qud_line = f"Huvudfråga i samtalet: {state.current_qud_text}\n" if state.current_qud_text else ""
    history_lines = "\n".join(f"- {q}" for q in user_questions) or "-"
    docs_line = ", ".join(doc_names) if doc_names else "inga"

    prompt = f"""Du skriver om en följdfråga till en fristående fråga som kan
sökas utan konversationshistorik. Om frågan redan är fristående,
returnera den oförändrad. Använd ENDAST ord ur följdfrågan,
huvudfrågan och dokumentnamnen — inga nya begrepp.

{qud_line}Användarens tidigare frågor:
{history_lines}

Aktiva dokument: {docs_line}

Följdfråga: {question}

Fristående fråga:"""

    raw = llm.generate(prompt).strip()
    rewritten = _sanitize_rewrite(raw)

    # Grundläggande rimlighet
    if not rewritten or len(rewritten) < 3 or len(rewritten) > 500:
        logger.info("Rewrite förkastad (längd): %r", raw[:200])
        return question, False

    # Vokabulärkontroll: omskrivningen får inte införa innehållsord
    # som saknas i användarens fråga, QUD:n, tidigare användarfrågor
    # eller dokumentnamnen. Det är filtret som stoppar hallucinerade
    # ord ('beslutskonstanta') från att styra retrieval.
    vocabulary = set(_tokenize(question))
    if state.current_qud_text:
        vocabulary.update(_tokenize(state.current_qud_text))
    for q in user_questions:
        vocabulary.update(_tokenize(q))
    for name in doc_names:
        vocabulary.update(_tokenize(name))

    novel = _novel_content_words(rewritten, vocabulary)
    if novel:
        logger.info(
            "Rewrite förkastad (okända innehållsord %s): rå=%r sanerad=%r",
            novel, raw[:200], rewritten[:200],
        )
        return question, False

    if rewritten.casefold() == question.strip().casefold():
        return question, False

    return rewritten, True
