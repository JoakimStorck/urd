"""
Samtalsminne för URD.

Håller ett litet tillståndsobjekt per session. State:t är medvetet
smalt: det minns vilka källor och vilka svarsstycken som bar senaste
svaret, plus en aktiv huvudfråga (QUD — Question Under Discussion)
som pågår i samtalet.

QUD-modellen är inspirerad av Roberts QUD-teori (1996/2012), men
implementerad i en förenklad form: en enda aktiv fråga (inte en
stack), och alltid den ordagranna originaltexten från den yttring
som klassificerades som new_main_question. Ingen omskrivning eller
destillation — den spårbara råtexten är hela poängen.

Turfönstret (hur många turer som behålls) skalar med de tre
context-parametrarna i config. Eftersom varje tur = fråga-svar-par
= 2 entries, är det faktiska taket 2 * max(history-parametrarna).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from app.config import settings


def _max_turns_window() -> int:
    """
    Räkna ut hur många entries (fråga+svar = 2 entries per tur) som
    behövs i turhistoriken för att alla context-steg ska få vad de
    begär, med ett minimum på 6 entries (3 turer).
    """
    max_context_turns = max(
        settings.qud_background_turns,
        settings.social_history_turns,
        settings.classification_history_turns,
        3,  # minsta rimliga fönster även om alla config-värden är 0
    )
    return max_context_turns * 2


@dataclass
class ConversationState:
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    turns: list[dict] = field(default_factory=list)
    active_doc_paths: list[str] = field(default_factory=list)
    active_answer_snippets: list[str] = field(default_factory=list)

    # QUD — den aktiva huvudfrågan i samtalet. Ordagrann originaltext
    # från den yttring som senast klassificerades som new_main_question.
    # None om ingen huvudfråga ännu etablerats.
    current_qud_text: str | None = None
    # Vid vilken tur-position (index i turns) QUD:n sattes. Används för
    # att räkna "ålder" på QUD:n, synlig i debug.
    current_qud_turn_index: int | None = None

    def set_qud(self, text: str) -> None:
        """
        Sätt den aktiva QUD:n till ordagrann originaltext.

        Anropas när en yttring klassificerats som new_main_question,
        innan turen läggs till. Positionen registreras som nuvarande
        turns-längd, vilket är index där användarfrågan kommer att
        hamna när add_turn anropas direkt efter.
        """
        self.current_qud_text = text
        self.current_qud_turn_index = len(self.turns)

    @property
    def qud_age_turns(self) -> int | None:
        """
        Antal turer som passerat sedan QUD:n sattes.
        Returnerar None om ingen QUD är aktiv.

        En "tur" här = ett fråga-svar-par. Beräknas från turns-listan
        där varje tur är 2 entries.
        """
        if self.current_qud_turn_index is None:
            return None
        entries_since = len(self.turns) - self.current_qud_turn_index
        return max(0, entries_since // 2)

    def add_turn(
        self,
        question: str,
        answer: str,
        doc_paths: list[str],
    ) -> None:
        self.turns.append({"role": "user", "content": question})
        self.turns.append({"role": "assistant", "content": answer})

        self._trim_turns()

        self.active_doc_paths = doc_paths

        # Extrahera korta utdrag från svaret (första meningen per stycke)
        self.active_answer_snippets = _extract_snippets(answer, max_snippets=3)

    def add_social_turn(
        self,
        question: str,
        answer: str,
    ) -> None:
        """
        Registrera en social eller meta-tur i samtalet.

        Till skillnad från add_turn uppdateras INTE active_doc_paths eller
        active_answer_snippets. Dessa ska spegla vad som senast bars av
        källor, så att en följdfråga efter en social tur anknyter till
        den senaste dokumentturen, inte till det sociala svaret.
        """
        self.turns.append({"role": "user", "content": question})
        self.turns.append({"role": "assistant", "content": answer})

        self._trim_turns()

    def _trim_turns(self) -> None:
        window = _max_turns_window()
        if len(self.turns) > window:
            removed = len(self.turns) - window
            self.turns = self.turns[-window:]
            # Justera QUD-index så det pekar på rätt position i det
            # trimmade fönstret. Om QUD:n satt utanför fönstret
            # (dvs. index blir negativt) nollställs den inte — vi
            # behåller texten men markerar index som 0, eftersom
            # "äldre än fönstret" ändå återspeglas korrekt i qud_age_turns
            # via förhållandet mellan tur-längden och index.
            if self.current_qud_turn_index is not None:
                self.current_qud_turn_index = max(
                    0,
                    self.current_qud_turn_index - removed,
                )

    @property
    def has_history(self) -> bool:
        return len(self.turns) > 0


def _extract_snippets(answer: str, max_snippets: int = 3) -> list[str]:
    """Extrahera korta utdrag — första meningen per stycke."""
    snippets = []
    for paragraph in answer.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        # Ta första meningen (upp till första punkten)
        first_sentence = paragraph.split(". ")[0]
        if len(first_sentence) > 150:
            first_sentence = first_sentence[:150] + "..."
        snippets.append(first_sentence)
        if len(snippets) >= max_snippets:
            break
    return snippets


class SessionStore:
    """Enkel in-memory sessionshantering."""

    def __init__(self) -> None:
        self._sessions: dict[str, ConversationState] = {}

    def get_or_create(self, session_id: str | None) -> ConversationState:
        if session_id and session_id in self._sessions:
            return self._sessions[session_id]
        state = ConversationState()
        self._sessions[state.session_id] = state
        return state

    def get(self, session_id: str) -> ConversationState | None:
        return self._sessions.get(session_id)