"""
Samtalsminne för docchat.

Håller ett litet tillståndsobjekt per session. State:t är medvetet
smalt: det minns vilka källor och vilka svarsstycken som bar senaste
svaret, inte mer.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field


@dataclass
class ConversationState:
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    turns: list[dict] = field(default_factory=list)
    active_doc_paths: list[str] = field(default_factory=list)
    active_answer_snippets: list[str] = field(default_factory=list)

    def add_turn(
        self,
        question: str,
        answer: str,
        doc_paths: list[str],
    ) -> None:
        self.turns.append({"role": "user", "content": question})
        self.turns.append({"role": "assistant", "content": answer})

        # Behåll bara de senaste 6 turerna (3 fråga-svar-par)
        if len(self.turns) > 6:
            self.turns = self.turns[-6:]

        self.active_doc_paths = doc_paths

        # Extrahera korta utdrag från svaret (första meningen per stycke)
        self.active_answer_snippets = _extract_snippets(answer, max_snippets=3)

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