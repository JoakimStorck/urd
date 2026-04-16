"""
Hantering av sociala och meta-yttringar.

För yttringar som klassificerats som social_or_meta körs inte retrieval
eller syntes. Istället anropas LLM:en direkt med en prompt som ber om
ett kort, artigt svar — utan att producera faktapåståenden om
dokumentinnehåll.

Denna linje är viktig: systemets kärnegenskap är att faktapåståenden om
dokument ska vara källbaserade. En social hanterare som börjar svara
generativt om dokumentinnehåll skulle bryta den egenskapen.
Prompten är därför explicit om vad som får och inte får sägas.
"""

from __future__ import annotations

import logging

from app.config import settings
from app.llm import LocalLLM
from app.session_state import ConversationState

logger = logging.getLogger(__name__)


SOCIAL_PROMPT_TEMPLATE = """Du är en lokal dokumentassistent för interna styrdokument.
Användaren har just sagt något som inte kräver uppslag i dokumenten —
det kan vara en social kvittering, en meta-fråga om dig, eller en
reflektion över samtalet.

Svara kort och artigt. Viktiga regler:

- Gör INGA nya faktapåståenden om dokumentens innehåll. Du vet inget
  om dokumenten utöver det som redan har sagts i samtalshistoriken.
- Du får sammanfatta eller referera till det som redan sagts i samtalet.
  Om du gör det, var trogen mot vad som faktiskt sades.
- Du får svara på meta-frågor om dig själv (att du är en lokal
  dokumentassistent, att du bara svarar utifrån källor, etc.).
- Om yttringen är en ren kvittering ("tack", "okej", "bra svar"),
  svara kort och inbjudande till nästa fråga.
- Om du är osäker på vad användaren vill, säg det kort och be om
  en förtydligande fråga.
- Håll svaret till 1-3 meningar om möjligt.

{history_block}Användarens yttring:
{utterance}

Svar:"""


def _format_history(turns: list[dict], max_turns: int) -> str:
    """Formatera de senaste turerna för social-hanterarens prompt."""
    if not turns or max_turns <= 0:
        return ""

    entries = turns[-(max_turns * 2):]
    if not entries:
        return ""

    lines = []
    for entry in entries:
        role = "Användare" if entry["role"] == "user" else "Assistent"
        content = entry["content"]
        if len(content) > 400:
            content = content[:400] + "..."
        lines.append(f"{role}: {content}")

    return "Senaste samtalshistorik:\n" + "\n".join(lines) + "\n\n"


def handle_social(
    question: str,
    state: ConversationState,
    llm: LocalLLM,
) -> str:
    """
    Generera ett kort socialt/meta-svar utan retrieval.

    Använder settings.social_history_turns för att avgöra hur mycket
    historik som skickas med.
    """
    history_block = _format_history(
        state.turns,
        settings.social_history_turns,
    )

    prompt = SOCIAL_PROMPT_TEMPLATE.format(
        history_block=history_block,
        utterance=question,
    )

    try:
        answer = llm.generate(prompt).strip()
    except Exception as e:
        logger.warning("Social-hanterare misslyckades: %s", e)
        return (
            "Ursäkta, jag kunde inte generera ett svar just nu. "
            "Försök igen eller ställ en fråga om dokumenten."
        )

    if not answer:
        return "Säg till om det är något mer du vill titta på i dokumenten."

    return answer