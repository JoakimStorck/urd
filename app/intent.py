"""
Klassificering av användarens yttring.

Avgör om yttringen är en fristående dokumentfråga, en följdfråga
som bygger på tidigare turer, eller en social/meta-yttring som
inte kräver uppslag i dokumenten.

Felläget är asymmetriskt:
- Dokumentfråga som klassas som social → användaren får ett tunt
  svar utan källor (bryter systemets kärnegenskap).
- Social yttring som klassas som dokumentfråga → användaren får ett
  överarbetat svar (irriterande men inte farligt).

Prompten lutar därför mot document_question i tveksamma fall, och
vid parsningsfel fallbackar vi till document_question.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Literal

from app.config import settings
from app.llm import LocalLLM
from app.session_state import ConversationState

logger = logging.getLogger(__name__)


Intent = Literal["document_question", "followup", "social_or_meta"]


@dataclass
class Classification:
    intent: Intent
    reason: str | None = None
    raw: str | None = None
    used_fallback: bool = False


CLASSIFY_PROMPT_TEMPLATE = """Du klassificerar användarens yttring till en av tre kategorier.

KATEGORIER:

document_question — En fråga som kan besvaras genom uppslag i dokumenten
och som står på egna ben utan samtalshistorik.
Exempel: "Vilka regler gäller vid disputation?", "Vad är en adjungerad professor?"

followup — En fråga som bygger på tidigare turer i samtalet. Den
refererar implicit eller explicit till något som just sagts och är
svår att tolka utan den kontexten.
Exempel: "Vilka andra regler gäller?", "Och för postdoktorer då?",
"Berätta mer om det första steget.", "Vad är skillnaden mot det förra?"

social_or_meta — En social markör, tacksägelse, meta-fråga om systemet,
eller reflektion. Kräver inget uppslag i dokumenten.
Exempel: "Tack, bra svar.", "Hur fungerar du?", "Kan du sammanfatta det vi pratat om?",
"Okej.", "Vad duktig du är."

VIKTIGA REGLER:
- Om yttringen är en faktabaserad fråga som kräver tidigare kontext
  för att förstås → followup.
- Om yttringen är en faktabaserad fråga som står på egna ben →
  document_question, ÄVEN om samma ämne diskuterats tidigare.
- Tveksamma fall där yttringen kan tolkas som fråga om dokument →
  document_question (hellre än social_or_meta).
- En fråga om hur användaren eller systemet fungerar, eller om
  samtalet självt → social_or_meta.

{history_block}Aktuell yttring:
{utterance}

Svara ENBART med JSON, utan förklaringar eller markdown:
{{"intent": "document_question|followup|social_or_meta", "reason": "kort motivering"}}"""


def _format_history(turns: list[dict], max_turns: int) -> str:
    """Formatera de senaste turerna för klassificerarens prompt."""
    if not turns or max_turns <= 0:
        return ""

    # Varje "tur" i config-bemärkelse är ett fråga-svar-par (2 entries)
    entries = turns[-(max_turns * 2):]
    if not entries:
        return ""

    lines = []
    for entry in entries:
        role = "Användare" if entry["role"] == "user" else "Assistent"
        content = entry["content"]
        # Korta ned långa svar för att hålla prompten liten
        if len(content) > 300:
            content = content[:300] + "..."
        lines.append(f"{role}: {content}")

    return "Senaste samtalshistorik:\n" + "\n".join(lines) + "\n\n"


def _parse_classification_json(raw: str) -> Classification | None:
    """
    Parsa LLM-svaret som JSON. Hanterar markdown-fences och fritext.
    Returnerar None om parsningen misslyckas.
    """
    text = raw.strip()

    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    if not text.startswith("{"):
        brace_start = text.find("{")
        if brace_start >= 0:
            text = text[brace_start:]

    last_brace = text.rfind("}")
    if last_brace >= 0:
        text = text[: last_brace + 1]

    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Kunde inte parsa klassificerings-JSON: %s", e)
        return None

    intent_raw = str(data.get("intent", "")).strip()
    if intent_raw not in ("document_question", "followup", "social_or_meta"):
        logger.warning("Ogiltigt intent-värde: %r", intent_raw)
        return None

    reason = data.get("reason")
    if reason is not None:
        reason = str(reason).strip() or None

    return Classification(
        intent=intent_raw,  # type: ignore[arg-type]
        reason=reason,
        raw=text,
        used_fallback=False,
    )


def classify_utterance(
    question: str,
    state: ConversationState,
    llm: LocalLLM,
) -> Classification:
    """
    Klassificera yttringen till document_question, followup eller social_or_meta.

    Vid fel (parsningsfel, LLM-fel) returneras document_question som
    konservativ fallback — systemets kärnegenskap är källbaserade svar,
    och fallbackens pris är bara att vi tappar klassificeringens värde
    för just den turen.

    Om det inte finns någon historik kan yttringen inte vara en
    followup i sitt sammanhang; vi klassar utan historiken men
    behåller möjligheten att kategorisera som social_or_meta.
    """
    history_block = _format_history(
        state.turns,
        settings.classification_history_turns,
    )

    prompt = CLASSIFY_PROMPT_TEMPLATE.format(
        history_block=history_block,
        utterance=question,
    )

    try:
        raw = llm.generate(prompt)
    except Exception as e:
        logger.warning(
            "Klassificering misslyckades (LLM-fel): %s. Fallback till document_question.",
            e,
        )
        return Classification(
            intent="document_question",
            reason=f"llm_error: {type(e).__name__}",
            used_fallback=True,
        )

    result = _parse_classification_json(raw)
    if result is None:
        return Classification(
            intent="document_question",
            reason="parse_failed",
            raw=raw[:500] if raw else None,
            used_fallback=True,
        )

    # Om historik saknas helt, tvinga ner followup → document_question.
    # En "följdfråga" utan tidigare turer är semantiskt omöjlig.
    if result.intent == "followup" and not state.has_history:
        logger.info("Klassificerades som followup utan historik — tolkar som document_question.")
        return Classification(
            intent="document_question",
            reason="followup_without_history",
            raw=result.raw,
            used_fallback=False,
        )

    return result