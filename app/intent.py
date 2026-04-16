"""
Klassificering av användarens yttring inom en QUD-inspirerad modell.

Yttringen klassificeras i en av fyra huvudkategorier som beskriver
dess RELATION till samtalets aktiva huvudfråga (QUD — Question Under
Discussion), inte till yttringens innehåll i sig:

- new_main_question: öppnar en ny huvudtråd. Sätter en ny QUD.
- related_to_qud: hör till aktiv QUD. Subtyp anger stil:
    * subquestion: går djupare i en del av QUD
    * broadening: vidgar QUD, öppnar närliggande områden
    * narrowing_or_repair: preciserar efter att föregående svar
      varit för grovt eller missförstått
- verification_or_challenge: prövar eller ifrågasätter föregående svar.
  Refererar till ett svar, inte till QUD:n.
- social_or_meta: social markör, meta-fråga, reflektion. Ingen retrieval.

Modellen är inspirerad av Roberts QUD-teori (1996/2012) med förenklingar
för lokal drift och spårbarhet. Den formella QUD-stacken ersätts av en
enda aktiv huvudfråga, och subtyperna är informella stilmarkörer snarare
än teoretiska primitiver.

Felläget är asymmetriskt:
- Dokumentfråga klassad som social_or_meta → tunt svar utan källor
  (bryter systemets kärnegenskap).
- Social yttring klassad som dokumentfråga → överarbetat svar
  (irriterande men inte farligt).

Prompten lutar därför mot dokumentvägarna i tveksamma fall, och vid
parsningsfel fallbackar vi till new_main_question (den mest
konservativa klass som garanterar ett källbaserat svar).
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


Intent = Literal[
    "new_main_question",
    "related_to_qud",
    "verification_or_challenge",
    "social_or_meta",
]

# Substyle är endast giltig när intent == "related_to_qud".
# Den styr syntesstilen, inte retrievalbeteendet.
Substyle = Literal["subquestion", "broadening", "narrowing_or_repair"]


@dataclass
class Classification:
    intent: Intent
    substyle: Substyle | None = None  # bara meningsfull för related_to_qud
    reason: str | None = None
    raw: str | None = None
    used_fallback: bool = False


CLASSIFY_PROMPT_TEMPLATE = """Du klassificerar användarens yttring i förhållande till
samtalets aktiva huvudfråga ("QUD" — Question Under Discussion).

Din uppgift är att bestämma vilken RELATION yttringen har till den
aktiva huvudfrågan, inte vad yttringen handlar om i sig.

HUVUDKATEGORIER:

new_main_question — Yttringen öppnar en NY huvudtråd. Den kan stå
på egna ben utan samtalshistorik och är inte en fortsättning på den
aktiva huvudfrågan.
Exempel: "Vilka regler gäller vid disputation?", "Vad är en adjungerad professor?"
Även en fråga som byter ämne helt (t.ex. efter doktorander →
"Vilka regler gäller för tjänstledighet?") är en new_main_question.

related_to_qud — Yttringen hör till den aktiva huvudfrågan. Användaren
går djupare, vidgar, eller preciserar inom samma tråd. Ange då också
en "substyle" som beskriver hur:

  * subquestion — går djupare i en del av huvudfrågan.
    Exempel: "Och för postdoktorer då?", "Vem beslutar?",
    "Hur länge kan en sådan anställning vara?"

  * broadening — vidgar huvudfrågan, efterfrågar mer eller relaterat.
    Exempel: "Finns det något mer jag borde känna till?",
    "Vad mer gäller i det här sammanhanget?", "Vilka andra regler finns?"

  * narrowing_or_repair — preciserar, förtydligar, eller rättar efter
    att föregående svar varit för grovt, otydligt, eller missförstått.
    Exempel: "Nej, jag menade anställda.", "Men just för nyanställda?",
    "Det där var för allmänt — jag vill veta om lönesättning."

verification_or_challenge — Yttringen PRÖVAR eller IFRÅGASÄTTER
föregående svar. Den refererar till vad assistenten nyss sade, inte
till huvudfrågan.
Exempel: "Stämmer det verkligen?", "Är du säker?",
"Men jag trodde X — har du källa på det?", "Det låter fel."

social_or_meta — Social markör, tacksägelse, meta-fråga om systemet,
eller reflektion. Kräver ingen retrieval.
Exempel: "Tack, bra svar.", "Hur fungerar du?", "Okej.",
"Kan du sammanfatta det vi pratat om?"

VIKTIGA REGLER:
- Tveksamma fall mellan new_main_question och related_to_qud:
  välj related_to_qud om yttringen är svår att tolka utan den
  aktiva huvudfrågan, annars new_main_question.
- Tveksamma fall där yttringen kan tolkas som dokumentfråga:
  välj någon av dokumentkategorierna (inte social_or_meta).
- Om det inte finns någon aktiv huvudfråga kan yttringen inte vara
  related_to_qud — välj då new_main_question eller en av de andra.
- "substyle" anges ENDAST för related_to_qud. För övriga klasser ska
  substyle vara null.

{qud_block}{history_block}Aktuell yttring:
{utterance}

Svara ENBART med JSON, utan förklaringar eller markdown:
{{"intent": "...", "substyle": "..."|null, "reason": "kort motivering"}}"""


def _format_qud_block(qud_text: str | None) -> str:
    """Formatera den aktiva QUD:n för klassificerarens prompt."""
    if not qud_text:
        return "Aktiv huvudfråga: (ingen etablerad ännu)\n\n"
    return f'Aktiv huvudfråga: "{qud_text}"\n\n'


def _format_history(turns: list[dict], max_turns: int) -> str:
    """Formatera de senaste turerna för klassificerarens prompt."""
    if not turns or max_turns <= 0:
        return ""

    entries = turns[-(max_turns * 2):]
    if not entries:
        return ""

    lines = []
    for entry in entries:
        role = "Användare" if entry["role"] == "user" else "Assistent"
        content = entry["content"]
        if len(content) > 300:
            content = content[:300] + "..."
        lines.append(f"{role}: {content}")

    return "Senaste samtalshistorik:\n" + "\n".join(lines) + "\n\n"


_VALID_INTENTS = {
    "new_main_question",
    "related_to_qud",
    "verification_or_challenge",
    "social_or_meta",
}
_VALID_SUBSTYLES = {"subquestion", "broadening", "narrowing_or_repair"}


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
    if intent_raw not in _VALID_INTENTS:
        logger.warning("Ogiltigt intent-värde: %r", intent_raw)
        return None

    substyle_raw = data.get("substyle")
    substyle: Substyle | None = None
    if substyle_raw is not None and str(substyle_raw).strip().lower() not in ("", "null", "none"):
        candidate = str(substyle_raw).strip()
        if candidate in _VALID_SUBSTYLES:
            substyle = candidate  # type: ignore[assignment]
        else:
            logger.info("Ogiltig substyle %r — ignoreras.", candidate)

    # Substyle är bara meningsfull för related_to_qud. Nolla den
    # tyst för övriga klasser så att LLM-fel inte propagerar.
    if intent_raw != "related_to_qud":
        substyle = None

    reason = data.get("reason")
    if reason is not None:
        reason = str(reason).strip() or None

    return Classification(
        intent=intent_raw,  # type: ignore[arg-type]
        substyle=substyle,
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
    Klassificera yttringen i QUD-modellens fyra kategorier.

    Vid fel (parsningsfel, LLM-fel) returneras new_main_question som
    konservativ fallback. Detta garanterar att användaren får ett
    källbaserat svar via den vanliga retrieval-vägen, och att ingen
    tidigare QUD tyst fortsätter påverka retrieval när klassificeringen
    gått fel.

    Om det inte finns någon aktiv QUD eller historik kan yttringen
    inte vara related_to_qud eller verification_or_challenge i
    meningsfull mening; vi tvingar i så fall ner dem till
    new_main_question.
    """
    qud_block = _format_qud_block(state.current_qud_text)
    history_block = _format_history(
        state.turns,
        settings.classification_history_turns,
    )

    prompt = CLASSIFY_PROMPT_TEMPLATE.format(
        qud_block=qud_block,
        history_block=history_block,
        utterance=question,
    )

    try:
        raw = llm.generate(prompt)
    except Exception as e:
        logger.warning(
            "Klassificering misslyckades (LLM-fel): %s. Fallback till new_main_question.",
            e,
        )
        return Classification(
            intent="new_main_question",
            reason=f"llm_error: {type(e).__name__}",
            used_fallback=True,
        )

    result = _parse_classification_json(raw)
    if result is None:
        return Classification(
            intent="new_main_question",
            reason="parse_failed",
            raw=raw[:500] if raw else None,
            used_fallback=True,
        )

    # related_to_qud och verification_or_challenge kräver en aktiv QUD
    # eller åtminstone historik att referera till. Utan detta är
    # klassen semantiskt tom — tolka om som new_main_question.
    if result.intent in ("related_to_qud", "verification_or_challenge"):
        if state.current_qud_text is None and not state.has_history:
            logger.info(
                "Klassificerades som %s utan QUD eller historik — "
                "tolkar om som new_main_question.",
                result.intent,
            )
            return Classification(
                intent="new_main_question",
                reason=f"{result.intent}_without_context",
                raw=result.raw,
                used_fallback=False,
            )

    return result