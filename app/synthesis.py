"""
Huvudsyntesen: enstegsformulering direkt från källorna.

Den här modulen innehåller systemets huvudväg för svarsgenerering:
en enda LLM-generering från en prompt som både innehåller källorna
och instruktionerna för hur svaret ska formuleras. Prompten är
detaljbevarande — den kräver att listor återges i sin helhet,
tabeller inte komprimeras, och att källans exakta termer används
för formella moment.

Tidigare har en tvåstegsarkitektur funnits här: evidensextraktion
först (parafraserande JSON) och svarsformulering sedan. Den visade
sig komprimera bort konkreta detaljer och har ersatts av denna
direktformulering.

Rework-vägarna (elaboration, verification) bor i rework.py — de
har egna arkitekturer som motsvarar uppgifternas natur. Returtypen
SynthesisResult delas mellan huvudväg och rework-vägar och bor i
synthesis_types.py.
"""

from __future__ import annotations

import time

from app.llm import LocalLLM
from app.schemas import SourceHit
from app.synthesis_types import SynthesisResult


# ---------------------------------------------------------------------------
# Källformatering
# ---------------------------------------------------------------------------

def _format_sources_for_direct(hits: list[SourceHit]) -> str:
    """Formatera källor för huvudprompten."""
    blocks = []
    for i, hit in enumerate(hits, start=1):
        meta = hit.metadata
        header = f"[Källa {i}] {meta.file_name} — {meta.section_title}"
        blocks.append(f"{header}\n{hit.text}")
    return "\n\n".join(blocks)


def _format_background(turns: list[dict], max_turns: int) -> str:
    """
    Formatera de senaste turerna som bakgrundstext.

    Varje "tur" i config-bemärkelse är ett fråga-svar-par (2 entries
    i turns-listan). Returnerar tom sträng om ingen historik eller
    om max_turns <= 0.
    """
    if not turns or max_turns <= 0:
        return ""

    entries = turns[-(max_turns * 2):]
    if not entries:
        return ""

    lines = []
    for entry in entries:
        role = "Användare" if entry["role"] == "user" else "Assistent"
        content = entry["content"]
        if len(content) > 600:
            content = content[:600] + "..."
        lines.append(f"{role}: {content}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompter
# ---------------------------------------------------------------------------

BACKGROUND_BLOCK_TEMPLATE = """
SAMTALSBAKGRUND (endast som kontext för att förstå frågan):
{background_text}

VIKTIGT om samtalsbakgrunden:
- Den är INTE en källa. Påståenden i svaret får ENDAST bygga på
  källmaterialet nedan, aldrig på samtalsbakgrunden.
- Den hjälper dig tolka vad frågan syftar på (t.ex. vad "andra regler"
  eller "det" refererar till), men den är inte faktamaterial.
"""


DIRECT_SYNTHESIS_PROMPT = """Du är en lokal dokumentassistent för interna styrdokument.
Svara på frågan enbart utifrån källorna nedan.

KRITISKT FÖR KORREKTHET:

1. Använd de delar av källorna som DIREKT eller TYDLIGT besvarar frågan.
   - Om någon källa direkt besvarar frågan, använd den.
   - Om andra källor är mindre relevanta, indirekta eller handlar om en
     annan aspekt, ignorera dem hellre än att låta dem förstöra svaret.
   - Säg bara att källorna inte räcker om ingen källa ger ett direkt
     eller tydligt relevant stöd för svaret.

2. Förväxla inte olika typer av innehåll.
   - Arbetsuppgifter är inte samma sak som bedömningsgrunder.
   - Behörighetskrav är inte samma sak som ansvar eller uppdrag.
   - Personliga egenskaper är inte samma sak som procedursteg.
   - Men: om frågan gäller krav, kvalifikationer eller behörighet, använd
     avsnitt om behörighet och uttryckliga krav även om andra källor i
     materialet handlar om något annat.

3. Dra inga egna slutsatser från semantiskt närliggande text.
   - Skriv inte vad som "rimligen borde gälla".
   - Generalisera inte.
   - Om något bara kan antydas men inte sägs tydligt: avstå från att
     säga det.
   - Om en del av svaret stöds tydligt men andra delar inte gör det:
     ge den stödda delen och säg att resten inte framgår tydligt.

4. Abstain bara när det verkligen behövs.
   - Att vissa källor är indirekta är inte skäl nog att avstå.
   - Avstå bara om ingen källa innehåller information som tydligt kan
     användas för att besvara frågan.

KRITISKT FÖR SVARETS ANVÄNDBARHET:

- BEVARA ALLA KONKRETA DETALJER från de relevanta källorna: belopp,
  gränsvärden, roller, tidsfrister, procedurer, villkor och undantag.

- OM EN RELEVANT KÄLLA INNEHÅLLER EN NUMRERAD LISTA som direkt svarar på
  frågan: återge ALLA poster i listan, i samma ordning.

- OM EN RELEVANT KÄLLA INNEHÅLLER EN TABELL eller strukturerad
  uppställning som direkt svarar på frågan: återge de relevanta delarna
  tydligt. Återge hela tabellen om hela tabellen behövs för att besvara
  frågan.

- REDOGÖR för innehållet i de relevanta källorna, inget mer.
  UTVECKLA INTE svaret utöver vad källorna stöder.

- ANVÄND KÄLLORNAS EXAKTA TERMER för formella moment, roller och
  procedurer.

SVARSREGLER:

- Inled direkt med svaret.
- Om ingen källa räcker, säg det i första meningen.
- Om svaret bara delvis framgår, säg först det som faktiskt framgår och
  markera sedan kort vad som inte framgår.
- Besvara bara det som faktiskt efterfrågades.
- Ange källa efter varje påstående med [Källa N].
- Om flera relevanta källor säger olika saker eller pekar åt olika håll,
  redovisa detta uttryckligen.

{background_block}Källor:
{sources_block}

Fråga: {question}

Svar:"""


# ---------------------------------------------------------------------------
# Syntes
# ---------------------------------------------------------------------------

def synthesize(
    question: str,
    hits: list[SourceHit],
    llm: LocalLLM,
    background_turns: list[dict] | None = None,
    background_max_turns: int = 0,
) -> SynthesisResult:
    """
    Enstegssyntes med detaljbevarande prompt.

    background_turns och background_max_turns används för att ge
    modellen samtalskontext (t.ex. för related_to_qud där en kort
    följdfråga ska tolkas mot tidigare turer). Bakgrunden är inte
    en källa för påståenden — den är bara en tolkningsnyckel.
    """
    sources_block = _format_sources_for_direct(hits)

    background_block = ""
    if background_turns and background_max_turns > 0:
        background_text = _format_background(background_turns, background_max_turns)
        if background_text:
            background_block = BACKGROUND_BLOCK_TEMPLATE.format(
                background_text=background_text,
            )

    prompt = DIRECT_SYNTHESIS_PROMPT.format(
        background_block=background_block,
        sources_block=sources_block,
        question=question,
    )

    t0 = time.perf_counter()
    answer = llm.generate(prompt)
    t1 = time.perf_counter()

    return SynthesisResult(
        answer=answer,
        verification=None,
        used_fallback=False,
        timing_s={
            "direct_synthesis": round(t1 - t0, 3),
        },
    )