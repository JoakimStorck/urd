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

1. Kontrollera först om källorna DIREKT besvarar frågan.
   - Om källorna inte ger ett direkt svar, säg det uttryckligen i första meningen.
   - Svara inte mer specifikt än underlaget medger.

2. Förväxla inte olika typer av innehåll.
   - Arbetsuppgifter är inte samma sak som bedömningsgrunder.
   - Behörighetskrav är inte samma sak som ansvar eller uppdrag.
   - Personliga egenskaper är inte samma sak som procedursteg.
   - Om frågan gäller vad någon gör, använd bara källor som faktiskt
     beskriver arbetsuppgifter, ansvar, uppdrag eller befogenheter.
   - Använd inte närliggande avsnitt som substitut om kopplingen inte
     uttrycks i källan.

3. Dra inga egna slutsatser från semantiskt närliggande text.
   - Skriv inte vad som "rimligen borde gälla".
   - Generalisera inte.
   - Om något bara kan antydas men inte sägs tydligt: avstå från att
     säga det.

Att svara "källorna räcker inte" är ett fullständigt och acceptabelt
svar. Det är bättre att ge ett begränsat, ärligt svar än ett
fullständigt svar som går utöver vad källorna faktiskt säger.

Exempel på acceptabla abstain-svar:
- "Källorna handlar om X, inte om Y. Jag kan inte besvara frågan
  utifrån dessa källor."
- "Det som står i källorna är att [kort parafras]. Något utöver det
  ger källorna inte stöd för."

KRITISKT FÖR SVARETS ANVÄNDBARHET:

- BEVARA ALLA KONKRETA DETALJER från källorna: belopp, gränsvärden
  (t.ex. "500 tkr"), roller ("prefekt", "rektor", "Head of School"),
  tidsfrister, procedurer, villkor och undantag.

- OM EN KÄLLA INNEHÅLLER EN NUMRERAD LISTA: återge ALLA poster i
  listan, i samma ordning. Utelämna inga steg.

- OM EN KÄLLA INNEHÅLLER EN TABELL eller strukturerad uppställning:
  återge den i sin helhet som tabell eller strukturerad lista.
  Komprimera inte tabeller till löptext.

- REDOGÖR för innehållet i källorna, inget mer.
  UTVECKLA INTE svaret om information i källorna är knapphändig.

- ANVÄND KÄLLORNAS EXAKTA TERMER för formella moment, roller och
  procedurer.

SVARSREGLER:

- Inled direkt med det viktigaste svaret på frågan.
- Besvara bara det som faktiskt efterfrågades.
- Ange källa efter varje påstående med [Källa N].
- Om flera källor säger olika saker eller pekar åt olika håll,
  redovisa detta uttryckligen.
- Om svaret endast kan ges på en allmän nivå utifrån källorna, säg
  det tydligt och stanna där.

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