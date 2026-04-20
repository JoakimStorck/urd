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

OPERATION_BLOCKS = {
    "direct_lookup": "",
    "relation_membership": """
EXTRA INSTRUKTION FÖR DENNA FRÅGETYP:
- Svara först med ett kort besked: ja, nej, eller framgår inte tydligt.
- Förklara sedan kort vilket stöd som finns i källorna.
- Påstå inte klassifikation om källorna bara indirekt antyder den.
""",
    "comparison": """
EXTRA INSTRUKTION FÖR DENNA FRÅGETYP:
- Jämför endast sådant som uttryckligen stöds i källorna.
- Strukturera gärna svaret under korta jämförelsedimensioner:
  arbetsuppgifter, behörighet, ansvar, roll, eller vad som inte framgår.
- Om skillnaden inte uttrycks tydligt i källorna, säg det.
""",
    "requirements": """
EXTRA INSTRUKTION FÖR DENNA FRÅGETYP:
- Lista formella krav eller behörighetskrav tydligt.
- Blanda inte ihop krav med process, allmänna egenskaper eller lämplighetsbedömningar
  om detta inte uttryckligen stöds av källorna.
""",
    "process": """
EXTRA INSTRUKTION FÖR DENNA FRÅGETYP:
- Återge processen som steg eller tydlig ordning när källorna stödjer det.
- Bevara roller, ansvar och ordningsföljd.
- Komprimera inte bort mellanled.
""",
    "aggregation": """
EXTRA INSTRUKTION FÖR DENNA FRÅGETYP:
- Sammanställ de kategorier, typer eller roller som källorna uttryckligen stödjer.
- Presentera dem som en lista.
- Om listan kan vara ofullständig utifrån källmaterialet, säg det.
""",
}

DIRECT_SYNTHESIS_PROMPT = """Du är en lokal dokumentassistent för interna styrdokument.
Svara på frågan enbart utifrån källorna nedan.

GRUNDREGLER FÖR KORREKTHET:

- Använd endast sådant som uttryckligen står i källorna, eller följer
  omedelbart och direkt av deras ordalydelse.
- Använd inte allmän kunskap, typiska fall eller rimlighetsresonemang
  för att fylla ut svaret.
- Svara inte på en mer specifik, mer generell eller mer långtgående nivå
  än källorna stöder.
- Om källorna inte direkt eller tydligt stöder ett svar på frågan, säg
  det uttryckligen.
- Om delar av frågan besvaras av källorna men andra delar inte gör det,
  besvara bara den stödda delen och markera kort vad som inte framgår.

GRUNDREGLER FÖR RELEVANS:

- Använd i första hand de källor eller källdelar som tydligast besvarar
  frågan.
- Ignorera mindre relevanta eller indirekta källor hellre än att låta dem
  påverka svaret.
- Om ingen källa tydligt besvarar frågan: säg det i första meningen och
  stanna där eller återge högst vad källorna faktiskt säger.

GRUNDREGLER FÖR FORM:

- Lägg inte till exempel, förklaringar eller generaliseringar som saknar
  tydligt stöd i källorna.
- Undvik formuleringar som antyder mer än källan säger, såsom
  "inkluderar", "brukar", "vanligtvis", "typiskt" eller liknande, om
  inte just detta stöds av källan.
- Ange källa efter varje påstående med [Källa N].
- Inled direkt med svaret.

{background_block}{operation_block}Källor:
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
    question_operation: str = "direct_lookup",
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
        operation_block=OPERATION_BLOCKS.get(question_operation, ""),
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