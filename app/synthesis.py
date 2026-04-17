"""
Tvåstegssyntes: evidensextraktion följd av svarsformulering.

Steg 1 (extract_evidence) ber LLM:en identifiera relevanta textstycken
nära källan och returnera dem som strukturerad JSON. Syftet är att
disciplinera modellen — den ska parafrasera kort, ange exakt källa,
och markera om tolkning krävdes.

Steg 2 (generate_answer) formulerar svaret enbart utifrån den
extraherade evidensen. Aspekter utan stöd kommuniceras explicit.

Svarsformuleringen kan styras av en stilparameter som kommer från
QUD-klassificeringen i api-lagret:
- None: standardstil (default, fungerar för new_main_question).
- "subquestion": fokuserat svar på en delfråga inom etablerad
  huvudfråga — kort och avgränsat.
- "broadening": breddande svar som lyfter fram vad mer som finns
  i området runt huvudfrågan.
- "narrowing_or_repair": preciserande svar efter att föregående svar
  varit för grovt eller missförstått.
- "verification": striktare stöd krävs, tydligare abstention när
  källorna inte räcker.

Utöver synthesize() finns rework() som används för elaboration och
verification. Den gör INGEN ny retrieval utan opererar på föregående
turs källor (session.active_hits) och föregående svar. Evidens-
extraktionen instrueras att fokusera på material som INTE kom med i
föregående svar, eller — för verification — att strikt kontrollera
att tidigare påståenden faktiskt har källstöd.

Om JSON-parsningen i steg 1 misslyckas faller systemet tillbaka på
enstegsflödet (build_prompt + generate) så att användaren alltid
får ett svar. Fallback loggas i debug-objektet.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field

from app.llm import LocalLLM
from app.prompting import build_prompt
from app.schemas import SourceHit

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Datamodeller för evidens
# ---------------------------------------------------------------------------

@dataclass
class EvidenceItem:
    text: str
    source: str
    confidence: str  # "explicit" | "tolkning_krävdes"


@dataclass
class EvidenceResult:
    extracted: list[EvidenceItem] = field(default_factory=list)
    not_found: list[str] = field(default_factory=list)
    raw_json: str | None = None


@dataclass
class SynthesisResult:
    answer: str
    evidence: EvidenceResult | None = None
    used_fallback: bool = False
    fallback_reason: str | None = None
    timing_s: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Prompter
# ---------------------------------------------------------------------------

def _format_sources_for_evidence(hits: list[SourceHit]) -> str:
    """Formatera källor för evidensextraktionsprompten."""
    blocks = []
    for i, hit in enumerate(hits, start=1):
        meta = hit.metadata
        header = f"[Källa {i}] fil={meta.file_name}; rubrik={meta.section_title}"
        blocks.append(f"{header}\n{hit.text}")
    return "\n\n---\n\n".join(blocks)


def _format_sources_for_direct(hits: list[SourceHit]) -> str:
    """
    Formatera källor för den direkta enstegsprompten (B1-stil).

    Skillnad mot _format_sources_for_evidence: här används en något
    luftigare layout som underlättar för modellen att se sektioner
    och deras innehåll tydligt åtskilda.
    """
    blocks = []
    for i, hit in enumerate(hits, start=1):
        meta = hit.metadata
        header = f"[Källa {i}] {meta.file_name} — {meta.section_title}"
        blocks.append(f"{header}\n{hit.text}")
    return "\n\n".join(blocks)


# Direct-prompten (motsvarar B1 från experimentskriptet).
# Enstegssyntes med strikta strukturregler: återge numrerade listor
# i sin helhet, tabeller i sin helhet, använd källans exakta termer.
DIRECT_SYNTHESIS_PROMPT = """Du är en lokal dokumentassistent för interna styrdokument.
Svara på frågan enbart utifrån källorna nedan.

KRITISKT för svarets användbarhet:
- BEVARA ALLA KONKRETA DETALJER från källorna: belopp, gränsvärden
  (t.ex. "500 tkr"), roller ("prefekt", "rektor", "Head of School"),
  tidsfrister, procedurer, villkor, undantag.

- OM EN KÄLLA INNEHÅLLER EN NUMRERAD LISTA: återge ALLA poster i
  listan, i samma ordning. Utelämna inga steg, även om de verkar
  triviala eller liknar varandra. En process med 16 steg ska
  återges med 16 steg, inte sammanfattas till 8.

- OM EN KÄLLA INNEHÅLLER EN TABELL eller strukturerad uppställning
  (till exempel tidsramar, roller, belopp): återge den i sin helhet
  som tabell eller strukturerad lista. Komprimera inte tabeller till
  löptext.

- ANVÄND KÄLLORNAS EXAKTA TERMER för formella moment, roller och
  procedurer. Om källan säger "intervju" — säg "intervju", inte
  "samtal". Om källan säger "öppen nominering och ansökningsförfarande"
  — använd den termen, inte "ansökning och validering".

- Ange källa efter varje påstående med [Källa N].
- Inled direkt med det mest specifika svaret på frågan. Inga
  inledande "inramningar" eller "för att besvara frågan om...".
- Om källorna täcker aspekter bortom frågan, nämn INTE att de saknas.
  Bara besvara det som faktiskt frågades.

{background_block}Källor:
{sources_block}

Fråga: {question}

Svar:"""


EVIDENCE_PROMPT_TEMPLATE = """Läs källorna noggrant. Identifiera de textställen
som kan vara relevanta för frågan.

Tänk brett kring relevans: om frågan handlar om "anställning" och källorna
beskriver "antagning", "handledning" eller "utbildningstid" för samma roll,
är det relevant. Användare formulerar sig ofta i vardagsspråk — leta efter
innehåll som handlar om samma sakområde, inte bara exakt samma ord.

Men kontrollera att varje källa faktiskt handlar om det frågan gäller.
Om frågan gäller doktorander, inkludera inte information om docenter,
postdoktorer eller andra roller bara för att de nämns i samma dokument.
Om en källa handlar om en annan roll eller ett annat ämne, hoppa över den.

För varje relevant textstycke:
- Parafrasera det nära originalet. Bevara konkreta detaljer: belopp,
  beloppsgränser, roller, tidsfrister, villkor och beslutsordning.
  Dessa detaljer är ofta det viktigaste i svaret.
- Ange exakt vilken källa (Källa N) det kommer från
- Markera om du behövde tolka eller om det står uttryckligen

Extrahera hellre för många textstycken än för få. Det är bättre att
ta med ett tveksamt relevant stycke än att missa viktig information.

Svara ENBART med JSON, utan förklaringar eller markdown:
{{
  "extracted": [
    {{
      "text": "parafras nära originalet med konkreta detaljer bevarade",
      "source": "Källa N",
      "confidence": "explicit"
    }}
  ],
  "not_found": ["aspekter av frågan som källorna inte täcker"]
}}

confidence ska vara "explicit" om informationen står uttryckligen i källan,
eller "tolkning_krävdes" om du behövde tolka eller dra en slutsats.

Om källorna inte innehåller relevant information, returnera en tom
extracted-lista och beskriv vad som saknas i not_found.
{background_block}
Fråga:
{question}

Källmaterial:
{sources}"""


BACKGROUND_BLOCK_TEMPLATE = """
SAMTALSBAKGRUND (endast som kontext för att förstå frågan):
{background_text}

VIKTIGT om samtalsbakgrunden:
- Den är INTE en källa. Påståenden i "extracted" får ENDAST komma
  från källmaterialet nedan, aldrig från samtalsbakgrunden.
- Den hjälper dig tolka vad frågan syftar på (t.ex. vad "andra regler"
  eller "det" refererar till), men den är inte faktamaterial.
- Även om samtalsbakgrunden innehåller påståenden som verkar relevanta,
  ska de inte upprepas i extracted om de inte också stöds av källmaterialet.
"""


ANSWER_PROMPT_TEMPLATE = """Du är en lokal dokumentassistent för interna styrdokument.

Formulera ett svar på svenska baserat enbart på dessa extraherade
textstycken. Använd inte information som inte finns i listan.

Börja direkt med vad källorna säger. Svara inte med "det finns ingen
information om..." om du faktiskt har relevanta uppgifter att redovisa.

Om frågan använder ett annat ord än källorna för samma sak (t.ex.
"anställning" när källorna säger "antagning"), koppla ihop dem och
svara utifrån det som källorna faktiskt beskriver. Nämn kort vilka
termer källorna använder om det skiljer sig väsentligt från frågan.

Om ett påstående är markerat som "tolkning_krävdes", formulera
det med reservation (t.ex. "detta tycks innebära" eller "troligen").

Hänvisa med [Källa N] direkt efter påståenden som bygger på källan.
Var kort. Upprepa inte information.

Om det finns aspekter av frågan som extrakten inte täcker alls,
nämn det kort i slutet av svaret.

Svara på svenska, även om extrakten innehåller engelska termer.
{style_block}
Extraherade textstycken:
{evidence_json}

Fråga:
{question}"""


# Stilspecifika instruktioner som sätts in i ANSWER_PROMPT_TEMPLATE
# via {style_block}. Tom sträng för standardstil (new_main_question).
STYLE_BLOCKS = {
    None: "",
    "subquestion": """
Stil: Detta är en delfråga inom en pågående huvudtråd. Användaren vill
ha ett fokuserat svar på just denna delaspekt — utgå från att
helhetskontexten redan är etablerad. Upprepa inte bakgrund som
användaren redan fått. Håll svaret kort och avgränsat till delfrågan.
""",
    "broadening": """
Stil: Användaren vidgar sin huvudfråga och vill se vad mer som finns
i området. Lyft fram de relevanta aspekterna som källorna täcker,
även sådana som inte direkt besvarar frågan men är närliggande och
användbara. Strukturera gärna svaret som en översikt.
""",
    "narrowing_or_repair": """
Stil: Användaren preciserar eller rättar efter att föregående svar
varit för grovt, otydligt, eller missförstått deras avsikt. Fokusera
specifikt på den preciserade aspekten. Om tidigare svar byggde på
fel tolkning, var tydlig med att det nu handlar om något mer
avgränsat eller specifikt.
""",
    "verification": """
Stil: Användaren prövar eller ifrågasätter ett tidigare påstående.
Var därför särskilt noggrann med källstöd:
- Påståenden ska bygga på "explicit"-markerad evidens där det går.
- Om ett tidigare påstående (i samtalsbakgrunden eller i källorna)
  bara har "tolkning_krävdes"-stöd eller saknar stöd helt, säg det
  rakt ut.
- Om källorna INTE räcker för att bekräfta det användaren prövar,
  abstäng tydligt: säg vad som skulle behövas och att det inte finns
  tillräckligt stöd i återfunna källor.
- Ge inga nya påståenden utan tydligt källstöd.
""",
}


# ---------------------------------------------------------------------------
# JSON-parsning
# ---------------------------------------------------------------------------

def _parse_evidence_json(raw: str) -> EvidenceResult | None:
    """
    Försök parsa LLM-svaret som JSON.
    Hanterar markdown-fences och fritext runt JSON-blocket.
    Returnerar None om parsningen misslyckas.
    """
    text = raw.strip()

    # Strippa markdown-fences om de finns
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    # Försök hitta ett JSON-objekt om det finns fritext runt det
    if not text.startswith("{"):
        brace_start = text.find("{")
        if brace_start >= 0:
            text = text[brace_start:]

    # Trimma trailing-text efter sista }
    last_brace = text.rfind("}")
    if last_brace >= 0:
        text = text[: last_brace + 1]

    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Kunde inte parsa evidens-JSON: %s", e)
        return None

    # Validera och bygg EvidenceResult
    try:
        extracted = []
        for item in data.get("extracted", []):
            extracted.append(
                EvidenceItem(
                    text=str(item.get("text", "")),
                    source=str(item.get("source", "")),
                    confidence=str(item.get("confidence", "explicit")),
                )
            )

        not_found = [str(x) for x in data.get("not_found", [])]

        return EvidenceResult(
            extracted=extracted,
            not_found=not_found,
            raw_json=text,
        )
    except Exception as e:
        logger.warning("Kunde inte validera evidensstruktur: %s", e)
        return None


# ---------------------------------------------------------------------------
# Tvåstegssyntes
# ---------------------------------------------------------------------------

def _format_background(turns: list[dict], max_turns: int) -> str:
    """
    Formatera de senaste turerna som bakgrundstext för evidensprompten.

    Varje "tur" i config-bemärkelse är ett fråga-svar-par (2 entries i
    turns-listan). Returnerar tom sträng om ingen historik eller om
    max_turns <= 0.
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
        # Korta ned långa svar för att hålla prefill-kostnaden nere
        if len(content) > 600:
            content = content[:600] + "..."
        lines.append(f"{role}: {content}")

    return "\n".join(lines)


def extract_evidence(
    question: str,
    hits: list[SourceHit],
    llm: LocalLLM,
    background_turns: list[dict] | None = None,
    background_max_turns: int = 0,
) -> EvidenceResult | None:
    """
    Steg 1: Be LLM:en identifiera relevanta textstycken nära källan.
    Returnerar None om parsningen misslyckas.

    Om background_turns och background_max_turns > 0 läggs ett
    samtalsbakgrunds-block in i prompten. Bakgrunden används ENDAST för
    att hjälpa modellen förstå vad frågan syftar på — den får inte
    bidra med påståenden i extracted-listan.
    """
    sources_text = _format_sources_for_evidence(hits)

    background_block = ""
    if background_turns and background_max_turns > 0:
        background_text = _format_background(background_turns, background_max_turns)
        if background_text:
            background_block = BACKGROUND_BLOCK_TEMPLATE.format(
                background_text=background_text,
            )

    prompt = EVIDENCE_PROMPT_TEMPLATE.format(
        question=question,
        sources=sources_text,
        background_block=background_block,
    )

    raw = llm.generate(prompt)

    return _parse_evidence_json(raw)


def generate_answer(
    question: str,
    evidence: EvidenceResult,
    llm: LocalLLM,
    style: str | None = None,
) -> str:
    """
    Steg 2: Formulera svar enbart utifrån extraherad evidens.

    style styr svarsformuleringen — se STYLE_BLOCKS. Okända
    stilmarkörer behandlas som None (standardstil).
    """
    # Bygg en läsbar representation av evidensen för prompten
    evidence_entries = []
    for item in evidence.extracted:
        conf_marker = ""
        if item.confidence == "tolkning_krävdes":
            conf_marker = " [tolkning krävdes]"
        evidence_entries.append(
            f"- {item.source}: {item.text}{conf_marker}"
        )

    if evidence.not_found:
        evidence_entries.append("")
        evidence_entries.append("Saknas i källorna:")
        for gap in evidence.not_found:
            evidence_entries.append(f"- {gap}")

    evidence_text = "\n".join(evidence_entries)

    style_block = STYLE_BLOCKS.get(style, "")

    prompt = ANSWER_PROMPT_TEMPLATE.format(
        question=question,
        evidence_json=evidence_text,
        style_block=style_block,
    )

    return llm.generate(prompt)


def synthesize(
    question: str,
    hits: list[SourceHit],
    llm: LocalLLM,
    background_turns: list[dict] | None = None,
    background_max_turns: int = 0,
    style: str | None = None,
) -> SynthesisResult:
    """
    Enstegssyntes (direct) med B1-prompten: detaljbevarande och
    strukturregler (numrerade listor i sin helhet, tabeller i sin
    helhet, källans exakta termer).

    Ersätter tidigare tvåstegsflöde. Experimentet
    scripts/synthesis_experiment.py visade att tvåstegsarkitekturen
    komprimerade bort konkreta detaljer (belopp, roller, procedurer)
    och att B1 på samma hits producerar märkbart fylligare svar.
    Den gamla tvåstegsvarianten finns kvar som synthesize_twostep()
    för jämförande experiment.

    style-parametern behålls för API-kompatibilitet men ignoreras i
    denna enstegsimplementation — stilvariationer hanteras redan av
    prompten och av rework-vägens egna prompter.
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
        evidence=None,  # ingen mellanrepresentation i enstegsflödet
        used_fallback=False,
        timing_s={
            "direct_synthesis": round(t1 - t0, 3),
        },
    )


def synthesize_twostep(
    question: str,
    hits: list[SourceHit],
    llm: LocalLLM,
    background_turns: list[dict] | None = None,
    background_max_turns: int = 0,
    style: str | None = None,
) -> SynthesisResult:
    """
    Gammalt tvåstegsflöde. Bevaras för jämförande experiment men
    används inte längre i den ordinarie syntesvägen. Se synthesize()
    ovan för den nya enstegsvarianten.
    """
    # Steg 1: evidensextraktion
    t0 = time.perf_counter()
    evidence = extract_evidence(
        question,
        hits,
        llm,
        background_turns=background_turns,
        background_max_turns=background_max_turns,
    )
    t1 = time.perf_counter()

    if evidence is None:
        # JSON-parsning misslyckades — fallback till enstegsflödet
        logger.warning(
            "Tvåstegssyntes misslyckades (JSON-parsning). "
            "Faller tillbaka på enstegsflödet."
        )
        fallback_prompt = build_prompt(question, hits)
        answer = llm.generate(fallback_prompt)
        t2 = time.perf_counter()
        return SynthesisResult(
            answer=answer,
            evidence=None,
            used_fallback=True,
            fallback_reason="evidence_json_parse_failed",
            timing_s={
                "evidence_extraction": round(t1 - t0, 3),
                "fallback_generation": round(t2 - t1, 3),
            },
        )

    if not evidence.extracted:
        # Inga relevanta textstycken hittades — kommunicera ärligt
        if evidence.not_found:
            gaps = "; ".join(evidence.not_found)
            answer = (
                f"Källorna täcker inte frågan tillräckligt. "
                f"Följande saknas: {gaps}"
            )
        else:
            answer = (
                "Jag hittar inget tydligt stöd i de återfunna källorna "
                "för att besvara frågan."
            )
        return SynthesisResult(
            answer=answer,
            evidence=evidence,
            used_fallback=False,
            timing_s={
                "evidence_extraction": round(t1 - t0, 3),
            },
        )

    # Steg 2: formulera svar från evidens
    answer = generate_answer(question, evidence, llm, style=style)
    t2 = time.perf_counter()

    return SynthesisResult(
        answer=answer,
        evidence=evidence,
        used_fallback=False,
        timing_s={
            "evidence_extraction": round(t1 - t0, 3),
            "answer_generation": round(t2 - t1, 3),
        },
    )


# ---------------------------------------------------------------------------
# Rework: elaboration och verification på föregående källor
#
# Dessa vägar opererar INTE på ny retrieval utan på föregående turs
# active_hits. Målet är att arbeta med samma material som bar förra
# svaret, men med ett annat fokus:
# - elaboration: lyft fram vad som prioriterades bort
# - verification: strikt granskning av vad som redan sagts
# ---------------------------------------------------------------------------

ELABORATION_EVIDENCE_PROMPT_TEMPLATE = """Ditt tidigare svar till användaren var
komprimerat och kan ha utelämnat detaljer. Användaren ber dig nu
utveckla svaret genom att gå tillbaka till SAMMA källor som bar
det tidigare svaret, och lyfta fram sådant som INTE kom med.

TIDIGARE SVAR TILL ANVÄNDAREN:
{previous_answer}

Läs källorna på nytt och identifiera textstycken som är relevanta
för frågan men som INTE redan redovisades i det tidigare svaret.
Om en detalj redan nämndes i tidigare svar — utelämna den eller
markera den som redan given.

För varje NY textstycke:
- Parafrasera det nära originalet. Bevara konkreta detaljer:
  belopp, tidsfrister, roller, villkor, beslutsordning.
- Ange exakt vilken källa (Källa N).
- Markera om du behövde tolka eller om det står uttryckligen.

Om källorna INTE innehåller mer relevant material än vad som redan
redovisats, returnera en tom extracted-lista. Var ärlig — det är
bättre att säga "tidigare svar uttömde källorna" än att återupprepa.

Svara ENBART med JSON, utan förklaringar eller markdown:
{{
  "extracted": [
    {{
      "text": "parafras nära originalet med konkreta detaljer bevarade",
      "source": "Källa N",
      "confidence": "explicit"
    }}
  ],
  "not_found": ["aspekter som inte täcks i källorna alls"]
}}

Fråga:
{question}

Källmaterial (samma som bar det tidigare svaret):
{sources}"""


VERIFICATION_EVIDENCE_PROMPT_TEMPLATE = """Användaren ifrågasätter eller prövar ditt
tidigare svar. Du ska nu strikt granska vad som faktiskt har källstöd
i samma källor som bar det tidigare svaret.

TIDIGARE SVAR TILL ANVÄNDAREN:
{previous_answer}

Granska källorna och identifiera, för varje påstående som är
relevant för användarens prövning:
- Om det står uttryckligen i källan (confidence: "explicit")
- Om det krävs tolkning för att nå slutsatsen (confidence: "tolkning_krävdes")

Var strikt: ett påstående som i tidigare svar framställdes som
säkert men som egentligen bygger på tolkning ska markeras som
"tolkning_krävdes" här.

För påståenden som INTE har stöd i källorna alls — lägg dem i
not_found som "saknar stöd: [påstående]".

Om det tidigare svaret innehöll påståenden utan källstöd, är det
viktigt att notera det. Trogenhet mot källorna är här viktigare
än att försvara det tidigare svaret.

Svara ENBART med JSON, utan förklaringar eller markdown:
{{
  "extracted": [
    {{
      "text": "parafras av det påstående som har stöd",
      "source": "Källa N",
      "confidence": "explicit|tolkning_krävdes"
    }}
  ],
  "not_found": ["påståenden utan stöd, eller aspekter utan täckning"]
}}

Fråga:
{question}

Källmaterial (samma som bar det tidigare svaret):
{sources}"""


ELABORATION_ANSWER_PROMPT_TEMPLATE = """Du är en lokal dokumentassistent för interna
styrdokument. Användaren har bett dig utveckla ditt tidigare svar.
Följande är NYA textstycken ur samma källor som bar det tidigare svaret.

Formulera ett svar som BYGGER VIDARE på det tidigare utan att
upprepa det. Nämn gärna inledningsvis att du lägger till detaljer.

Regler:
- Använd bara den extraherade evidensen.
- Hänvisa med [Källa N] direkt efter påståenden.
- Upprepa inte det som redan sades i tidigare svar.
- Om ett påstående är markerat "tolkning_krävdes", formulera med
  reservation.
- Svara på svenska.

TIDIGARE SVAR (för att undvika upprepning):
{previous_answer}

Nya textstycken:
{evidence_json}

Aktuell yttring:
{question}"""


VERIFICATION_ANSWER_PROMPT_TEMPLATE = """Du är en lokal dokumentassistent för interna
styrdokument. Användaren prövar eller ifrågasätter ditt tidigare svar.

Formulera ett svar som ÄRLIGT redogör för vad källorna faktiskt stöder:
- Lyft fram det som har tydligt ("explicit") källstöd och citera [Källa N].
- Markera påståenden som krävde tolkning som osäkra — använd
  formuleringar som "detta tycks innebära" eller "troligen" och
  förklara vad tolkningen bygger på.
- Om det tidigare svaret innehöll påståenden UTAN källstöd, säg det
  rakt ut: "Jag kan inte hitta stöd i källorna för X" — använd inte
  omskrivningar.
- Om källorna inte räcker för att bekräfta eller avfärda det
  användaren prövar, abstäng tydligt: säg vad som skulle behövas.

Regler:
- Trogenhet mot källorna är viktigare än att försvara det tidigare svaret.
- Om det tidigare svaret var för säkert på vissa punkter, medge det.
- Hänvisa med [Källa N] direkt efter påståenden.
- Svara på svenska.

TIDIGARE SVAR (som användaren prövar):
{previous_answer}

Granskad evidens:
{evidence_json}

Aktuell yttring:
{question}"""


def _extract_rework_evidence(
    question: str,
    hits: list[SourceHit],
    previous_answer: str,
    llm: LocalLLM,
    mode: str,
) -> EvidenceResult | None:
    """
    Extrahera evidens från föregående källor, med en prompt som
    beror på rework-läget (elaboration eller verification).
    """
    sources_text = _format_sources_for_evidence(hits)

    if mode == "elaboration":
        template = ELABORATION_EVIDENCE_PROMPT_TEMPLATE
    elif mode == "verification":
        template = VERIFICATION_EVIDENCE_PROMPT_TEMPLATE
    else:
        logger.warning("Okänt rework-läge: %r. Använder elaboration.", mode)
        template = ELABORATION_EVIDENCE_PROMPT_TEMPLATE

    prompt = template.format(
        question=question,
        sources=sources_text,
        previous_answer=previous_answer,
    )

    raw = llm.generate(prompt)
    return _parse_evidence_json(raw)


def _generate_rework_answer(
    question: str,
    evidence: EvidenceResult,
    previous_answer: str,
    llm: LocalLLM,
    mode: str,
) -> str:
    """Formulera rework-svar baserat på evidens och tidigare svar."""
    evidence_entries = []
    for item in evidence.extracted:
        conf_marker = ""
        if item.confidence == "tolkning_krävdes":
            conf_marker = " [tolkning krävdes]"
        evidence_entries.append(f"- {item.source}: {item.text}{conf_marker}")

    if evidence.not_found:
        evidence_entries.append("")
        evidence_entries.append("Saknar stöd eller täckning:")
        for gap in evidence.not_found:
            evidence_entries.append(f"- {gap}")

    evidence_text = "\n".join(evidence_entries)

    if mode == "verification":
        template = VERIFICATION_ANSWER_PROMPT_TEMPLATE
    else:
        template = ELABORATION_ANSWER_PROMPT_TEMPLATE

    prompt = template.format(
        question=question,
        evidence_json=evidence_text,
        previous_answer=previous_answer,
    )

    return llm.generate(prompt)


def rework(
    question: str,
    hits: list[SourceHit],
    previous_answer: str,
    llm: LocalLLM,
    mode: str,
) -> SynthesisResult:
    """
    Arbeta mot föregående turs källor utan ny retrieval.

    mode styr syftet:
    - "elaboration": lyft fram det som prioriterades bort i tidigare svar
    - "verification": strikt granskning av vad som faktiskt har källstöd

    Returnerar alltid ett SynthesisResult. Om evidens-JSON inte går
    att parsa används enstegsflöde som fallback. Om ingen ny evidens
    kan extraheras för elaboration abstainar rework ärligt snarare
    än att upprepa tidigare svar.
    """
    t0 = time.perf_counter()
    evidence = _extract_rework_evidence(question, hits, previous_answer, llm, mode)
    t1 = time.perf_counter()

    if evidence is None:
        # JSON-parsningsfel — för rework gör vi ingen ny retrieval.
        # Returnera en ärlig abstain.
        logger.warning(
            "Rework (%s) evidens-JSON kunde inte parsas — abstainar.", mode,
        )
        if mode == "verification":
            answer = (
                "Jag kunde inte genomföra granskningen av mitt tidigare "
                "svar just nu. Försök gärna omformulera frågan."
            )
        else:
            answer = (
                "Jag kunde inte utveckla mitt tidigare svar just nu. "
                "Försök gärna omformulera frågan, eller ställ den "
                "annorlunda."
            )
        return SynthesisResult(
            answer=answer,
            evidence=None,
            used_fallback=True,
            fallback_reason=f"rework_{mode}_parse_failed",
            timing_s={"evidence_extraction": round(t1 - t0, 3)},
        )

    if not evidence.extracted:
        # För elaboration: om inget nytt fanns att lägga till, var ärlig.
        # För verification: om inget hade stöd, var ännu ärligare.
        if mode == "verification":
            if evidence.not_found:
                gaps = "; ".join(evidence.not_found)
                answer = (
                    f"Vid granskning saknar följande tydligt stöd i "
                    f"källorna: {gaps}. Det kan betyda att det tidigare "
                    f"svaret gick längre än källorna medger på dessa "
                    f"punkter."
                )
            else:
                answer = (
                    "Jag hittar inget som ifrågasätter eller bekräftar "
                    "det tidigare svaret i de källor som bar det. För "
                    "en tydligare prövning behövs ytterligare dokument."
                )
        else:
            answer = (
                "Det tidigare svaret redovisade i huvudsak det som står "
                "i de återfunna källorna. För ytterligare detaljer "
                "behövs andra dokument — du kan ställa en ny fråga "
                "med mer specifik inriktning."
            )
        return SynthesisResult(
            answer=answer,
            evidence=evidence,
            used_fallback=False,
            timing_s={"evidence_extraction": round(t1 - t0, 3)},
        )

    answer = _generate_rework_answer(question, evidence, previous_answer, llm, mode)
    t2 = time.perf_counter()

    return SynthesisResult(
        answer=answer,
        evidence=evidence,
        used_fallback=False,
        timing_s={
            "evidence_extraction": round(t1 - t0, 3),
            "answer_generation": round(t2 - t1, 3),
        },
    )