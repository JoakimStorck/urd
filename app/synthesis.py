"""
Tvåstegssyntes: evidensextraktion följd av svarsformulering.

Steg 1 (extract_evidence) ber LLM:en identifiera relevanta textstycken
nära källan och returnera dem som strukturerad JSON. Syftet är att
disciplinera modellen — den ska parafrasera kort, ange exakt källa,
och markera om tolkning krävdes.

Steg 2 (generate_answer) formulerar svaret enbart utifrån den
extraherade evidensen. Aspekter utan stöd kommuniceras explicit.

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

Fråga:
{question}

Källmaterial:
{sources}"""


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

Extraherade textstycken:
{evidence_json}

Fråga:
{question}"""


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

def extract_evidence(
    question: str,
    hits: list[SourceHit],
    llm: LocalLLM,
) -> EvidenceResult | None:
    """
    Steg 1: Be LLM:en identifiera relevanta textstycken nära källan.
    Returnerar None om parsningen misslyckas.
    """
    sources_text = _format_sources_for_evidence(hits)

    prompt = EVIDENCE_PROMPT_TEMPLATE.format(
        question=question,
        sources=sources_text,
    )

    raw = llm.generate(prompt)

    return _parse_evidence_json(raw)


def generate_answer(
    question: str,
    evidence: EvidenceResult,
    llm: LocalLLM,
) -> str:
    """
    Steg 2: Formulera svar enbart utifrån extraherad evidens.
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

    prompt = ANSWER_PROMPT_TEMPLATE.format(
        question=question,
        evidence_json=evidence_text,
    )

    return llm.generate(prompt)


def synthesize(
    question: str,
    hits: list[SourceHit],
    llm: LocalLLM,
) -> SynthesisResult:
    """
    Kör tvåstegssyntes med fallback till enstegsflödet.

    Returnerar alltid ett SynthesisResult med svar. Om tvåstegs-
    syntesen misslyckas (JSON-parsningsfel) används enstegsflödet
    och detta markeras i resultatet.
    """
    # Steg 1: evidensextraktion
    t0 = time.perf_counter()
    evidence = extract_evidence(question, hits, llm)
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
    answer = generate_answer(question, evidence, llm)
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