"""
Rework: elaboration och verification utan ny retrieval-kedja.

Dessa vägar opererar på material som redan hämtats i föregående
dokumenttur snarare än på en helt ny retrieval. Men de två
uppgifterna är olika till sin natur, och har därför olika arkitektur:

- elaboration: formuleringsuppgift — "hitta och presentera material
  som inte var med i förra svaret". Löses med en direktprompt i samma
  stil som huvudsyntesens B1-prompt. Ingen JSON-mellanrepresentation,
  ingen strukturerad extraktion. Det empiriska utfallet av tvåstegs-
  extraktion för elaboration var att den komprimerade bort detaljer —
  exakt samma problem som gjorde att huvudvägen övergavs.

- verification: klassificeringsuppgift — "vilka av användarens eller
  assistentens tidigare påståenden har stöd i källorna?". Här ger
  strukturerad mellanrepresentation faktiskt diagnostiskt värde:
  varje finding klassificeras som supported/unclear/unsupported
  med referens till den källa som stödet kommer från. Det är
  denna klassificering som *är* produkten.

Båda funktionerna returnerar SynthesisResult för kompatibilitet med
huvudsyntesens returtyp — men SynthesisResult.verification är bara
satt för verify(), och elaborate() sätter det till None.

För elaboration krävs att kallaren först gör en egen retrieval av
material som INTE var med i föregående svar. Den retrieval-logiken
ligger i RagService.retrieve_for_elaboration i retrieval.py, inte
här — detta håller rework.py fritt från retrieval-beroenden och
därmed enklare att testa och resonera om.
"""

from __future__ import annotations

import json
import logging
import re
import time

from app.llm import LocalLLM
from app.schemas import SourceHit
from app.synthesis_types import (
    SynthesisResult,
    VerificationFinding,
    VerificationReport,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Formatering av källor (lokal kopia av mönstret från synthesis.py)
# ---------------------------------------------------------------------------

def _format_sources(hits: list[SourceHit]) -> str:
    """Formatera källor för prompter."""
    blocks = []
    for i, hit in enumerate(hits, start=1):
        meta = hit.metadata
        header = f"[Källa {i}] {meta.file_name} — {meta.section_title}"
        blocks.append(f"{header}\n{hit.text}")
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Elaboration — direktprompt, ingen mellanrepresentation
# ---------------------------------------------------------------------------

ELABORATE_PROMPT = """Du är en lokal dokumentassistent för interna styrdokument.
Användaren har bett dig utveckla ditt tidigare svar. Följande är NYA
källstycken från samma dokument som bar det tidigare svaret, men som
inte var med i det ursprungliga urvalet.

Ditt uppdrag är att BYGGA VIDARE på det tidigare svaret med material
ur dessa nya källor. Upprepa INTE det som redan sades.

KRITISKT för svarets användbarhet:
- BEVARA ALLA KONKRETA DETALJER från källorna: belopp, gränsvärden,
  roller, tidsfrister, procedurer, villkor, undantag.
- OM EN KÄLLA INNEHÅLLER EN NUMRERAD LISTA: återge ALLA poster i
  listan, i samma ordning. Komprimera inte listor.
- OM EN KÄLLA INNEHÅLLER EN TABELL eller strukturerad uppställning:
  återge den i sin helhet.
- ANVÄND KÄLLORNAS EXAKTA TERMER för formella moment, roller och
  procedurer.
- Ange källa efter varje påstående med [Källa N].
- Inled gärna kort med att signalera att du lägger till detaljer
  till det tidigare svaret (t.ex. "Utöver det jag nämnde tidigare
  gäller också...").
- Om de nya källorna INTE tillför något meningsfullt utöver vad som
  redan sades, var ärlig: säg att det tidigare svaret i huvudsak
  täckte det som står i källorna, och föreslå en mer specifik fråga.

TIDIGARE SVAR (för att undvika upprepning):
{previous_answer}

Nya källor:
{sources}

Aktuell yttring:
{question}

Svar:"""


ELABORATE_EMPTY_ANSWER = (
    "Det tidigare svaret återgav i huvudsak det som står i källdokumenten. "
    "Jag hittar inget ytterligare material som besvarar frågan. "
    "För mer detaljer på en specifik aspekt kan du ställa en mer riktad fråga."
)


def elaborate(
    question: str,
    new_hits: list[SourceHit],
    previous_answer: str,
    llm: LocalLLM,
) -> SynthesisResult:
    """
    Utveckla tidigare svar med material som inte var med första gången.

    new_hits ska vara chunks som retrieval-kedjan har identifierat som
    relevanta men som INTE bar det tidigare svaret — typiskt sektioner
    från samma dokument som aldrig kom igenom det ursprungliga urvalet.

    Om new_hits är tom abstainar elaboration ärligt snarare än att
    upprepa tidigare svar.
    """
    if not new_hits:
        return SynthesisResult(
            answer=ELABORATE_EMPTY_ANSWER,
            verification=None,
            used_fallback=False,
            timing_s={},
        )

    prompt = ELABORATE_PROMPT.format(
        previous_answer=previous_answer,
        sources=_format_sources(new_hits),
        question=question,
    )

    t0 = time.perf_counter()
    answer = llm.generate(prompt)
    t1 = time.perf_counter()

    return SynthesisResult(
        answer=answer,
        verification=None,
        used_fallback=False,
        timing_s={"elaborate": round(t1 - t0, 3)},
    )


# ---------------------------------------------------------------------------
# Verification — tvåstegs med smal datamodell
# ---------------------------------------------------------------------------

VERIFICATION_EXTRACT_PROMPT = """Användaren prövar eller ifrågasätter ditt tidigare svar.
Du ska strikt granska påståendena i det tidigare svaret mot källorna.

TIDIGARE SVAR TILL ANVÄNDAREN:
{previous_answer}

ANVÄNDARENS PRÖVNING:
{question}

Identifiera de påståenden i det tidigare svaret som är relevanta för
användarens prövning. För varje påstående, klassificera dess stöd i
källorna:

- "supported": påståendet står uttryckligen i källan. Ange exakt
  källa (t.ex. "Källa 2").
- "unclear": påståendet kräver tolkning av källan eller bygger på
  att man drar en slutsats. Ange den mest relevanta källan.
- "unsupported": påståendet har inget stöd alls i källorna. source
  kan vara null.

Var strikt. Ett påstående som i tidigare svar framställdes som
säkert men som bara kan härledas via tolkning ska klassas som
"unclear", inte "supported".

Om användarens prövning handlar om något som INTE nämndes i tidigare
svar men som finns i källorna, får du lägga till findings om detta
också — med lämplig status.

Svara ENBART med JSON, utan förklaringar eller markdown:
{{
  "findings": [
    {{
      "claim": "kort beskrivning av påståendet",
      "status": "supported|unclear|unsupported",
      "source": "Källa N" eller null
    }}
  ]
}}

Källor:
{sources}"""


VERIFICATION_ANSWER_PROMPT = """Du är en lokal dokumentassistent för interna styrdokument.
Användaren har prövat ett tidigare svar. Du har granskat det mot källorna
och klassificerat påståendena som supported, unclear eller unsupported.

Formulera nu ett svar som ÄRLIGT redogör för granskningen:

- Lyft fram påståenden som är "supported" och ange [Källa N] direkt
  efter dem.
- För "unclear"-påståenden, markera försiktigheten: använd
  formuleringar som "detta tycks innebära", "troligen", "kan tolkas
  som", och förklara vad tolkningen bygger på.
- För "unsupported"-påståenden, säg det rakt ut: "Jag kan inte hitta
  stöd i källorna för X." Undvik omskrivningar.
- Om det tidigare svaret innehöll påståenden utan stöd, medge det
  öppet. Trogenhet mot källorna är viktigare än att försvara det
  tidigare svaret.
- Om granskningen inte räcker för att bekräfta eller avfärda
  användarens prövning, abstäng tydligt och säg vad som skulle
  behövas.

Skriv på svenska. Var kortfattad men inte avvisande.

TIDIGARE SVAR (som användaren prövar):
{previous_answer}

GRANSKNING (findings från källorna):
{findings_text}

ANVÄNDARENS YTTRING:
{question}

Svar:"""


_VALID_STATUSES = {"supported", "unclear", "unsupported"}


def _parse_verification_json(raw: str) -> VerificationReport | None:
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
        logger.warning("Kunde inte parsa verification-JSON: %s", e)
        return None

    try:
        findings: list[VerificationFinding] = []
        for item in data.get("findings", []):
            status_raw = str(item.get("status", "")).strip().lower()
            if status_raw not in _VALID_STATUSES:
                logger.info("Ogiltig status %r — hoppar över finding.", status_raw)
                continue

            source_raw = item.get("source")
            source: str | None = None
            if source_raw is not None and str(source_raw).strip().lower() not in ("", "null", "none"):
                source = str(source_raw).strip()

            findings.append(
                VerificationFinding(
                    claim=str(item.get("claim", "")).strip(),
                    status=status_raw,  # type: ignore[arg-type]
                    source=source,
                )
            )

        return VerificationReport(findings=findings, raw_json=text)
    except Exception as e:
        logger.warning("Kunde inte validera verification-struktur: %s", e)
        return None


def _format_findings(report: VerificationReport) -> str:
    """Formatera findings som läsbar text för svarssteget."""
    if not report.findings:
        return "(Inga påståenden kunde klassificeras.)"

    lines = []
    for f in report.findings:
        source_part = f" [{f.source}]" if f.source else ""
        lines.append(f"- [{f.status}]{source_part} {f.claim}")
    return "\n".join(lines)


def verify(
    question: str,
    hits: list[SourceHit],
    previous_answer: str,
    llm: LocalLLM,
) -> SynthesisResult:
    """
    Granska tidigare svar mot källorna.

    Körs i två steg: först klassificeras påståendena i tidigare svar
    som supported/unclear/unsupported, sedan formuleras ett
    granskningssvar utifrån klassificeringen. Mellanrepresentationen
    (VerificationReport) bifogas i SynthesisResult för debugbarhet.

    Om JSON-parsningen misslyckas returneras en ärlig abstain snarare
    än att falla tillbaka på enstegsformulering — verification utan
    strukturerad klassificering tappar sitt diagnostiska värde och
    vore bedrägligt att presentera som en granskning.
    """
    t0 = time.perf_counter()

    extract_prompt = VERIFICATION_EXTRACT_PROMPT.format(
        previous_answer=previous_answer,
        question=question,
        sources=_format_sources(hits),
    )
    raw = llm.generate(extract_prompt)
    report = _parse_verification_json(raw)

    t1 = time.perf_counter()

    if report is None:
        logger.warning("Verification: JSON-parsning misslyckades, abstainar.")
        return SynthesisResult(
            answer=(
                "Jag kunde inte genomföra granskningen av mitt tidigare svar just nu. "
                "Försök gärna omformulera frågan."
            ),
            verification=None,
            used_fallback=True,
            fallback_reason="verification_parse_failed",
            timing_s={"extract": round(t1 - t0, 3)},
        )

    if not report.findings:
        return SynthesisResult(
            answer=(
                "Jag hittar inget i källorna som tydligt bekräftar eller avfärdar "
                "det du prövar. För en tydligare granskning behövs ytterligare "
                "dokument eller en mer specifik frågeställning."
            ),
            verification=report,
            used_fallback=False,
            timing_s={"extract": round(t1 - t0, 3)},
        )

    answer_prompt = VERIFICATION_ANSWER_PROMPT.format(
        previous_answer=previous_answer,
        findings_text=_format_findings(report),
        question=question,
    )
    answer = llm.generate(answer_prompt)
    t2 = time.perf_counter()

    return SynthesisResult(
        answer=answer,
        verification=report,
        used_fallback=False,
        timing_s={
            "extract": round(t1 - t0, 3),
            "formulate": round(t2 - t1, 3),
        },
    )