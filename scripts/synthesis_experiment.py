"""
Syntes-experiment.

Prövar tre syntesvarianter mot samma hits, för att isolera om
kvalitetsproblemen sitter i arkitektur, prompt, eller modell.

Körs från projektroten med URD-servern AVSTÄNGD:

    python scripts/synthesis_experiment.py

Skriptet bygger sin egen RagService och laddar modellerna lokalt.
Det delar inte resurser med en eventuellt körande server — och
Qdrant har fil-lås, så båda kan inte köras samtidigt.

Två frågor testas:

1. "Vilken process finns för att tillsätta en ny proprefekt?"
   — retrieval fungerar här, vi använder den som den är.
   Isolerar syntes-kvalitet när rätt material finns.

2. "Vad gäller för medfinansiering?" med MANUELLT valda hits från
   forskningsansökan-dokumentet. Isolerar syntes från retrieval —
   för att se om syntesen KAN svara väl när vi matar den med
   det material vi vet att den behöver.

Varje fråga körs i tre varianter:

A. Nuvarande tvåstegssyntes (rag.answer använder denna internt)
B. Enstegssyntes med detaljbevarande prompt
C. B + revideringspass där modellen uppmanas addera missade detaljer

Resultaten skrivs ut till terminalen och sparas i JSON för att kunna
läsas i lugn och ro efteråt.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Lägg till projektroten i sys.path så skriptet kan köras från
# projektroten utan att kräva PYTHONPATH=. Projektroten är
# föräldern till scripts/-katalogen.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import json
import time
from datetime import datetime

from app.retrieval import RagService
from app.schemas import SourceHit


# ---------------------------------------------------------------------------
# Prompter för varianterna
# ---------------------------------------------------------------------------

# Variant B: enstegssyntes med detaljbevarande prompt.
# Inga mellansteg — källorna visas rakt, modellen svarar direkt.
PROMPT_B = """Du är en lokal dokumentassistent för interna styrdokument.
Svara på frågan enbart utifrån källorna nedan.

KRITISKT för svarets användbarhet:
- BEVARA ALLA KONKRETA DETALJER från källorna: belopp, gränsvärden
  (t.ex. "500 tkr"), roller ("prefekt", "rektor", "Head of School"),
  tidsfrister, procedurer, villkor, undantag.
- Citera källornas formuleringar nära originaltexten när det gäller
  specifika sifferuppgifter, roller eller beslutsordningar.
- Ange källa efter varje påstående med [Källa N].
- Inled direkt med det mest specifika svaret på frågan. Inga
  inledande "inramningar" eller "för att besvara frågan om...".
- Om källorna täcker aspekter bortom frågan, nämn INTE att de saknas.
  Bara besvara det som faktiskt frågades.
- Om frågan kräver flera delsvar, strukturera i korta paragrafer —
  ett delsvar per paragraf.

Källor:
{sources_block}

Fråga: {question}

Svar:"""


# Variant B1: variant B + strikta strukturregler. Prövar om modellen
# kan fås att återge numrerade listor och tabeller i sin helhet, och
# behålla dokumentens exakta termer istället för att omformulera dem.
PROMPT_B1 = """Du är en lokal dokumentassistent för interna styrdokument.
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

Källor:
{sources_block}

Fråga: {question}

Svar:"""


# Variant B2: variant B1 + explicit checklista som modellen går igenom
# INNAN den formulerar svaret. Prövar om ett eget inventeringssteg
# hjälper modellen att inte glömma något.
PROMPT_B2 = """Du är en lokal dokumentassistent för interna styrdokument.
Svara på frågan enbart utifrån källorna nedan.

STEG 1 — INVENTERING (gör detta internt, skriv ut svaret nedan):
Gå igenom varje källa och notera för dig själv:
- Innehåller källan en numrerad lista? Hur många poster?
- Innehåller källan en tabell eller strukturerad uppställning?
- Vilka konkreta detaljer finns: belopp, tidsfrister, roller,
  specifika procedurer, villkor och undantag?

STEG 2 — FORMULERING:
Baserat på inventeringen, formulera svaret enligt dessa regler:

- BEVARA ALLA KONKRETA DETALJER: belopp, gränsvärden, roller,
  tidsfrister, procedurer, villkor, undantag.

- OM EN KÄLLA INNEHÅLLER EN NUMRERAD LISTA: återge ALLA poster, i
  samma ordning. Utelämna inga steg. En lista med 16 punkter ska
  återges med 16 punkter.

- OM EN KÄLLA INNEHÅLLER EN TABELL eller strukturerad uppställning
  (t.ex. tidsramar, roller, belopp): återge den i sin helhet som
  tabell eller strukturerad lista.

- ANVÄND KÄLLORNAS EXAKTA TERMER. Om källan säger "intervju" — säg
  "intervju", inte "samtal". Formulera inte egna termer.

- Ange källa efter varje påstående med [Källa N].
- Inled direkt med svaret. Inga inramningar.
- Nämn INTE att något saknas om frågan inte krävde det.

Källor:
{sources_block}

Fråga: {question}

Svar (skriv här, utan att visa inventeringen):"""


# Variant C: första passet = variant B, sedan ett revideringspass.
PROMPT_C_REVIEW = """Du har producerat följande svar på en fråga om interna styrdokument:

FRÅGA: {question}

FÖRSTA SVAR:
{first_answer}

KÄLLOR (samma som i första passet):
{sources_block}

UPPGIFT: Jämför första svaret mot källorna. Identifiera KONKRETA detaljer
som står i källorna men saknas eller är för generellt uttryckta i första
svaret. Konkreta detaljer innebär:

- Specifika belopp eller gränsvärden
- Namngivna roller och vem som beslutar vad
- Konkreta procedurer (inte bara "en process finns")
- Tidsfrister eller mandatperioder
- Villkor och undantag

Producera ett REVIDERAT SVAR som bevarar första svarets struktur men
lyfter in dessa detaljer med [Källa N]-hänvisningar. Om första svaret
redan bär detaljerna, upprepa det och ange det.

Reviderat svar:"""


# ---------------------------------------------------------------------------
# Hjälpfunktioner
# ---------------------------------------------------------------------------

def format_sources_block(hits: list[SourceHit]) -> str:
    """Formatera en lista av hits som källblock för prompten."""
    parts = []
    for i, h in enumerate(hits, start=1):
        parts.append(
            f"[Källa {i}] {h.metadata.file_name} — "
            f"{h.metadata.section_title}\n{h.text}"
        )
    return "\n\n".join(parts)


def run_variant_b(rag: RagService, question: str, hits: list[SourceHit]) -> tuple[str, float]:
    """Variant B: enstegssyntes med detaljbevarande prompt."""
    sources_block = format_sources_block(hits)
    prompt = PROMPT_B.format(sources_block=sources_block, question=question)
    t0 = time.perf_counter()
    answer = rag.llm.generate(prompt)
    return answer, time.perf_counter() - t0


def run_variant_b1(rag: RagService, question: str, hits: list[SourceHit]) -> tuple[str, float]:
    """Variant B1: B + strikta strukturregler (lista helt, tabell helt, exakta termer)."""
    sources_block = format_sources_block(hits)
    prompt = PROMPT_B1.format(sources_block=sources_block, question=question)
    t0 = time.perf_counter()
    answer = rag.llm.generate(prompt)
    return answer, time.perf_counter() - t0


def run_variant_b2(rag: RagService, question: str, hits: list[SourceHit]) -> tuple[str, float]:
    """Variant B2: B1 + inventeringssteg innan formulering."""
    sources_block = format_sources_block(hits)
    prompt = PROMPT_B2.format(sources_block=sources_block, question=question)
    t0 = time.perf_counter()
    answer = rag.llm.generate(prompt)
    return answer, time.perf_counter() - t0


def run_variant_c(
    rag: RagService,
    question: str,
    hits: list[SourceHit],
    first_answer: str,
) -> tuple[str, float]:
    """Variant C: revideringspass ovanpå första svaret."""
    sources_block = format_sources_block(hits)
    prompt = PROMPT_C_REVIEW.format(
        question=question,
        first_answer=first_answer,
        sources_block=sources_block,
    )
    t0 = time.perf_counter()
    answer = rag.llm.generate(prompt)
    return answer, time.perf_counter() - t0


def get_manual_hits_medfinansiering(rag: RagService) -> list[SourceHit]:
    """
    Hämta manuellt alla chunkar från forskningsansökan-dokumentet.
    Vi ger syntesen ALLT tillgängligt material — om det inte räcker
    har vi ett tydligt svar på att problemet är syntes, inte urval.
    """
    doc_path = "./docs/regler-for-ansokningar-om-extern-forskningsfinansiering.pdf"

    # Prova några vanliga varianter av path
    candidates_paths = [
        doc_path,
        "docs/regler-for-ansokningar-om-extern-forskningsfinansiering.pdf",
        "regler-for-ansokningar-om-extern-forskningsfinansiering.pdf",
    ]

    for p in candidates_paths:
        chunks = rag.bm25_index.get_chunks_by_source(p)
        if chunks:
            print(f"  Hittade {len(chunks)} chunkar i {p}")
            return chunks

    # Sista utvägen: leta i BM25-indexet efter dokumentet via filnamn
    print("  Varning: hittade inga chunkar via exakt path. Letar i indexet...")
    all_chunks = []
    # BM25-indexet är inte direkt iterbart — vi gör en bred sökning
    hits = rag.bm25_index.top_k(
        "extern forskningsfinansiering medfinansiering prefekt rektor",
        k=100,
    )
    for h in hits:
        if "forskningsfinansiering" in h.metadata.file_name.lower():
            all_chunks.append(h)

    # Dedup på chunk_id
    seen = set()
    unique = []
    for c in all_chunks:
        if c.chunk_id not in seen:
            seen.add(c.chunk_id)
            unique.append(c)

    print(f"  Hittade {len(unique)} unika chunkar via fallback-sökning")
    return unique


def print_section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def print_variant(label: str, answer: str, elapsed: float) -> None:
    print(f"\n--- VARIANT {label} ({elapsed:.1f}s) ---")
    print(answer.strip())


# ---------------------------------------------------------------------------
# Huvudflöde
# ---------------------------------------------------------------------------

def main() -> None:
    print("Startar syntes-experiment. Laddar modeller...")
    rag = RagService()
    print("Klart. Börjar körning.\n")

    results: dict = {
        "timestamp": datetime.now().isoformat(),
        "cases": [],
    }

    # -----------------------------------------------------------------------
    # FALL 1: Proprefekt — retrieval fungerar, vi använder den rakt av
    # -----------------------------------------------------------------------
    q1 = "Vilken process finns för att tillsätta en ny proprefekt?"
    print_section(f"FALL 1: {q1}")
    print("Retrieval körs normalt via rag.answer(). Vi plockar ut hits och")
    print("kör variant B och C mot samma material.\n")

    t0 = time.perf_counter()
    response_a = rag.answer(q1)
    time_a = time.perf_counter() - t0
    hits_1 = response_a.sources

    print(f"Retrieval gav {len(hits_1)} hits:")
    for i, h in enumerate(hits_1, start=1):
        print(f"  [{i}] {h.metadata.file_name} / {h.metadata.section_title} "
              f"(score={h.score:.2f})")

    print_variant("A (nuvarande tvåstegssyntes)", response_a.answer, time_a)

    answer_b, time_b = run_variant_b(rag, q1, hits_1)
    print_variant("B (enstegssyntes, detaljbevarande prompt)", answer_b, time_b)

    answer_b1, time_b1 = run_variant_b1(rag, q1, hits_1)
    print_variant("B1 (B + strikta strukturregler)", answer_b1, time_b1)

    answer_b2, time_b2 = run_variant_b2(rag, q1, hits_1)
    print_variant("B2 (B1 + inventeringssteg)", answer_b2, time_b2)

    answer_c, time_c = run_variant_c(rag, q1, hits_1, answer_b)
    print_variant("C (enstegssyntes + revideringspass)", answer_c, time_c)

    results["cases"].append({
        "question": q1,
        "retrieval": "automatic",
        "num_hits": len(hits_1),
        "hits_meta": [
            {
                "file_name": h.metadata.file_name,
                "section_title": h.metadata.section_title,
                "score": round(h.score, 3),
            }
            for h in hits_1
        ],
        "variants": {
            "A": {"answer": response_a.answer, "time_s": round(time_a, 2)},
            "B": {"answer": answer_b, "time_s": round(time_b, 2)},
            "B1": {"answer": answer_b1, "time_s": round(time_b1, 2)},
            "B2": {"answer": answer_b2, "time_s": round(time_b2, 2)},
            "C": {"answer": answer_c, "time_s": round(time_c, 2)},
        },
    })

    # -----------------------------------------------------------------------
    # FALL 2: Medfinansiering — retrieval abstainar, vi matar in manuellt
    # -----------------------------------------------------------------------
    q2 = "Vad gäller för medfinansiering?"
    print_section(f"FALL 2: {q2}")
    print("Retrieval abstainar på denna fråga i nuvarande system.")
    print("Vi matar syntesen med HELA forskningsansökan-dokumentet,")
    print("för att isolera syntes-förmåga från retrieval-precision.\n")

    hits_2 = get_manual_hits_medfinansiering(rag)

    if not hits_2:
        print("  FEL: hittade inga chunkar att arbeta mot.")
        print("  Hoppar över fall 2.\n")
    else:
        print(f"  Matar syntesen med {len(hits_2)} chunkar.\n")

        # Variant A för fall 2: kör den ordinarie tvåstegssyntesen mot våra
        # manuellt valda hits. synthesize() är själva tvåstegsflödet —
        # den gör extraktion + svarsformulering.
        from app.synthesis import synthesize

        t0 = time.perf_counter()
        synth_result = synthesize(q2, hits_2, rag.llm)
        answer_a_2 = synth_result.answer
        time_a_2 = time.perf_counter() - t0

        print_variant("A (nuvarande tvåstegssyntes)", answer_a_2, time_a_2)

        answer_b_2, time_b_2 = run_variant_b(rag, q2, hits_2)
        print_variant("B (enstegssyntes, detaljbevarande prompt)", answer_b_2, time_b_2)

        answer_b1_2, time_b1_2 = run_variant_b1(rag, q2, hits_2)
        print_variant("B1 (B + strikta strukturregler)", answer_b1_2, time_b1_2)

        answer_b2_2, time_b2_2 = run_variant_b2(rag, q2, hits_2)
        print_variant("B2 (B1 + inventeringssteg)", answer_b2_2, time_b2_2)

        answer_c_2, time_c_2 = run_variant_c(rag, q2, hits_2, answer_b_2)
        print_variant("C (enstegssyntes + revideringspass)", answer_c_2, time_c_2)

        results["cases"].append({
            "question": q2,
            "retrieval": "manual (whole document)",
            "num_hits": len(hits_2),
            "hits_meta": [
                {
                    "file_name": h.metadata.file_name,
                    "section_title": h.metadata.section_title,
                }
                for h in hits_2
            ],
            "variants": {
                "A": {"answer": answer_a_2, "time_s": round(time_a_2, 2)},
                "B": {"answer": answer_b_2, "time_s": round(time_b_2, 2)},
                "B1": {"answer": answer_b1_2, "time_s": round(time_b1_2, 2)},
                "B2": {"answer": answer_b2_2, "time_s": round(time_b2_2, 2)},
                "C": {"answer": answer_c_2, "time_s": round(time_c_2, 2)},
            },
        })

    # -----------------------------------------------------------------------
    # Spara resultaten
    # -----------------------------------------------------------------------
    results_dir = Path(".urd/experiments")
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = results_dir / f"synthesis_experiment_{timestamp}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print_section("KLART")
    print(f"Resultaten sparade i {output_path}")
    print("\nJämför varianterna manuellt. Frågor att ställa till varje svar:")
    print("  - Finns källornas numrerade listor återgivna FULLSTÄNDIGT?")
    print("  - Finns eventuell tidsramstabell med?")
    print("  - Används källans exakta termer, eller har modellen omformulerat?")
    print("  - Har något smugit in som INTE finns i källorna?")
    print("")
    print("Möjliga slutsatser:")
    print("  - B1/B2 märkbart bättre än B → prompt är spaken, tvåstegs är")
    print("    fel arkitektur men kan ersättas med bra enstegsprompt.")
    print("  - B2 märkbart bättre än B1 → inventeringssteget har verkligt värde.")
    print("  - Alla B-varianter lika tunna → modellens begränsning, inte prompt.")
    print("    Titta på kontextfönster och överväg modellbyte.")


if __name__ == "__main__":
    main()