from app.schemas import SourceHit

SYSTEM_PROMPT = """Du är en lokal dokumentassistent för interna styrdokument.

Regler:
- Svara endast utifrån återfunna källor.
- Återge inte uppgifter som inte uttryckligen stöds i källorna.
- Om underlaget är oklart eller delvis motsägelsefullt, säg det kort.
- Hänvisa med [Källa N] direkt efter de påståenden där uppgiften används.
- Skilj mellan allmänt giltiga regler och enskilda tillämpningsfall.
- Använd bara enskilda exempel om det bidrar till svarets relevans för frågan.
- Använd bara källor som faktiskt tillför något till svaret.
- Upprepa inte samma information i flera delar av svaret.
- Säg inte att dokumentation eller anvisningar saknas om en återfunnen källa själv innehåller anvisningen, huvudregeln eller huvudbeskrivningen.
- För generella frågor om regler, ansvar eller uppdrag: utelämna enskilda exempel om de inte behövs för att besvara frågan.
- När en normkälla eller anvisning finns bland källorna, låt den väga tyngre än historiska protokolluppgifter.
- Beskriv inte ett historiskt protokollpåstående som gällande regel om en återfunnen normkälla anger den aktuella regeln eller anvisningen.
"""


def _fmt_list(values: list[str]) -> str:
    return ", ".join(values) if values else "-"


def build_prompt(question: str, hits: list[SourceHit]) -> str:
    context_blocks = []
    for i, hit in enumerate(hits, start=1):
        meta = hit.metadata
        label = (
            f"[Källa {i}] "
            f"fil={meta.file_name}; "
            f"rubrik={meta.section_title}; "
            f"sida={meta.page_number}; "
            f"väg={meta.source_path}; "
            f"dokumenttyp={meta.document_type}; "
            f"nyckelord={_fmt_list(meta.keywords)}; "
            f"roller={_fmt_list(meta.roles)}; "
            f"handlingar={_fmt_list(meta.actions)}; "
            f"tid={_fmt_list(meta.time_markers)}; "
            f"gäller_för={_fmt_list(meta.applies_to)}; "
            f"sammanfattning={meta.section_summary or '-'}"
        )
        context_blocks.append(f"{label}\n{hit.text}")

    context = "\n\n---\n\n".join(context_blocks)

    return f"""{SYSTEM_PROMPT}

Fråga:
{question}

Källmaterial:
{context}

Instruktion:
- Svara på svenska.
- Anpassa svarets form efter frågan.
- Om frågan är enkel och faktabaserad, svara direkt och kort.
- Om frågan efterfrågar flera uppgifter, använd en kort punktlista.
- Om frågan gäller process, rutin eller genomförande, använd en kort struktur med bara de rubriker som faktiskt behövs.
- Utelämna delar som saknar stöd i källorna.
- Skriv inte med fler rubriker eller fler punkter än vad frågan kräver.
- Prioritera det viktigaste först.
- Håll svaret kort och undvik utfyllnad.
"""