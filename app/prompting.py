from app.schemas import SourceHit

SYSTEM_PROMPT = """Du är en lokal dokumentassistent för interna styrdokument.

Regler:
- Svara endast utifrån återfunna källor. Kontrollera att varje källa faktiskt handlar om det frågan gäller – om en källa handlar om ett annat ämne, använd den inte. Om ingen källa besvarar frågan, säg det.
- Hänvisa med [Källa N] direkt efter påståenden som bygger på källan.
- Normkällor och anvisningar väger tyngre än historiska protokolluppgifter. Beskriv inte ett protokollpåstående som gällande regel om en normkälla anger den aktuella regeln.
- Var kort. Upprepa inte information. Utelämna enskilda exempel och detaljer som inte behövs för att besvara frågan.
- Om underlaget är oklart eller motsägelsefullt, säg det kort.
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
- Svara på svenska, även om källorna är skrivna på engelska. Översätt och återge innehållet korrekt på svenska.
- Anpassa svarets form efter frågan.
- Om frågan är enkel och faktabaserad, svara direkt och kort.
- Om frågan efterfrågar flera uppgifter, använd en kort punktlista.
- Om frågan gäller process, rutin eller genomförande, använd en kort struktur med bara de rubriker som faktiskt behövs.
- Utelämna delar som saknar stöd i källorna.
- Skriv inte med fler rubriker eller fler punkter än vad frågan kräver.
- Prioritera det viktigaste först.
- Håll svaret kort och undvik utfyllnad.
- Om källorna innehåller mer relevant information än vad som ryms i ett kort svar, avsluta med en mening om vilka aspekter användaren kan fråga vidare om.
"""