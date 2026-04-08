from app.schemas import SourceHit

SYSTEM_PROMPT = """Du är en lokal dokumentassistent för interna styrdokument.

Regler:
- Svara endast utifrån återfunna källor.
- Återge inte uppgifter som inte uttryckligen stöds i källorna.
- Om flera tidsuppgifter eller ansvarsfördelningar förekommer, skilj dem tydligt åt.
- Om underlaget är oklart eller delvis motsägelsefullt, säg det.
- Om källorna inte räcker för en viss del, skriv det uttryckligen.
- Hänvisa med [Källa N] i de punkter där uppgifterna används.
- Avsluta med en kort lista över vilka källor svaret bygger på.
"""

def build_prompt(question: str, hits: list[SourceHit]) -> str:
    context_blocks = []
    for i, hit in enumerate(hits, start=1):
        meta = hit.metadata
        label = (
            f"[Källa {i}] fil={meta.file_name}; "
            f"rubrik={meta.section_title}; "
            f"sida={meta.page_number}; "
            f"väg={meta.source_path}"
        )
        context_blocks.append(f"{label}\n{hit.text}")

    context = "\n\n---\n\n".join(context_blocks)

    return f"""{SYSTEM_PROMPT}

Fråga:
{question}

Källmaterial:
{context}

Instruktion:
Svara på svenska i korta punkter.

Strukturera svaret under dessa rubriker när underlag finns:
1. Syfte
2. Ansvar
3. Tidpunkt
4. Underlag/handlingar
5. Genomförande

Viktigt:
- Ta inte med något som inte uttryckligen stöds av källmaterialet.
- Om en rubrik saknar stöd i källorna, skriv inte en gissning.
- Skilj tydligt mellan olika tidsfrister och olika ansvariga roller.
"""