# docchat

Lokal AI-baserad dokumentchat för interna styrdokument.

`docchat` är en tunn lokal RAG-lösning för att läsa interna dokument, indexera dem och besvara frågor med tydlig källvisning. Fokus ligger på administrativa styrdokument, rutiner, beslut och instruktioner. Systemet är byggt för lokal körning på en enskild maskin och använder lokala modeller och lokal lagring.

Projektet försöker inte vara en allmän chattassistent. Det är en avgränsad dokumentassistent med källnära svar.

---

## Innehåll

- [Syfte](#syfte)
- [Arkitektur](#arkitektur)
- [Dokumentflöde](#dokumentflöde)
- [Installation](#installation)
- [Konfiguration](#konfiguration)
- [CLI](#cli)
- [Köra tjänsten manuellt](#köra-tjänsten-manuellt)
- [Status](#status)
- [Avgränsningar](#avgränsningar)
- [Frågesvarspolicy](#frågesvarspolicy)
- [Nästa steg](#nästa-steg)

---

## Syfte

Målet är att demonstrera hur en lokal AI-baserad dokumentassistent kan användas för att läsa interna styrdokument, hitta relevanta avsnitt, sammanställa svar på svenska och visa vilka källor svaret bygger på.

---

## Arkitektur

Systemet består av en lättviktig lokal kedja:

| Komponent | Roll |
|---|---|
| [Docling](https://github.com/DS4SD/docling) | Dokumentextraktion (PDF, DOCX, XLSX) |
| [Sentence Transformers](https://www.sbert.net/) | Lokala embeddings |
| [Qdrant](https://qdrant.tech/) | Lokal vektordatabas |
| [Ollama](https://ollama.com/) | Lokal modellkörning |
| FastAPI | Lokal API-tjänst |
| Webbgränssnitt | Chat och källvisning |

### Utvecklingslinje

Projektet följer en tydlig ordning:

1. **Deterministisk strukturutvinning** — dokument exporteras till markdown, delas i sektioner utifrån rubriker, chunkas inom sektion.
2. **Semantisk metadataextraktion** — lokal LLM annoterar varje sektion med strukturerad metadata (dokumenttyp, roller, begrepp, tidsmarkörer, sammanfattning).
3. **Retrieval och svarsgenerering** — relevanta segment hämtas ur Qdrant, lokal modell genererar svar utifrån återfunna källor.

### Katalogstruktur

```text
app/
  api.py
  cli.py
  config.py
  embeddings.py
  ingest.py
  llm.py
  preprocess_llm.py
  prompting.py
  qdrant_store.py
  retrieval.py
  schemas.py
  static/
    index.html
docs/
  IIT-lokala regler och rutiner/
    ...
scripts/
  ingest_docs.py
```

---

## Dokumentflöde

```
docs/          →  Docling      →  Markdown
Markdown       →  Sektioner    →  Rubrikbaserad uppdelning
Sektioner      →  LLM          →  Semantisk metadata per sektion
Sektioner      →  Chunkning    →  Segment (vid behov)
Segment        →  Embeddings   →  Qdrant
Fråga          →  Retrieval    →  Relevanta segment
Segment        →  LLM          →  Svar med källhänvisningar
```

**Ingest** — dokument under `docs/` läses in och extraheras med Docling.

**Struktur** — dokumentet exporteras till markdown och delas i sektioner utifrån rubriker. Chunkning sker bara om en sektion är för lång; kortare sektioner hålls ihop.

**Annotering** — varje sektion kan annoteras med semantisk metadata via lokal LLM: dokumenttyp, nyckelbegrepp, roller, handlingar, tidsmarkörer och en kort sammanfattning.

**Indexering** — segment embeddas och lagras i Qdrant med tillhörande metadata i payload.

**Fråga/svar** — frågan embeddas, relevanta segment hämtas, en prompt byggs från källorna, lokal modell genererar svar, källor visas tillsammans med svaret.

---

## Installation

### Förutsättningar

- Python 3.10+
- [Ollama](https://ollama.com/) installerat och körande lokalt
- Vald modell nedladdad i Ollama (t.ex. `ollama pull mistral`)

### Installera beroenden

```bash
pip install -r requirements.txt
```

### Installera CLI via pipx

```bash
pipx install --editable .
```

```bash
docchat --help
```

---

## Konfiguration

Konfiguration hämtas från miljövariabler via `.env` eller från defaultvärden i `app/config.py`.

| Variabel | Beskrivning |
|---|---|
| `DOCS_PATH` | Sökväg till dokumentkatalogen |
| `QDRANT_PATH` | Lokal sökväg för Qdrant-data |
| `QDRANT_COLLECTION` | Namn på Qdrant-kollektionen |
| `EMBEDDING_MODEL` | Embeddingmodell (default: `intfloat/multilingual-e5-large`) |
| `OLLAMA_MODEL` | Modell för frågesvar (default: `mistral`) |
| `PREPROCESS_OLLAMA_MODEL` | Modell för semantisk annotering |
| `TOP_K` | Antal segment som hämtas per fråga |
| `CHUNK_SIZE` | Max tecken per chunk |
| `CHUNK_OVERLAP` | Överlapp i tecken mellan chunkar |
| `PREPROCESS_ENABLED` | Aktivera LLM-baserad annotering (`true`/`false`) |

---

## CLI

### Visa hjälp

```bash
docchat --help
```

### Starta webbservern

```bash
docchat serve
```

### Läs in dokument

```bash
docchat ingest
```

### Återskapa index

```bash
docchat reset-index
```

### Återskapa index och kör ingest

```bash
docchat reindex
```

### Ställ en fråga direkt från terminalen

```bash
docchat ask "Vad gäller för halvtidsseminarium?"
```

```bash
docchat ask "Vad gäller för halvtidsseminarium?" --debug
```

```bash
docchat ask "Vad gäller för halvtidsseminarium?" --no-sources
```

---

## Köra tjänsten manuellt

```bash
uvicorn app.api:app --reload
```

Öppna sedan `http://127.0.0.1:8000/` i en webbläsare.

---

## Status

### Genomfört

- Lokal end-to-end-kedja
- Dokumentingest för PDF, DOCX och XLSX
- Lokal embedding med `multilingual-e5-large`
- Lokal vektorsökning via Qdrant
- Lokal LLM för frågesvar via Ollama
- FastAPI-tjänst med `/chat`-endpoint
- Enkelt webbgränssnitt
- Sektionsbaserad chunkning från markdown
- Metadatafält för sektionsrubrik och rubriknivå

### Pågående

- LLM-baserad semantisk annotering per sektion (`preprocess_llm.py`)
- Utökade semantiska metadatafält i indexet
- Förbättrad retrieval med metadata och lätt omrankning

---

## Avgränsningar

Projektet prioriterar för närvarande inte:

- Finjustering av modellen
- Avancerad agentlogik
- Fleranvändarstöd
- Nätverksexponering
- Full produktionssäkerhet
- Tung ramverksorkestrering

De största tekniska riskerna ligger i dokumentbehandlingen. PDF-struktur kan vara svårtolkad, tabeller kan extraheras dåligt och rubriker kan gå förlorade om texten plattas till för tidigt. Semantisk metadata från LLM kan bli övergenererad om prompt och validering är för svaga. Projektet hanterar detta genom deterministisk strukturutvinning i första steget, strikt JSON-schema för LLM-extraktion, låg temperatur och validering av strukturerad output.

---

## Frågesvarspolicy

Systemet ska svara endast utifrån återfunna källor, inte fylla i luckor med gissningar, markera osäkerhet när underlaget inte räcker och alltid visa vilka källor svaret bygger på.

Det viktigaste är inte att systemet låter allvetande, utan att det på ett kontrollerat sätt hittar och sammanställer relevant innehåll ur verkliga dokument.

---

## Nästa steg

- Färdigställa preprocess-LLM per sektion
- Lagra semantiska metadatafält i Qdrant
- Använda metadata i retrieval och enkel omrankning
- Förbättra källpanelen med rikare metadata
- Lägga till enkel intern utvärdering av retrieval-kvalitet