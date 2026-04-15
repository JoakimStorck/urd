# URD

Local AI-powered document assistant for internal governance documents.

URD reads internal policy documents, indexes their content, and answers questions with source references. It is built for local operation using open models and local storage. No data leaves the machine.

The name refers to the Norse Norn who held knowledge of all that has happened. It resembles *ord* (Swedish for “word”) and has a Unix-like quality: `urd ask "..."`. It can also be read as an acronym for **Unified Retrieval and Deliberation**.

---

## What it does

You point URD at a folder of internal documents — policies, procedures, delegation orders, meeting minutes. It extracts text, indexes it, and lets you ask questions in plain language. Answers cite specific sources so you can verify them.

URD is not a general-purpose chat assistant. It is a document assistant that stays close to its sources.

It can run in two modes:

- **server mode** (`urd serve`) — the full document assistant, with indexing, retrieval and answer generation
- **client mode** (`urd connect`) — a lightweight local client that serves the web UI locally and proxies requests to a URD server

This makes it possible to run the heavy system on one machine and use the interface from another.

---

## How it works

URD uses a retrieval-augmented generation (RAG) architecture with several layers designed to handle the real difficulty of internal documents: users ask in their own everyday language, not in the terminology the documents happen to use.

**Ingestion.** Documents (PDF, DOCX, XLSX) are extracted via Docling, split into sections by heading structure, and chunked. Each chunk is prefixed with its document title and section heading before embedding, so the vector representation carries context — a chunk that says “this applies” knows what “this” refers to.

**Hybrid retrieval.** Questions are matched against the index using both semantic search (multilingual-e5-large via Qdrant) and BM25 keyword search. BM25 catches exact terminology that semantic search misses. The two candidate pools are merged.

**Cross-encoder reranking.** A multilingual cross-encoder (mmarco-mMiniLMv2) scores each candidate against the question. Candidates with negative scores are filtered out. This replaced roughly 550 lines of hand-written heuristic reranking that could not generalise across question types.

**Document expansion.** For top-scoring documents, remaining chunks from the same document are retrieved and reranked. This ensures that when a document is relevant as a whole — a complete procedure with delegation rules, process flows and meeting formats — the system finds the practical details, not just the chunk that happened to match.

**Two-stage synthesis.** The LLM generates answers in two steps. First, it extracts evidence: short paraphrases tied to specific sources, with a confidence marker. Second, it formulates the answer from the extracted evidence only. This reduces inverted logic, fabrication and half-translation compared to single-step generation. If evidence extraction fails, the system falls back to single-step generation so the user always gets an answer.

**Conversation memory.** Follow-up questions are rewritten into standalone questions using conversation context, so “tell me more about the decision meeting” works after asking about research applications.

**Client/server mode.** In client mode, URD serves the web UI locally and proxies `/chat` and `/document` to a remote URD server. This avoids browser issues with direct access to the server and gives a fast local interface on the client machine.

---

## Architecture

| Component | Role |
|---|---|
| [Docling](https://github.com/DS4SD/docling) | Document extraction (PDF, DOCX, XLSX) |
| [Sentence Transformers](https://www.sbert.net/) | Local embeddings and cross-encoder reranking |
| [Qdrant](https://qdrant.tech/) | Local vector database |
| [Ollama](https://ollama.com/) | Local LLM inference |
| FastAPI | API server |
| Web UI | Chat interface with source display |

### Models

The system uses four models, each chosen for its specific role:

| Function | Model | Why |
|---|---|---|
| Embeddings | `intfloat/multilingual-e5-large` | Multilingual model that handles Swedish well. Encodes both queries and chunks for semantic search. |
| Reranking | `jeffwan/mmarco-mMiniLMv2-L12-H384-v1` | Multilingual cross-encoder. Chosen after the English-only model (`ms-marco-MiniLM`) failed to distinguish Swedish compounds like `kursansvar` and `kursutbud`. |
| Answer generation | Mistral-Nemo 12B (via Ollama) | Replaced Mistral 7B, which distorted logic in delegation rules and mixed terms inconsistently. Nemo handles synthesis from English sources to Swedish answers better, though not perfectly. |
| Metadata extraction | Mistral 7B (via Ollama) | Structured metadata extraction (keywords, roles, document type) is a simpler task that does not require the synthesis ability of a larger model. |

Using different models per role reflects a deliberate trade-off: each task has different requirements for language understanding, reasoning and speed. The reranker and embedding model run as local inference with sentence-transformers; the generative models run through Ollama.

---

## Installation

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running
- Models pulled:
  - `ollama pull mistral-nemo`
  - `ollama pull mistral`

### Install

```bash
pipx install .
````

### Verify

```bash
urd --help
```

---

## Quick start

### 1. Index documents

```bash
urd ingest
```

### 2. Start the server

```bash
urd serve
```

Open:

```text
http://127.0.0.1:8000
```

### 3. Ask questions from CLI

```bash
urd ask "What rules apply when hiring a doctoral student?"
```

### 4. Connect from another machine

On the client machine:

```bash
urd connect --server pop-os:8000
```

This starts a local client UI, typically on:

```text
http://127.0.0.1:8765
```

The browser then talks only to the local client, while the client proxies requests to the URD server.

---

## Configuration

Configuration is resolved with the following priority:

1. CLI flags
2. Environment variables
3. `.urd/config.json`
4. hard-coded defaults

### View and change configuration

```bash
urd config                    # Show all values and their source
urd config get top_k          # Show a specific value
urd config set top_k 5        # Set a value
urd config set server pop-os:8000
urd config reset              # Reset to defaults
```

### Key settings

| Setting           | Default                                | Description                               |
| ----------------- | -------------------------------------- | ----------------------------------------- |
| `docs_path`       | `./docs`                               | Document folder                           |
| `server`          | empty                                  | Default upstream server for `urd connect` |
| `ollama_model`    | `mistral-nemo`                         | Model for answer generation               |
| `top_k`           | `3`                                    | Number of sources used per answer         |
| `chunk_size`      | `1200`                                 | Max characters per chunk                  |
| `embedding_model` | `intfloat/multilingual-e5-large`       | Embedding model                           |
| `reranker_model`  | `jeffwan/mmarco-mMiniLMv2-L12-H384-v1` | Cross-encoder model                       |

---

## CLI reference

```bash
urd serve                          # Start API server and web UI
urd serve --top-k 5                # Start with custom top_k

urd connect --server pop-os:8000   # Start local client UI connected to server
urd connect                        # Use server from config

urd ask "question"                 # Ask a question (auto-detects server)
urd ask "question" --debug         # Show timing and retrieval details
urd ask "question" --via-server    # Force server mode
urd ask "follow-up" --new-session  # Start fresh session

urd ingest                         # Index new and changed documents
urd ingest --force                 # Re-index all documents
urd reindex                        # Reset index and re-ingest everything
urd enrich                         # Run LLM metadata extraction
urd stats                          # Show index and document status
urd config                         # Show configuration
urd config set key value           # Change a setting
urd test                           # Run test battery
urd test --no-answers              # Run tests, show only timing
urd reset-index                    # Delete and recreate the index
```

---

## Testing

URD includes a test harness for evaluating answer quality and timing.

```bash
cat > .urd/questions.json << 'EOF'
[
  {
    "question": "What rules apply when hiring a doctoral student?",
    "notes": "Should mention employment duration, 20% teaching"
  },
  {
    "question": "What is required for a successful dissertation?",
    "notes": "Should mention examination requirements, public defence"
  }
]
EOF

urd test
```

Results are saved under:

```text
.urd/results/
```

Each test run records answers, sources, timing breakdowns and synthesis diagnostics.

---

## Project structure

```text
app/
  api.py              # FastAPI server
  cli.py              # CLI commands
  config.py           # Configuration with .urd/config.json support
  connect_api.py      # Local client mode for urd connect
  embeddings.py       # Embedding model wrapper
  followup.py         # Follow-up question rewriting
  ingest.py           # Document extraction and chunking
  llm.py              # Ollama LLM wrapper
  preprocess_llm.py   # LLM-based metadata extraction
  prompting.py        # Single-step prompt (fallback)
  qdrant_store.py     # Qdrant vector store
  retrieval.py        # Hybrid retrieval pipeline
  schemas.py          # Data models
  session_state.py    # Conversation state
  synthesis.py        # Two-stage evidence extraction + answer generation
  static/
    index.html        # Web interface
```

---

## Constraints

* **Local operation is a hard requirement.** No cloud APIs, no data leaving the machine.
* **Open models and local storage are required.**
* **The system does not speculate.** Answers are based on retrieved sources.
* **Uncertainty is communicated.**
* **Sources are a verification mechanism, not a convenience.**

---

## Known limitations

* **Terminology gaps.** Questions using different words than the documents (for example “employment” vs “admission”) may miss relevant content. A claims layer, planned but not yet implemented, would address this.
* **Synthesis errors.** The LLM can still invert logic in complex delegation structures, mix Swedish and English terms, or add information not present in sources. Two-stage synthesis reduces but does not eliminate this.
* **Difficult source formats.** Documents mixing languages, table-based rules and process diagrams as images are harder for the system to interpret correctly.
* **Client mode is still young.** `urd connect` works, but client update logic and security hardening are still under development.

---

## License

Licensed under the Apache License 2.0. See `LICENSE` for details.

