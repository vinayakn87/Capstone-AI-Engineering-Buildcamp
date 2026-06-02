# Mutual Fund Factsheet RAG Agent — Build Plan

## Context

Starting from a fully empty codebase (only empty `__init__.py` files exist). The architecture is documented in `CLAUDE.md`. We are building a RAG system that:
1. Auto-scrapes the latest monthly factsheets from the AMFI India website to seed the corpus
2. Ingests mutual fund factsheet PDFs (auto-scraped or user-uploaded in Q&A mode)
3. Supports two distinct user interaction modes (see below)
4. Uses ChromaDB (swappable) + SQLite for the persistent knowledge base; OpenAI for synthesis

## Two Interaction Modes

### Mode 1 — Q&A Mode (knowledge-base driven)
- User types a question about a mutual fund
- Agent retrieves from the persistent knowledge base (ChromaDB + SQLite)
- **Fund present** → synthesize and answer
- **Fund absent** → agent says "I don't have data for [fund]. Please upload the factsheet." → user uploads PDF → PDF is **ingested into the knowledge base** → agent re-runs query → answers

### Mode 2 — Direct Upload Mode (session-only)
- User explicitly uploads a factsheet PDF before or while asking questions
- Uploaded PDF is parsed and chunked **in-memory / temp store** for this session only
- Agent uses **both** the session PDF content AND the persistent knowledge base for retrieval
- Uploaded PDF is **never persisted** to ChromaDB or SQLite — discarded when session ends
- Useful for: previewing a new factsheet, private/sensitive documents, or just-released PDFs not yet scraped

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│               MUTUAL FUND FACTSHEET RAG AGENT - ARCHITECTURE             │
└──────────────────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────┐
  │               SCRAPER (scraper/)                  │
  │  amfi_scraper.py + downloader.py                 │
  │  requests + BeautifulSoup                        │
  │  → PDFs land in data/raw_pdfs/                   │
  └──────────────────────┬───────────────────────────┘
                         │
                         ▼
  ┌──────────────────────────────────────────────────┐
  │       INGESTION PIPELINE (ingestion/)             │
  │  PDF Parser → Chunker → Embedder → Write stores  │
  │  pdfplumber   500-tok   sentence-transformers    │
  └──────────────────────┬───────────────────────────┘
           ┌─────────────┴──────────────┐
           ▼                            ▼
  ┌────────────────────┐    ┌──────────────────────────────┐
  │  STRUCTURED STORE  │    │   VECTOR STORE (persistent)  │
  │  SQLite            │    │   ChromaDB                   │
  │  (sqlite-utils)    │    │   [abstracted → swap Qdrant] │
  └────────────────────┘    └──────────────────────────────┘
                                        │
                         ┌──────────────┘
                         │
  ┌──────────────────────▼───────────────────────────┐
  │              AGENT (agent/)                       │
  │                                                  │
  │  retriever.py                                    │
  │  • searches persistent store (ChromaDB + SQLite) │
  │  • optionally merges session chunks (Mode 2)     │
  │  • raises FundNotFoundError if fund absent       │
  │    (Mode 1 only, when no session PDF either)     │
  │                                                  │
  │  synthesizer.py → OpenAI gpt-4o-mini             │
  └──────────────────────┬───────────────────────────┘
                         │
                         ▼
  ┌──────────────────────────────────────────────────┐
  │           CHAT UI (chat/app.py)  — Streamlit      │
  │                                                  │
  │  ┌─────────────────────────────────────────────┐ │
  │  │  MODE 1: Q&A                                │ │
  │  │  User types question                        │ │
  │  │  → Agent retrieves from KB                  │ │
  │  │  → FundNotFound? Show upload prompt         │ │
  │  │    → Upload PDF → ingest into KB → re-query │ │
  │  └─────────────────────────────────────────────┘ │
  │                                                  │
  │  ┌─────────────────────────────────────────────┐ │
  │  │  MODE 2: Direct Upload (session-only)       │ │
  │  │  User uploads PDF → parsed in-memory        │ │
  │  │  User asks questions                        │ │
  │  │  → Agent uses session chunks + KB           │ │
  │  │  → PDF never written to KB                  │ │
  │  │  → Chunks discarded on session end          │ │
  │  └─────────────────────────────────────────────┘ │
  └──────────────────────────────────────────────────┘
```

## Build Order (step-by-step, bottom-up)

### Step 0 — Architecture Diagram
- Save `docs/architecture.md` with the diagram above (Mermaid + ASCII)

### Step 1 — Store Layer Abstractions
- `store/base.py` — abstract base classes: `VectorStore`, `StructuredStore`
  - `VectorStore`: `add(docs)`, `search(query_embedding, top_k) -> list[Chunk]`, `has_fund(fund_name) -> bool`, `count() -> int`
  - `StructuredStore`: `upsert_fund(metadata)`, `query(filters) -> list[dict]`, `has_fund(fund_name) -> bool`
- `store/vector_store.py` — ChromaDB implementation of `VectorStore` (persistent)
- `store/structured_store.py` — SQLite implementation of `StructuredStore`
- `store/session_store.py` — in-memory `VectorStore` implementation for Mode 2 session chunks (no persistence; same interface as `VectorStore`)

### Step 2 — Ingestion Pipeline
- `ingestion/pdf_parser.py` — `pdfplumber` text + table extraction; returns `FundDocument`
- `ingestion/chunker.py` — sliding window chunker (~500 tokens, ~50 overlap)
- `ingestion/embedder.py` — `sentence-transformers` all-MiniLM-L6-v2 embedding generation
- `ingestion/pipeline.py`:
  - `ingest_pdf(path, persist=True)` — parse → chunk → embed → write to stores if `persist=True`; returns `list[Chunk]` always
  - When `persist=False` (Mode 2): returns chunks without touching ChromaDB or SQLite
  - CLI: `python -m ingestion.pipeline` (processes `data/raw_pdfs/`, always `persist=True`)

### Step 3 — AMFI Scraper
- `scraper/__init__.py`
- `scraper/amfi_scraper.py` — scrapes AMFI India for latest monthly factsheet links per fund house
- `scraper/downloader.py` — downloads PDFs into `data/raw_pdfs/`, skips already-downloaded
- `scraper/runner.py` — CLI: `python -m scraper.runner` → scrape + download + ingest

### Step 4 — Agent
- `agent/retriever.py`:
  - `retrieve(question, session_chunks=None)` — embeds question → searches persistent ChromaDB + SQLite; if `session_chunks` provided, merges them too; if no results and no session chunks → raises `FundNotFoundError(fund_name)`
- `agent/synthesizer.py` — context + question → OpenAI → answer
- `agent/agent.py`:
  - `ask(question, session_chunks=None) -> AgentResponse`
  - `AgentResponse`: `Answer(text)` | `FundNotFound(fund_name)`

### Step 5 — Chat UI
- `chat/app.py` — Streamlit app with two modes toggled in sidebar:

  **Mode 1 (Q&A):**
  - Chat input → `agent.ask(question, session_chunks=None)`
  - `FundNotFound` → display message → show `st.file_uploader` → on upload → `ingest_pdf(path, persist=True)` → re-run query

  **Mode 2 (Direct Upload):**
  - Sidebar PDF uploader → `ingest_pdf(path, persist=False)` → store chunks in `st.session_state.session_chunks`
  - Chat input → `agent.ask(question, session_chunks=session_state.session_chunks)`
  - Session cleared on new upload or page refresh

### Step 6 — Tests
- `tests/test_ingestion.py` — unit tests for parser, chunker, embedder
- `tests/test_store.py` — integration tests for persistent + session stores (real ChromaDB + SQLite, **not mocks**)
- `tests/test_agent.py` — end-to-end: Mode 1 (ingest → query) and Mode 2 (session chunks → query)
- `tests/test_scraper.py` — unit test scraper URL parsing with mocked HTTP

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Vector store | ChromaDB (default) | Zero-config embedded; interface abstracted for swap |
| Session store | In-memory list of Chunk objects | No persistence needed; same retrieval interface |
| Mode 2 isolation | `persist=False` flag on `ingest_pdf` | Single code path for parsing/chunking; only write is conditional |
| Missing fund (Mode 1) | `FundNotFoundError` + UI upload prompt | Clean signal; upload triggers `persist=True` ingest |
| Mode 2 fund gap | No error — use what's in session chunks | User explicitly provided the PDF; we trust it |
| Embeddings | `sentence-transformers` all-MiniLM-L6-v2 | Local, free, fast; 384-dim vectors |
| LLM | OpenAI gpt-4o-mini | Good quality/cost ratio for synthesis |

## New Dependencies to Add to requirements.txt
- `requests` — HTTP for scraper
- `beautifulsoup4` — HTML parsing for scraper

## Critical Files

| File | Purpose |
|---|---|
| `store/base.py` | Abstract interfaces — pluggability contract |
| `store/vector_store.py` | ChromaDB persistent implementation |
| `store/session_store.py` | In-memory implementation for Mode 2 |
| `store/structured_store.py` | SQLite implementation |
| `ingestion/pipeline.py` | Core ingestion: `ingest_pdf(path, persist=True/False)` |
| `scraper/amfi_scraper.py` | AMFI website scraping |
| `agent/agent.py` | `ask(question, session_chunks=None) -> AgentResponse` |
| `agent/retriever.py` | Dual-track retrieval + `FundNotFoundError` |
| `chat/app.py` | Streamlit UI with Mode 1 / Mode 2 toggle |
| `docs/architecture.md` | Architecture diagram |

## Environment

- `OPENAI_API_KEY` must be set
- `pip install -r requirements.txt`

## Verification

1. Run `python -m scraper.runner` — downloads factsheet PDFs into `data/raw_pdfs/`
2. Run `python -m ingestion.pipeline` — populates ChromaDB + SQLite
3. **Mode 1:** Run `streamlit run chat/app.py`, ask about a known fund → gets answer
4. **Mode 1 missing fund:** Ask about an unknown fund → prompted to upload → upload → re-query succeeds → fund now in KB
5. **Mode 2:** Switch to Direct Upload mode → upload PDF → ask question → gets answer → verify PDF not in KB after session
6. Run `python -m pytest tests/` — all tests green
