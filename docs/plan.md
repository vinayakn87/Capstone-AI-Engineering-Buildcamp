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
  │  retriever.py  (thin, pure vector search)        │
  │  • embeds question → searches ChromaDB (top-k)   │
  │  • optionally merges session chunks (Mode 2)     │
  │  • NO has_fund gate, NO FundNotFoundError        │
  │                                                  │
  │  synthesizer.py → OpenAI gpt-4o-mini             │
  │  • grounding + fund disambiguation               │
  │  • soft absence ("share what we have" + upload)  │
  │  • citation table of funds/pages used            │
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
  │  │  → Answer + citation table of funds used    │ │
  │  │  → Unsatisfied / data absent? Upload PDF    │ │
  │  │    → ingest into KB → re-query              │ │
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
> **DESIGN CHANGE (2026-06-30) — pure-RAG retrieval, no `has_fund` gate.** See the
> "Design Change Log" at the bottom for what changed and why. Summary: the retriever
> is now *thin* — embed → vector search → return chunks. It does **not** call
> `has_fund()` and does **not** raise `FundNotFoundError`. Fund disambiguation and
> "I don't have this" are handled **downstream in the synthesizer prompt**.
> `has_fund()` stays in the store interface, unused, reserved for a later agentic phase.

- `agent/retriever.py`:
  - `retrieve(question, session_chunks=None, top_k=5) -> list[Chunk]` — embeds the
    question → `store.search()` → returns the top-k chunks (each carrying
    `fund_name`/`source_file`/`page`). If `session_chunks` provided (Mode 2), rank them
    in-memory by cosine similarity to the query and **merge** with store results, then
    re-sort and trim to `top_k`. **No `has_fund` gate, no `FundNotFoundError`.**
- `agent/synthesizer.py` — context + question → OpenAI → answer. The real work now lives here:
  - **Grounding:** answer ONLY from the retrieved chunks; never invent figures.
  - **Disambiguation:** chunks may span multiple funds; identify which fund(s) the
    context covers, answer per-fund when several are plausible, ask a narrowing
    follow-up when the question is ambiguous.
  - **Absence ("share what we have"):** when context doesn't cover the question, answer
    with what it *does* have and invite the user to upload the relevant factsheet (soft,
    prompt-driven — replaces the old deterministic `FundNotFound` trigger).
  - **Citation table:** every response includes a table of the actual
    `fund_name` + `source_file`/`page` the answer drew from.
- `agent/agent.py`:
  - `ask(question, session_chunks=None) -> AgentResponse`
  - `AgentResponse`: `Answer(text, sources)` — `sources` feeds the citation table.
    (No `FundNotFound` variant in this phase; absence is conveyed inside `Answer`.)

### Step 5 — Chat UI
- `chat/app.py` — Streamlit app with two modes toggled in sidebar:

  **Mode 1 (Q&A):**
  - Chat input → `agent.ask(question, session_chunks=None)` → `Answer` (with the
    citation table of funds used)
  - The upload affordance is **always available** (not gated by a `FundNotFound`
    signal anymore). When the answer indicates it lacks the data — or the user is
    simply unsatisfied — the user uploads via `st.file_uploader` →
    `ingest_pdf(path, persist=True)` → re-run query.

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
| Fund detection | **None in retriever** (pure vector search) | Pure RAG, no intelligence layer between query and retriever; ambiguity/absence handled in synthesizer prompt |
| Missing fund (Mode 1) | **Soft, prompt-driven** — synthesizer says what it has + invites upload | `has_fund`/`FundNotFoundError` too restrictive for pure RAG; gate deferred to agentic phase |
| `has_fund()` | Kept in store interface, **unused for now** | Tested + reserved for the agentic phase; bypassing costs nothing, deleting throws away a tested gate |
| Surfacing funds used | Citation table in every response | User always sees which `fund_name`(s)/pages an answer drew from → guards against silent cross-fund answers |
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
| `agent/retriever.py` | Thin pure-vector retrieval (+ Mode 2 session merge); **no gate** |
| `chat/app.py` | Streamlit UI with Mode 1 / Mode 2 toggle |
| `docs/architecture.md` | Architecture diagram |

## Environment

- `OPENAI_API_KEY` must be set
- `pip install -r requirements.txt`

## Verification

1. Run `python -m scraper.runner` — downloads factsheet PDFs into `data/raw_pdfs/`
2. Run `python -m ingestion.pipeline` — populates ChromaDB + SQLite
3. **Mode 1:** Run `streamlit run chat/app.py`, ask about a known fund → gets answer
4. **Mode 1 missing fund:** Ask about an unknown fund → synthesizer answers that it lacks that data (and shows the citation table of what it *did* find) → user uploads → `ingest_pdf(persist=True)` → re-query succeeds → fund now in KB
5. **Mode 2:** Switch to Direct Upload mode → upload PDF → ask question → gets answer → verify PDF not in KB after session
6. Run `python -m pytest tests/` — all tests green

## Design Change Log

### 2026-06-30 — Drop `has_fund` gate; handle ambiguity/absence in the synthesizer
**What changed.** The retriever no longer detects the fund or gates on `has_fund()`,
and `FundNotFoundError` is no longer raised in the Q&A path. Retrieval is now pure
vector search (embed → `search` → return chunks). All fund disambiguation, "answer
per-fund", "ask a narrowing follow-up", and "I don't have this → upload" behaviour
moved **into the synthesizer prompt**. Every response carries a **citation table**
of the actual funds/pages used. `has_fund()` remains implemented and tested in the
store but is **not called** anywhere in this phase.

**Why.**
- *Faithful to pure RAG.* The original intent was no intelligence layer between query
  and retriever. A fund-detection step (regex/lexical/LLM) reintroduced exactly that.
  Pure vector search + LLM synthesis keeps the retriever dumb.
- *`has_fund` was too restrictive.* It needs an exact `fund_name` string, but users
  type free text and shorthand; deriving that string cleanly was the hard, fragile
  part (every option had a threshold or dependency). Skipping it removes the brittle step.
- *The LLM is the right tool for ambiguity.* Given chunks spanning several funds,
  "which fund did you mean / here's per-fund answers" is squarely an LLM strength.

**Accepted tradeoffs (eyes open).**
- *Missing-fund detection goes from deterministic → probabilistic.* Vector search
  always returns top-k, even for an absent fund. The synthesizer must notice the
  mismatch from chunk metadata and say so. Mitigation: pass `fund_name`/`page` into
  the context **and** show the citation table so the user always sees which funds the
  answer used; if unsatisfied, they upload.
- *Cross-fund contamination risk on numeric questions.* Unfiltered top-k can mix
  funds. Mitigation: surface per-chunk `fund_name` in the context and require the
  citation table — the load-bearing safeguard against silently answering with the
  wrong fund's figures.

**Reversibility.** `has_fund()` is untouched in `store/`. Re-introducing a
deterministic gate (or a hybrid lexical pre-filter) in a later **agentic phase** is a
small, additive change — nothing here forecloses it.
