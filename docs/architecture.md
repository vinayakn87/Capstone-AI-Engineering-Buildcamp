# Mutual Fund Factsheet RAG Agent — Architecture

## Component Diagram

```mermaid
graph TB
    subgraph SCRAPER["🔍 Scraper (scraper/)"]
        SC["amfi_scraper.py\nrequests + BeautifulSoup"]
        DL["downloader.py\nHTTP PDF download"]
        SC --> DL
    end

    subgraph INGESTION["⚙️ Ingestion Pipeline (ingestion/)"]
        PARSER["pdf_parser.py\npdfplumber\ntext + tables"]
        CHUNKER["chunker.py\nsliding window\n~500 tokens / 50 overlap"]
        EMBEDDER["embedder.py\nsentence-transformers\nall-MiniLM-L6-v2 (384-dim)"]
        PARSER --> CHUNKER --> EMBEDDER
    end

    subgraph STORES["💾 Store Layer (store/)"]
        BASE["base.py\nVectorStore ABC\nStructuredStore ABC"]
        subgraph PERSISTENT["Persistent (Knowledge Base)"]
            CHROMA["vector_store.py\nChromaDB\n[swappable → Qdrant]"]
            SQLITE["structured_store.py\nSQLite via sqlite-utils\nfunds · nav · expense_ratios"]
        end
        subgraph SESSION["Session-Only (Mode 2)"]
            INMEM["session_store.py\nIn-memory VectorStore\ndiscarded on session end"]
        end
        BASE --> CHROMA
        BASE --> SQLITE
        BASE --> INMEM
    end

    subgraph AGENT["🤖 Agent (agent/)"]
        RETRIEVER["retriever.py\nsemantic search + structured query\nmerge & rank context\nFundNotFoundError"]
        SYNTHESIZER["synthesizer.py\nOpenAI gpt-4o-mini\ncontext → answer"]
        RETRIEVER --> SYNTHESIZER
    end

    subgraph UI["💬 Chat UI (chat/app.py)"]
        MODE1["Mode 1: Q&A\nAsk question → KB lookup\nFundNotFound → upload prompt\nUpload → ingest into KB"]
        MODE2["Mode 2: Direct Upload\nUpload PDF → session-only\nAsk question → session + KB\nNever written to KB"]
    end

    DL -->|"PDFs → data/raw_pdfs/"| PARSER
    EMBEDDER -->|"persist=True"| CHROMA
    PARSER -->|"fund metadata"| SQLITE
    EMBEDDER -->|"persist=False\n(Mode 2)"| INMEM

    CHROMA --> RETRIEVER
    SQLITE --> RETRIEVER
    INMEM -->|"session_chunks"| RETRIEVER
    SYNTHESIZER --> MODE1
    SYNTHESIZER --> MODE2
```

## ASCII Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│               MUTUAL FUND FACTSHEET RAG AGENT                            │
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
  │                                                  │
  │  ingest_pdf(path, persist=True)  → KB            │
  │  ingest_pdf(path, persist=False) → session only  │
  └──────────────────────┬───────────────────────────┘
           ┌─────────────┴──────────────┐
           ▼                            ▼
  ┌────────────────────┐    ┌──────────────────────────────┐
  │  STRUCTURED STORE  │    │   VECTOR STORE               │
  │  SQLite            │    │   ChromaDB (persistent KB)   │
  │  (sqlite-utils)    │    │   [abstracted → swap Qdrant] │
  │  funds / nav /     │    │                              │
  │  expense_ratios    │    │   session_store.py           │
  └─────────┬──────────┘    │   (in-memory, Mode 2 only)  │
            │               └──────────────┬───────────────┘
            │                              │
            └──────────────┬───────────────┘
                           │
  ┌────────────────────────▼─────────────────────────┐
  │              AGENT (agent/)                       │
  │                                                  │
  │  retriever.py                                    │
  │  • searches persistent store (ChromaDB + SQLite) │
  │  • merges session chunks if provided (Mode 2)    │
  │  • raises FundNotFoundError if fund absent        │
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
  │  │  MODE 1: Q&A (knowledge-base driven)        │ │
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

## Data Flow Summary

| Step | Component | Input | Output | Library |
|------|-----------|-------|--------|---------|
| 1 | Scraper | AMFI website | PDFs in `data/raw_pdfs/` | requests, BeautifulSoup |
| 2 | PDF Parser | PDF file | Text chunks + metadata | pdfplumber |
| 3 | Chunker | Raw text | ~500-token chunks | custom |
| 4 | Embedder | Text chunks | 384-dim vectors | sentence-transformers |
| 5 | Vector Store | Embeddings | ChromaDB collection | chromadb |
| 6 | Structured Store | Fund metadata | SQLite tables | sqlite-utils |
| 7 | Retriever | User question | Ranked context chunks | sentence-transformers + chromadb |
| 8 | Synthesizer | Context + question | Natural-language answer | openai (gpt-4o-mini) |
| 9 | Chat UI | Answer / FundNotFound | Rendered chat message | streamlit |

## Store Abstraction (Pluggability)

`store/base.py` defines abstract interfaces. To swap ChromaDB for Qdrant:
1. Implement `QdrantVectorStore(VectorStore)` in `store/qdrant_store.py`
2. Change the import in `ingestion/pipeline.py` and `agent/retriever.py`
3. Nothing else changes

## Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `OPENAI_API_KEY` | Yes | OpenAI synthesis (gpt-4o-mini) |
