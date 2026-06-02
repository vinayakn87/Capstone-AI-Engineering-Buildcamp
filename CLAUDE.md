# Mutual Fund Factsheet RAG Agent

## Project Overview
A retrieval-augmented generation (RAG) system that ingests mutual fund factsheet PDFs and answers user questions through a Streamlit chat interface, backed by an OpenAI model as the LLM.

## Architecture

```
data/
  raw_pdfs/       – original factsheet PDFs (not committed)
  processed/      – parsed chunks + metadata JSON (not committed)

ingestion/        – PDF parsing pipeline (pdfplumber)
store/            – ChromaDB vector store + SQLite structured store (sqlite-utils)
agent/            – orchestration: retrieval → context assembly → OpenAI synthesis
chat/             – Streamlit chat UI
tests/            – unit and integration tests
```

## Key Design Decisions
- **PDF parsing**: `pdfplumber` for text and table extraction from factsheet PDFs
- **Embeddings**: `sentence-transformers` (local, no API cost for indexing)
- **Vector store**: ChromaDB (local persistent)
- **Store abstraction**: `store/base.py` ABC — swap to Qdrant/Pinecone by implementing the same interface
- **Structured store**: SQLite via `sqlite-utils` (fund metadata, NAV, expense ratios)
- **LLM**: OpenAI (via `openai` SDK) for synthesis
- **UI**: Streamlit
- **Chunking**: Sliding window 500 tokens / 50 token overlap — standard RAG chunk size
- **Missing fund handling**: `FundNotFoundError` + UI prompt — agent signals gap; UI collects upload; ingestion runs inline
- **Scraper**: `requests` + `BeautifulSoup` — lightweight; AMFI site is static HTML

## Data Flow
1. Drop PDFs into `data/raw_pdfs/`
2. Run `ingestion/` pipeline → chunks land in `data/processed/` + indexed into ChromaDB + SQLite
3. User asks a question in Streamlit → `agent/` retrieves relevant chunks → OpenAI synthesizes answer

## Environment Variables
- `OPENAI_API_KEY` – required for the OpenAI synthesis step

## Running Locally
```bash
pip install -r requirements.txt
# ingest PDFs
python -m ingestion.pipeline
# launch chat
streamlit run chat/app.py
```

## Working Agreement
- Always ask for explicit approval before calling ExitPlanMode
- Always ask for explicit approval before writing any code files
- Do not run or execute code unless the user explicitly asks. If verification is needed, say "you can verify by running X" instead
- This is a learning exercise: act as a guide, not a code generator
  - Explain what needs to be built and why
  - Describe the interface/contract the code should satisfy
  - Give hints, pseudocode, or point to relevant docs/patterns when the user is stuck
  - Review code the user writes and give feedback
  - Only write code if the user is truly stuck and explicitly asks
