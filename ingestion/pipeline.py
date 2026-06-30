# =============================================================================
# ingestion/pipeline.py — orchestration: parse → chunk → embed → store
#
# WHY THIS FILE EXISTS:
# Ties the three stages together into one callable. This is the entry point the
# Streamlit upload flow and the CLI both use.
#
#   ingest_pdf(path, store, embedder, persist=True) -> list[Chunk]
#     - Mode 1 (persist=True):  write chunks into the vector store (knowledge base)
#     - Mode 2 (persist=False): return chunks WITHOUT writing (session-only)
#   Always returns the chunks, so the caller can use them either way.
# =============================================================================


# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------
# You need:
#   - `Path`                    → from pathlib, to derive fund_name + source_file
#   - `parse_pdf`               → from ingestion.pdf_parser
#   - `chunk_text`              → from ingestion.chunker
#   - `Embedder`                → from ingestion.embedder
#   - `ChromaVectorStore`       → from store.vector_store (for the CLI)
#   - `Chunk`                   → from store.base (return type)
#
# Write your imports here:
from pathlib import Path
from ingestion.pdf_parser import parse_pdf
from ingestion.chunker import chunk_text
from store.base import Chunk
from store.vector_store import ChromaVectorStore
from ingestion.embedder import Embedder


# -----------------------------------------------------------------------------
# _fund_name_from_path(path: str) -> str
#
# WHY: Each chunk needs a fund_name for filtering/has_fund. Proper extraction
# from the PDF cover page comes later — for now derive a readable name from the
# filename so the pipeline is end-to-end runnable.
#
# STEPS (simple version):
#   1. stem = Path(path).stem        e.g. "KotakMFFactsheetMay2026"
#   2. return stem                   (good enough for now; refine later)
#
# Write _fund_name_from_path() here:
def _fund_name_from_path(path):
    filename = Path(path).stem
    return filename


# -----------------------------------------------------------------------------
# ingest_pdf(path, store, embedder, persist=True) -> list[Chunk]
#
# WHY: The core ingestion call. One code path for both modes; only the final
# write is conditional on `persist`. Seeding function for creating RAG database
#
# STEPS:
#   1. fund_name   = _fund_name_from_path(path)
#      source_file = Path(path).name        ← filename with extension
#   2. pages = parse_pdf(path)              ← list of (page_number, content)
#   3. all_chunks = []
#      for (page_number, content) in pages:
#          chunks = chunk_text(content, page=page_number,
#                              fund_name=fund_name, source_file=source_file)
#          all_chunks.extend(chunks)
#   4. embedder.embed_chunks(all_chunks)    ← fills each chunk.embedding in place
#   5. if persist:
#          store.add(all_chunks)            ← write to the knowledge base
#   6. return all_chunks
#
# Write ingest_pdf() here:

def ingest_pdf(path, store, embedder, persist=True) -> list[Chunk]:
    
    fund_name = _fund_name_from_path(path)
    source_file = Path(path).name
    
    output_parsed = parse_pdf(path)
    
    all_chunks = []
    for (page, scheme_name, content) in output_parsed:
        chunk_list = chunk_text(
            text = content, 
            page = page, 
            fund_name = scheme_name or fund_name, 
            source_file = source_file
            )
        all_chunks.extend(chunk_list)
    
    embedder.embed_chunks(all_chunks)
  
    if persist:
        store.add(all_chunks)
    return all_chunks

# -----------------------------------------------------------------------------
# CLI: python -m ingestion.pipeline
#
# WHY: Batch-ingest every PDF in data/raw_pdfs/ into the persistent store.
#
# STEPS (inside `if __name__ == "__main__":`):
#   1. store    = ChromaVectorStore()           ← default persist_dir
#      embedder = Embedder()                     ← loads model once
#   2. for pdf in Path("data/raw_pdfs").glob("*.pdf"):
#          chunks = ingest_pdf(str(pdf), store, embedder, persist=True)
#          print(f"{pdf.name}: {len(chunks)} chunks ingested")
#   3. print(f"Total chunks in store: {store.count()}")
#
# Write the __main__ block here:

def main():
    embedder = Embedder()
    store = ChromaVectorStore()

    for pdf in Path("data/raw_pdfs").glob("*.pdf"):
        chunks = ingest_pdf(str(pdf), store, embedder, persist=True)
        print(f"{pdf.name}: {len(chunks)} chunks ingested")

    print(f"Total chunks in store: {store.count()}")


if __name__ == "__main__":
    main()
