# =============================================================================
# ingestion/chunker.py — sliding-window chunker
#
# WHY THIS FILE EXISTS:
# Splits a page's text into overlapping ~500-word windows, each wrapped as a
# Chunk (embedding left empty — the embedder fills it next). Graduated from the
# notebook cell `chunk_text()`.
# =============================================================================


# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------
# You need:
#   - `Path`               → from pathlib, to build chunk_id from the filename stem
#   - `Chunk`              → from store.base, the unit you produce
#
# Write your imports here:
from pathlib import Path
from store.base import Chunk


# -----------------------------------------------------------------------------
# chunk_text(text, page, fund_name, source_file, chunk_size=500, overlap=50)
#   -> list[Chunk]
#
# WHY: One page can be thousands of words — too big for a single embedding.
# We slide a fixed-size window with overlap so context isn't cut mid-thought.
#
# STEPS (your working notebook version):
#   1. words = text.split()
#   2. step = chunk_size - overlap
#   3. for i, start in enumerate(range(0, len(words), step)):
#        - window  = words[start : start + chunk_size]
#        - content = " ".join(window)
#        - chunk_id = f"{Path(source_file).stem}_p{page}_c{i}"
#        - build a Chunk(embedding=[], metadata={}, ...) and append
#   4. return the list of chunks
#
# EDGE CASE: if text is empty, words is [] and range(...) yields nothing →
# returns [] naturally. Good — a blank page produces no chunks.
#
# Write chunk_text() here:

def chunk_text(text: str, page: int, fund_name: str, 
               source_file: str, chunk_size: int = 500, 
               overlap: int = 50) -> list[Chunk]:
    # Step 1: split text into words
    split_text = text.split()
        
    start = 0
    end = start + chunk_size
    chunk_list = []

    while start < len(split_text):
    # Step 2:
        window = split_text[start:end]

    # Step 3: for each window, join words back into a string
        content = " ".join(window)
    # Step 4: create a chunk_id: f"{fund_name}_p{page}_{index}"
        chunk_id = f"{Path(source_file).stem}_p{page}_c{start+1}"
    # Step 5: create a Chunk object (leave embedding as empty list [] for now)
        chunk = Chunk(chunk_id=chunk_id,
                      text = content,
                      embedding=[],
                      fund_name=fund_name,
                      source_file=source_file,
                      page=page,
                      metadata={})
        
        chunk_list.append(chunk)
        
        start += chunk_size - overlap
        end = start + chunk_size

    # Step 6: return the list of Chunks
    return chunk_list