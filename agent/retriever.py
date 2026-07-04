# =============================================================================
# agent/retriever.py — pure-vector retrieval (the "R" in RAG)
#
# WHY THIS FILE EXISTS:
# Turns a user's free-text question into the most relevant Chunks. This is the
# bridge between the user and the synthesizer: question → embedding → vector
# search → top-k chunks (each carrying fund_name / source_file / page metadata,
# which the synthesizer needs for grounding + the citation table).
#
# DESIGN (2026-06-30 — see the Design Change Log in the plan):
#   - PURE vector search. NO fund detection, NO has_fund() gate, NO
#     FundNotFoundError. The retriever is deliberately "dumb" — there is no
#     intelligence layer between the query and the store.
#   - Ambiguity (chunks span several funds) and absence ("I don't have this")
#     are handled DOWNSTREAM in the synthesizer prompt, not here.
#   - has_fund() stays in the store, unused, reserved for a later agentic phase.
#
# WHY A CLASS (not a plain function):
# It holds reused state — the `store` and the `embedder` — set up once and used
# on every query. Same pattern as Embedder / ChromaVectorStore. (Smell test from
# python_concepts.md: expensive/reused state → class.)
# =============================================================================


# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------
# You need:
#   - `numpy`               → cosine similarity for the Mode-2 re-rank (it's already
#                             a transitive dep via sentence-transformers). math also
#                             works if you'd rather avoid numpy.
#   - `Chunk`               → from store.base (what we return)
#   - `VectorStore`         → from store.base (type hint for the store we read)
#   - `Embedder`            → from ingestion.embedder (embeds the question)
#
# Write your imports here:
import numpy as np
from store.base import Chunk
from store.base import VectorStore
from ingestion.embedder import Embedder

# -----------------------------------------------------------------------------
# _cosine_similarity(a: list[float], b: list[float]) -> float
#
# WHY: For Mode 2 we must rank session chunks (held in memory, never in the
# store) against the query, then merge them with the store's hits into ONE
# ranking. Both the query and every chunk already have an embedding, so we just
# score them ourselves. cosine = how aligned two vectors are, in [-1, 1];
# higher = more similar.
#
# WHY WE CAN DO THIS: store.search() returns each Chunk WITH its `embedding`
# (include=["embeddings"]), and session chunks were embedded at ingest time. So
# every candidate already carries the vector we need — no re-embedding.
#
# STEPS:
#   1. Convert a, b to numpy arrays.
#   2. cosine = dot(a, b) / (norm(a) * norm(b))
#   3. Guard a zero-norm vector (return 0.0) so you never divide by zero.
#   4. Return the float.
#
# Write _cosine_similarity() here:

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    a = np.array(a)
    b = np.array(b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a != 0 and norm_b != 0:
        cosine = np.dot(a, b)/(norm_a * norm_b)
        return float(cosine)
    
    return 0.0





# -----------------------------------------------------------------------------
# CLASS: Retriever
#
# WHY: Bundles the store + embedder so a single instance answers many queries
# without reloading or re-wiring anything.
# =============================================================================

class Retriever:

# -----------------------------------------------------------------------------
# __init__(self, store: VectorStore, embedder: Embedder)
#
# WHY: Construct once, reuse for every question. Both are injected (not created
# inside) so tests can pass a tmp_path store and the shared session embedder, and
# so the SAME embedder is reused across ingestion + retrieval (one model load).
#
# STEPS:
#   1. self._store    = store
#   2. self._embedder = embedder
#
# Write __init__ here:

    def __init__(self, store: VectorStore, embedder: Embedder):
        self._embedder = embedder
        self._store = store

# -----------------------------------------------------------------------------
# retrieve(self, question: str, session_chunks: list[Chunk] | None = None,
#          top_k: int = 5) -> list[Chunk]
#
# WHY: The one public method. Embeds the question, pulls the best chunks from the
# store, and — in Mode 2 — folds in the session's own chunks so an uploaded PDF
# that was never persisted can still be retrieved this session.
#
# CONTRACT:
#   - Returns a list of up to top_k Chunks, most-relevant first.
#   - Each returned Chunk carries fund_name / source_file / page (the synthesizer
#     relies on this for grounding + the citation table).
#   - Empty store AND no session_chunks → returns [] (the store already guards the
#     empty case). The synthesizer turns [] into "I don't have data on this".
#   - NEVER raises for a missing fund. Absence is the synthesizer's job.
#
# STEPS:
#   1. q = self._embedder.embed_one(question)          ← one 384-d query vector
#   2. store_hits = self._store.search(q, top_k=top_k) ← top_k from the persistent KB
#   3. If no session_chunks: return store_hits (nothing to merge — Mode 1 path).
#   4. MODE 2 MERGE (session_chunks present):
#        a. candidates = store_hits + session_chunks   ← one combined pool
#        b. score each candidate: _cosine_similarity(q, candidate.embedding)
#        c. sort candidates by score, descending
#        d. return the first top_k
#      (Re-scoring the store hits too keeps both sources on ONE comparable scale —
#       store_hits already have their embeddings from include=["embeddings"].)
#
# NOTE (de-dup, optional): if a Mode-2 user uploads a PDF that's ALSO already in
# the KB, the same text could appear twice (once from each source). Deterministic
# chunk_ids make this easy to drop — keep the first occurrence per chunk_id. Skip
# for v1 unless tests show duplicates.
#
# Write retrieve() here:

    def retrieve(
            self, 
            question: str, 
            session_chunks: list[Chunk] | None = None,
            top_k: int = 5
        ) -> list[Chunk]:

        q = self._embedder.embed_one(question)
        store_hits = self._store.search(q, top_k=top_k)
        if session_chunks is None:
            return store_hits
        
        candidates = session_chunks + store_hits
        candidates.sort(key=lambda chunk:_cosine_similarity(q, chunk.embedding), reverse=True)
        
        return candidates[:top_k]