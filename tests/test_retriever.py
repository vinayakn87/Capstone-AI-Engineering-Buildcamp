# =============================================================================
# tests/test_retriever.py — UNIT tests for the retriever (isolated)
#
# SCOPE: tests the retriever's OWN logic in isolation — NO real embedder (no
# model load), NO real ChromaDB. We inject a FAKE embedder + FAKE store so we
# CONTROL the vectors and can assert the Mode-2 ranking deterministically.
# (The real-stack version lives in the later integration test.)
#
# Two testable surfaces:
#   1. _cosine_similarity(a, b) → PURE function. Test directly.
#   2. Retriever.retrieve()     → Mode-1 passthrough vs Mode-2 merge+rank+trim,
#                                 driven by fakes with known embeddings.
#
# Run with: python -m pytest tests/test_retriever.py -v
# =============================================================================


# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------
# You need:
#   - `pytest`                      → for approx() on float comparisons
#   - `Chunk`                       → from store.base
#   - `Retriever`, `_cosine_similarity`  → from agent.retriever
#
# Write your imports here:
import pytest
from store.base import Chunk
from agent.retriever import Retriever, _cosine_similarity


# -----------------------------------------------------------------------------
# HELPER: make_chunk(chunk_id, embedding, text="t", fund_name="Fund X", ...)
#
# WHY: retrieve() ranks chunks by their embedding, so tests need chunks with
# CHOSEN embeddings. A factory keeps the vectors front-and-centre.
#
# STEPS:
#   Define make_chunk(chunk_id, embedding, text="t", fund_name="Fund X",
#       source_file="f.pdf", page=1) -> Chunk returning a fully-formed Chunk with
#   that chunk_id + embedding (metadata={}).
#
# Write make_chunk() here:
def make_chunk(
        chunk_id, 
        embedding, 
        text="t", 
        fund_name="Fund X",
        source_file="f.pdf", 
        page=1
    ) -> Chunk:

    return Chunk(
            chunk_id=chunk_id, 
            text=text, 
            embedding=embedding, 
            fund_name=fund_name, 
            source_file=source_file,
            page=page,
            metadata={}
        )
    


# -----------------------------------------------------------------------------
# FAKES: a fake embedder and a fake store to isolate the retriever
#
# WHY: retrieve() only calls two things on its collaborators:
#     self._embedder.embed_one(question) -> list[float]
#     self._store.search(query_embedding, top_k=...) -> list[Chunk]
# So the fakes only need those two methods — nothing else.
#
# STEPS:
#   class _FakeEmbedder:
#       __init__(self, vector)              ← the query vector to always return
#       embed_one(self, question) -> list[float]:  return self._vector
#         (question is ignored — we control the query vector directly)
#
#   class _FakeStore:
#       __init__(self, hits)                ← the list[Chunk] search should return
#       search(self, query_embedding, top_k=5) -> list[Chunk]:
#           return self._hits[:top_k]       ← honour top_k like the real store
#
# Write the fakes here:

class _FakeEmbedder:

    def __init__(self, vector):
        self._vector = vector

    def embed_one(self, question) -> list[float]:
        return self._vector

class _FakeStore:

    def __init__(self, hits):
        self._hits = hits

    def search(self, query_embedding, top_k=5) -> list[Chunk]:
        return self._hits[:top_k]

# =============================================================================
# TESTS — _cosine_similarity (pure)
# =============================================================================

# -----------------------------------------------------------------------------
# TEST — identical vectors → 1.0 ; orthogonal → 0.0 ; zero vector → 0.0 (guard)
#
# WHY: locks the math + the divide-by-zero guard. Use pytest.approx for floats.
#
# STEPS (one behaviour each — feel free to split into 3 tests):
#   1. assert _cosine_similarity([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
#   2. assert _cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
#   3. assert _cosine_similarity([0, 0], [1, 1]) == 0.0     ← zero-norm → guard
#
# Write test_cosine_similarity() here:

def test_cosine_similarity():
    assert _cosine_similarity([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert _cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert _cosine_similarity([0, 0], [1, 1]) == 0.0


# =============================================================================
# TESTS — Retriever.retrieve
# =============================================================================

# -----------------------------------------------------------------------------
# TEST — Mode 1 (no session chunks): returns the store hits, untouched
#
# WHY: with session_chunks=None the retriever must just pass the store's results
# straight through (no re-ranking, no merge).
#
# STEPS:
#   1. hits = [make_chunk("s1", [1, 0]), make_chunk("s2", [0, 1])]
#   2. retr = Retriever(store=_FakeStore(hits), embedder=_FakeEmbedder([1, 0]))
#   3. out = retr.retrieve("any question")            ← session_chunks defaults None
#   4. assert out == hits                             ← same objects, same order
#
# Write test_retrieve_mode1_passthrough() here:

def test_retrieve_mode1_passthrough():
    hits = [
            make_chunk("s1", [1, 0]), 
            make_chunk("s2", [0, 1])
            ]
    retr = Retriever(store= _FakeStore(hits), embedder= _FakeEmbedder([1,0]))
    out = retr.retrieve("any question")
    assert out == hits

# -----------------------------------------------------------------------------
# TEST — Mode 2 merge + rank: session chunk closest to the query ranks first
#
# WHY: the whole point of Mode 2 — session chunks and store hits compete on ONE
# cosine scale, most-similar first.
#
# SETUP (query = [1, 0]):
#   - store hit    S: embedding [0, 1]     → cosine 0.0  (farthest)
#   - session A:      embedding [1, 0]      → cosine 1.0  (closest)
#   - session B:      embedding [0.5, 0.5]  → cosine ~0.707 (middle)
#
# STEPS:
#   1. retr = Retriever(store=_FakeStore([make_chunk("S", [0, 1])]),
#                       embedder=_FakeEmbedder([1, 0]))
#   2. session = [make_chunk("A", [1, 0]), make_chunk("B", [0.5, 0.5])]
#   3. out = retr.retrieve("q", session_chunks=session)
#   4. assert [c.chunk_id for c in out] == ["A", "B", "S"]   ← ranked by cosine
#      (and assert the store hit S is present → both sources merged)
#
# Write test_retrieve_mode2_ranks_by_similarity() here:

def test_retrieve_mode2_ranks_by_similarity():
    retr = Retriever(store=_FakeStore([make_chunk("S", [0, 1])]),
                     embedder=_FakeEmbedder([1, 0]))
    session = [make_chunk("A", [1, 0]), make_chunk("B", [0.5, 0.5])]
    out = retr.retrieve("q", session_chunks=session)
    assert [c.chunk_id for c in out] == ["A", "B", "S"] 

# -----------------------------------------------------------------------------
# TEST — top_k trims the merged result
#
# WHY: retrieve must never return more than top_k, even after merging.
#
# STEPS:
#   1. store returns [make_chunk("S", [0, 1])]; session = [A [1,0], B [0.5,0.5]]
#   2. out = retr.retrieve("q", session_chunks=session, top_k=2)
#   3. assert len(out) == 2
#      assert [c.chunk_id for c in out] == ["A", "B"]   ← top-2 by similarity
#
# Write test_retrieve_respects_top_k() here:

def test_retrieve_respects_top_k():
    store_hits = [make_chunk("S", [0, 1])]
    session = [
                make_chunk("A", [1,0]), 
                make_chunk("B", [0.5,0.5])
              ]
    retr = Retriever(store=_FakeStore(store_hits), embedder=_FakeEmbedder([1,0]))
    out = retr.retrieve("my question", session_chunks=session, top_k=2)
    assert [c.chunk_id for c in out] == ["A", "B"]

# -----------------------------------------------------------------------------
# TEST — empty store + no session → []
#
# WHY: cold-start / nothing-found must return [] cleanly (the synthesizer turns
# [] into "I don't have data"). No crash.
#
# STEPS:
#   1. retr = Retriever(store=_FakeStore([]), embedder=_FakeEmbedder([1, 0]))
#   2. assert retr.retrieve("q") == []
#
# Write test_retrieve_empty_store() here:

def test_retrieve_empty_store():
    retr = Retriever(store=_FakeStore([]), embedder=_FakeEmbedder([]))
    assert retr.retrieve("my question") == []