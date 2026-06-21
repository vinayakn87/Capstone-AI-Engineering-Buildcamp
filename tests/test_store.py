# =============================================================================
# tests/test_store.py — integration tests for the vector store layer
#
# TESTING PROTOCOL (from CLAUDE.md):
#   - Store tests use a REAL ChromaDB pointed at a temp dir — no mocks.
#   - Each test file covers: happy path, empty/missing input, and one edge case.
#   - Run with: python -m pytest tests/test_store.py -v
# =============================================================================


# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------
# You need:
#   - `pytest`                       → for fixtures + running the tests
#   - `ChromaVectorStore`            → from store.vector_store (the class under test)
#   - `Chunk`                        → from store.base (to build test data)
#
# Write your imports here:
import pytest
from store.vector_store import ChromaVectorStore
from store.base import Chunk


# -----------------------------------------------------------------------------
# HELPER: make_chunk(...)
#
# WHY: Every test needs Chunk objects. A small factory keeps tests readable and
# avoids repeating all seven fields each time. Give it sensible defaults so a
# test only overrides what it cares about (e.g. just fund_name, or just text).
#
# CONTRACT:
#   make_chunk(chunk_id, text="...", fund_name="Test Fund", page=1) -> Chunk
#   - embedding: a fixed-length list of floats. all-MiniLM-L6-v2 is 384-dim,
#     but for a *store* test the numbers are arbitrary — what matters is that
#     every embedding in one collection has the SAME length. Use e.g. [0.1]*384
#     or vary one value so different chunks aren't identical vectors.
#
# Write make_chunk() here:

def make_chunk(
    chunk_id: str,
    text: str = "sample chunk text",
    fund_name: str = "Test Fund",
    page: int = 1,
    embedding: list[float] | None = None,
) -> Chunk:
    # Default embedding: 384-dim (matches all-MiniLM-L6-v2) but values are
    # arbitrary for a store test. Vary the first value by chunk_id length so
    # different chunks aren't identical vectors, while staying deterministic.
    if embedding is None:
        embedding = [0.1] * 384

    return Chunk(
        chunk_id=chunk_id,
        text=text,
        embedding=embedding,
        fund_name=fund_name,
        source_file="test_factsheet.pdf",
        page=page,
        metadata={},
    )




# -----------------------------------------------------------------------------
# FIXTURE: store(tmp_path)
#
# WHY: Each test must get a FRESH, ISOLATED store so tests don't pollute each
# other. pytest's built-in `tmp_path` fixture hands you a unique temp directory
# per test — point ChromaVectorStore at it.
#
# STEPS:
#   1. Decorate with @pytest.fixture
#   2. Take `tmp_path` as an argument (pytest injects it automatically)
#   3. Return ChromaVectorStore(persist_dir=str(tmp_path))
#
# Write the fixture here:

@pytest.fixture
def store(tmp_path):
    return ChromaVectorStore(persist_dir=str(tmp_path))

# -----------------------------------------------------------------------------
# TEST 1 — HAPPY PATH: add → count → search → has_fund
#
# WHY: Proves the core write/read cycle works end to end against real ChromaDB.
#
# STEPS:
#   1. Build a few chunks (e.g. 3) for the same fund via make_chunk()
#   2. store.add(chunks)
#   3. assert store.count() == 3
#   4. results = store.search(<one of the embeddings>, top_k=2)
#        - assert len(results) == 2
#        - assert every result is a Chunk instance
#        - assert the closest result has the chunk_id you searched for
#          (searching with a chunk's OWN embedding should return it first)
#   5. assert store.has_fund("<that fund>") is True
#
# Write test_happy_path(store) here:

def test_happy_path(store):
    
    chunks = [
                make_chunk(chunk_id="c01", embedding=[0.1] + [0.35]*383),
                make_chunk(chunk_id="c02", embedding=[0.1, 0.9] + [0.35]*382),
                make_chunk(chunk_id="c03", embedding=[0.1, 0.9, 0.35] +[ 0.1]*381)
            ]
    
    store.add(chunks)

    assert store.count() == 3

    result = store.search(query_embedding= chunks[0].embedding, top_k=2)
    assert len(result) == 2
    assert all([isinstance(c, Chunk) for c in result])
    assert result[0].chunk_id == "c01"
    assert store.has_fund("Test Fund") is True

# -----------------------------------------------------------------------------
# TEST 2 — EMPTY / MISSING INPUT
#
# WHY: The retriever relies on these to decide FundNotFoundError. They must be
# correct on an empty store.
#
# STEPS (use the fresh `store` fixture — it starts empty):
#   1. assert store.count() == 0
#   2. assert store.search([0.1] * 384) == []        ← empty-collection guard
#   3. assert store.has_fund("Nonexistent Fund") is False
#   4. assert store.add([]) is None and count stays 0 ← empty add is a no-op
#
# Write test_empty_store(store) here:

def test_empty_store(store):
    assert store.count() == 0
    assert store.search([0.1] * 384) == []
    assert store.has_fund("My Fund") is False
    assert store.add([]) is None
    assert store.count() == 0

    


# -----------------------------------------------------------------------------
# TEST 3 — EDGE CASE: upsert does not duplicate
#
# WHY: We chose upsert (not add) so re-ingesting the same PDF updates rows
# instead of creating duplicates. This guards that decision.
#
# STEPS:
#   1. Build a chunk with a fixed chunk_id
#   2. store.add([chunk])           → count == 1
#   3. store.add([chunk]) again     → count is STILL 1 (not 2)
#   4. (optional) add the same id but with different text, then search and
#      assert the returned text is the UPDATED one
#
# Write test_upsert_no_duplicates(store) here:

def test_upsert_no_duplicates(store):
    chunk = make_chunk("c0", embedding=[0.1]*384)
    store.add([chunk])
    assert store.count() == 1     # inserted

    store.add([chunk])            # same id again
    assert store.count() == 1     # UPDATED in place — still 1, not 2
    
    chunk = make_chunk("c0", text = "check if this is updated")
    store.add([chunk])
    results = store.search(query_embedding = [0.1]*384, top_k = 1)
    assert results[0].text == "check if this is updated"