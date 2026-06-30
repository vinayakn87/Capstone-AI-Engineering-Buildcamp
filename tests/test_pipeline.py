# =============================================================================
# tests/test_pipeline.py — END-TO-END ingestion tests (parse → chunk → embed → store)
#
# Two scenarios, both against the REAL stack:
#   - REAL pdfplumber parsing the REAL KotakMF factsheet
#   - REAL Embedder (sentence-transformers)
#   - REAL ChromaVectorStore in a per-test tmp_path (no mocks — store protocol)
#
#   SCENARIO 1 — single file
#   SCENARIO 2 — multiple files
#
# NOTE: these are SLOW (189-page PDF → hundreds of chunks → real embeddings).
# Mark them @pytest.mark.slow so they can be skipped with `-m "not slow"`.
#
# Run with: python -m pytest tests/test_pipeline.py -v
# =============================================================================


# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------
# You need:
#   - `pytest`                       → fixtures + markers
#   - `shutil`                       → copy the real PDF to 2 names (scenario 2)
#   - `Path`                         → from pathlib, to build/inspect paths
#   - `ingest_pdf`                   → from ingestion.pipeline (the function under test)
#   - `ChromaVectorStore`            → from store.vector_store
#   - `Embedder`                     → from ingestion.embedder
#   - `Chunk`                        → from store.base (type checks)
#
# Write your imports here:
import pytest
import shutil
from pathlib import Path
from ingestion.pipeline import ingest_pdf
from store.vector_store import ChromaVectorStore
from store.base import Chunk
from ingestion.embedder import Embedder



# -----------------------------------------------------------------------------
# CONSTANT: REAL_PDF — path to the real factsheet
#
#   REAL_PDF = Path("data/raw_pdfs/KotakMFFactsheetMay2026.pdf")
#
# Write the constant here:

REAL_PDF = Path("data/raw_pdfs/KotakMFFactsheetMay2026.pdf")

# -----------------------------------------------------------------------------
# FIXTURES
# -----------------------------------------------------------------------------
# embedder  — session-scoped, load the model ONCE:
#     @pytest.fixture(scope="session")
#     def embedder(): return Embedder()
#
# store     — function-scoped, fresh ChromaDB per test (reuse the tmp_path pattern
#             from test_store.py):
#     @pytest.fixture
#     def store(tmp_path): return ChromaVectorStore(persist_dir=str(tmp_path))
#
# real_pdf  — skip the whole module if the PDF isn't present (so the suite still
#             runs on a machine without the 28MB file):
#     @pytest.fixture
#     def real_pdf():
#         if not REAL_PDF.exists():
#             pytest.skip(f"missing test PDF: {REAL_PDF}")
#         return str(REAL_PDF)
#
# Write the three fixtures here:

@pytest.fixture(scope="session")
def embedder():
    return Embedder()

@pytest.fixture()
def store(tmp_path):
    return ChromaVectorStore(persist_dir=str(tmp_path))

@pytest.fixture
def real_pdf():
    if not REAL_PDF.exists():
        pytest.skip(f"missing test PDF: {REAL_PDF}")
    return str(REAL_PDF)

@pytest.fixture
def all_pdfs():
    pdfs = sorted(Path("data/raw_pdfs").glob("*.pdf"))
    if len(pdfs) < 2:
        pytest.skip(f"need >= 2 PDFs in data/raw_pdfs, found {len(pdfs)}")
    return [str(p) for p in pdfs]


# =============================================================================
# SCENARIO 1 — SINGLE FILE
# =============================================================================

# -----------------------------------------------------------------------------
# TEST — persist=True writes the whole file into the store
#
# WHY: proves the full pipeline runs and the store ends up holding exactly the
# chunks ingest_pdf produced.
#
# STEPS:
#   1. chunks = ingest_pdf(real_pdf, store, embedder, persist=True)
#   2. assert len(chunks) > 0
#      assert all(isinstance(c, Chunk) for c in chunks)
#   3. assert store.count() == len(chunks)        ← everything got written
#   4. Every chunk should carry an embedding now:
#        assert all(len(c.embedding) == 384 for c in chunks)
#   5. has_fund gate — derive the fund from the data itself (we can't hardcode a
#      scheme name from a 189-page PDF):
#        fund = chunks[0].fund_name
#        assert store.has_fund(fund) is True
#        assert store.has_fund("Definitely Not A Real Fund") is False
#   6. search round-trips — embed a chunk's own text and expect a hit:
#        q = embedder.embed_one(chunks[0].text)
#        results = store.search(q, top_k=5)
#        assert len(results) > 0 and all(isinstance(c, Chunk) for c in results)
#
# Write test_single_file_persist(store, embedder, real_pdf) here:

@pytest.mark.slow
def test_single_file_persist(store, embedder, real_pdf):
    chunks = ingest_pdf(
        real_pdf,
        store,
        embedder,
        persist=True
    )

    assert len(chunks) > 0
    assert all(isinstance(c, Chunk) for c in chunks)
    assert store.count() == len(chunks)
    assert all(len(c.embedding) == 384 for c in chunks)
    fund = chunks[0].fund_name
    assert store.has_fund(fund) is True
    assert store.has_fund("Definitely Not A Real Fund") is False

    q = embedder.embed_one(chunks[0].text)
    results = store.search(q, top_k=1)
    assert len(results) > 0 and all(isinstance(c, Chunk) for c in results)



# -----------------------------------------------------------------------------
# TEST — persist=False (Mode 2): returns chunks but writes NOTHING
#
# WHY: Mode 2 (Direct Upload) must never touch the persistent KB.
#
# STEPS:
#   1. chunks = ingest_pdf(real_pdf, store, embedder, persist=False)
#   2. assert len(chunks) > 0                      ← still parsed + chunked + embedded
#   3. assert store.count() == 0                   ← but nothing persisted
#
# Write test_single_file_no_persist(store, embedder, real_pdf) here:

@pytest.mark.slow
def test_single_file_no_persist(store, embedder, real_pdf):
    chunks = ingest_pdf(real_pdf, store, embedder, persist=False)
    assert len(chunks) > 0
    assert store.count() == 0

# =============================================================================
# SCENARIO 2 — MULTIPLE FILES
# =============================================================================

# -----------------------------------------------------------------------------
# TEST — ingesting several REAL PDFs accumulates in one store
#
# WHY: the CLI loops over every PDF in data/raw_pdfs/ into a single store; the
# counts must add up and EACH file must be queryable. Different PDFs (different
# fund houses) → different source_file → different chunk_ids → no upsert
# collision, so the per-file ingests genuinely add together.
#
# SETUP: collect every PDF in data/raw_pdfs/. This test only means something with
# 2+ files, so skip otherwise. A small fixture keeps it tidy (add to FIXTURES):
#     @pytest.fixture
#     def all_pdfs():
#         pdfs = sorted(Path("data/raw_pdfs").glob("*.pdf"))
#         if len(pdfs) < 2:
#             pytest.skip(f"need >= 2 PDFs in data/raw_pdfs, found {len(pdfs)}")
#         return [str(p) for p in pdfs]
#
# STEPS:
#   1. total = 0
#      funds = []
#   2. for pdf in all_pdfs:
#          chunks = ingest_pdf(pdf, store, embedder, persist=True)
#          assert len(chunks) > 0                  ← each file produced chunks
#          total += len(chunks)
#          funds.append(chunks[0].fund_name)       ← remember one fund per file
#   3. assert store.count() == total               ← additive across all files, no dedup
#   4. assert all(store.has_fund(f) is True for f in funds)   ← every file queryable
#
# NOTE: this drops the old shutil.copy trick — with real distinct PDFs you no
# longer need tmp_path copies (and `import shutil` can go if nothing else uses it).
#
# Write test_multiple_files_persist(store, embedder, all_pdfs) here:
@pytest.mark.slow
def test_multiple_files_persist(store, embedder, all_pdfs):
    total = 0
    funds = []
    for pdf in all_pdfs:
        chunks = ingest_pdf(pdf, store, embedder, persist=True)
        assert len(chunks) > 0
        total += len(chunks)
        funds.append(chunks[0].fund_name)
    assert store.count() == total                          # additive, no dedup
    assert all(store.has_fund(f) is True for f in funds)   # each file queryable


# -----------------------------------------------------------------------------
# TEST — re-ingesting the SAME file is idempotent (edge: upsert)
#
# WHY: re-running the pipeline on the same PDF must NOT double the store
# (deterministic chunk_ids → upsert overwrites in place).
#
# STEPS:
#   1. chunks = ingest_pdf(real_pdf, store, embedder, persist=True)
#   2. first = store.count()
#   3. ingest_pdf(real_pdf, store, embedder, persist=True)   ← same file again
#   4. assert store.count() == first                          ← unchanged, not doubled
#
# Write test_reingest_same_file_idempotent(store, embedder, real_pdf) here:

@pytest.mark.slow
def test_reingest_same_file_idempotent(store, embedder, real_pdf):
    chunks = ingest_pdf(real_pdf, store, embedder, persist=True)
    first = store.count()
    ingest_pdf(real_pdf, store, embedder, persist=True)   #same file again
    assert store.count() == first                         #unchanged, not doubled



