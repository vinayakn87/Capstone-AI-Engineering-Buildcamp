# =============================================================================
# tests/test_ingestion.py — component tests for the ingestion pipeline
#
# COVERS (each: happy path, empty/missing input, one edge case):
#   - table_to_text       (pure)        — pdf_parser
#   - detect_scheme_name  (fake page)   — pdf_parser
#   - extract_page_content(fake page)   — pdf_parser
#   - chunk_text          (pure)        — chunker
#   - Embedder            (real model)  — embedder
#
# TESTING PROTOCOL (from CLAUDE.md):
#   - Unit tests may substitute the EXTERNAL pdfplumber Page (a fake stub is fine)
#     but must not mock our own internal logic.
#   - Embedder uses the REAL model — no mock (slow; loaded once via a fixture).
#   - Run with: python -m pytest tests/test_ingestion.py -v
#               (skip the slow model tests:  -m "not slow")
# =============================================================================


# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------
# You need:
#   - `pytest`                                   → fixtures + markers
#   - `table_to_text, detect_scheme_name,
#      extract_page_content`                     → from ingestion.pdf_parser
#      (prefer explicit imports over `import *`)
#   - `chunk_text`                               → from ingestion.chunker
#   - `Embedder`                                 → from ingestion.embedder
#   - `Chunk`                                    → from store.base (type checks)
#
# Write your imports here:
import pytest
from ingestion.pdf_parser import table_to_text, detect_scheme_name, extract_page_content
from ingestion.chunker import chunk_text
from ingestion.embedder import Embedder
from store.base import Chunk

# -----------------------------------------------------------------------------
# HELPER: FakePage — a stand-in for a pdfplumber Page
#
# WHY: the two parser functions read DIFFERENT things off a page, so the stub
# must serve both:
#   - extract_page_content() calls .extract_text() + .extract_tables()  (plain strings)
#   - detect_scheme_name()   calls .extract_text_lines()  (lines WITH font sizes)
# We don't need a real PDF — just an object returning canned values. This
# substitutes the EXTERNAL dependency (allowed), not our own code.
#
# CONTRACT:
#   FakePage(text="", tables=None, lines=None)
#     text   : plain string for extract_text()         (extract_page_content tests)
#     tables : list of tables for extract_tables()      (default [])
#     lines  : list of (text, size) tuples             (detect_scheme_name tests)
#
#     .extract_text()        -> self.text
#     .extract_tables()      -> self.tables
#     .extract_text_lines()  -> rebuild pdfplumber's shape from `lines`:
#          [{"text": t, "chars": [{"size": s}]} for (t, s) in self.lines]
#       (real pdfplumber returns line dicts whose "chars" each carry a "size";
#        we only need "size", so one synthetic char per line is enough)
#
# WATCH OUT: don't use a mutable default (tables=[] / lines=[]) — use None then
# build inside __init__ (the mutable-default-argument trap).
#
# Write the FakePage class here:

class FakePage:
    def __init__(self, text: str = "", tables: list | None = None, lines: list | None = None):
        # Store text as-is. For tables/lines: if None, default to an empty list
        # INSIDE the body (never `tables=[]` in the signature — mutable-default trap).
        self.text = text
        self.tables = tables if tables is not None else []
        self.lines = lines if lines is not None else []
    

    def extract_text(self) -> str:
        # Return the stored plain text.
        return self.text

    def extract_tables(self) -> list:
        # Return the stored tables.
        return self.tables

    def extract_text_lines(self) -> list:
        # Rebuild pdfplumber's line shape from self.lines (a list of (text, size) tuples):
        #   each line -> {"text": <text>, "chars": [{"size": <size>}]}
        return [{"text": t, "chars": [{"size": s}]} for (t, s) in self.lines]
        


# -----------------------------------------------------------------------------
# FIXTURE: embedder  (session-scoped — load the model ONCE for all embedder tests)
#
# WHY: SentenceTransformer load is expensive (~100MB). scope="session" builds it
# a single time and shares it across every test that asks for `embedder`.
#
# STEPS:
#   1. @pytest.fixture(scope="session")
#   2. def embedder(): return Embedder()
#
# Write the embedder fixture here:
@pytest.fixture(scope="session")
def embedder():
    return Embedder()

# =============================================================================
# table_to_text  (pure function — no PDF needed)
# =============================================================================
#
# TEST 1 — happy path:
#   table = [["Scheme", "NAV"], ["Kotak Bluechip", "50.2"]]
#   assert table_to_text(table) == "Scheme | NAV\nKotak Bluechip | 50.2"
#   (rows joined by "\n", cells by " | ")
#
# TEST 2 — None cells become "" (edge):
#   table = [["A", None], [None, "B"]]
#   assert table_to_text(table) == "A | \n | B"
#
# TEST 3 — empty / None input → "" (missing input):
#   assert table_to_text([]) == ""
#   assert table_to_text(None) == ""
#
# Write the three table_to_text tests here:

def test_table_to_text_happy_path():
    table = [["Scheme", "NAV"], ["Kotak Bluechip", "50.2"]]
    assert table_to_text(table=table) == "Scheme | NAV\nKotak Bluechip | 50.2"

def test_table_to_text_some_missing():
    table = [["A", None], [None, "B"]]
    assert table_to_text(table=table) == "A | \n | B"

def test_table_to_text_missing_input():
    assert table_to_text([]) == ""
    assert table_to_text(None) == ""

# =============================================================================
# detect_scheme_name  (uses FakePage — font size + keyword + clip logic)
# =============================================================================
# Reminder of the logic under test: among lines whose text contains "fund"/"scheme"
# and is < 80 chars, pick the LARGEST font; then clip the result at "fund".
# A page with no such candidate returns None.
#
# TEST 1 — largest-font "fund" line wins (happy path)
#   page = FakePage(lines=[
#       ("KOTAK SPECIAL OPPORTUNITIES FUND", 18),   # big title
#       ("Fund Manager: Mr Devender Singhal", 8),   # small, also says "fund"
#   ])
#   assert detect_scheme_name(page) == "KOTAK SPECIAL OPPORTUNITIES FUND"
#   (size breaks the tie between two "fund" lines)
#
# TEST 2 — non-"fund" callout is rejected even if it's BIGGER (edge)
#   page = FakePage(lines=[
#       ("Investment style", 20),     # biggest, but no "fund"/"scheme" → dropped
#       ("Kotak Liquid Fund", 12),    # the real scheme
#   ])
#   assert detect_scheme_name(page) == "Kotak Liquid Fund"
#   (proves the keyword guard runs BEFORE the size pick)
#
# TEST 3 — clip at "fund" when subtitle merged onto the title line (edge)
#   page = FakePage(lines=[
#       ("Kotak Special Opportunities Fund An open ended equity scheme", 18),
#   ])
#   assert detect_scheme_name(page) == "Kotak Special Opportunities Fund"
#
# TEST 4 — no scheme title on the page → None (missing input)
#   page = FakePage(lines=[("PORTFOLIO", 14), ("Disclaimer text", 8)])
#   assert detect_scheme_name(page) is None
#   (the pipeline relies on None to trigger the fund_house fallback)
#
# Write the four detect_scheme_name tests here:

def test_scheme_name_happy_path():
    page = FakePage(
        lines=[
      ("KOTAK SPECIAL OPPORTUNITIES FUND", 18),   # big title
      ("Fund Manager: Mr Devender Singhal", 8),   # small, also says "fund"
    ])

    assert detect_scheme_name(page) == "KOTAK SPECIAL OPPORTUNITIES FUND"

def test_scheme_name_fund_smallersize():
    page = FakePage(lines=[
      ("Investment style", 20),     # biggest, but no "fund"/"scheme" → dropped
      ("Kotak Liquid Fund", 12),    # the real scheme
    ])

    assert detect_scheme_name(page) == "Kotak Liquid Fund"

def test_scheme_name_clip():
    page = FakePage(lines=[
      ("Kotak Special Opportunities Fund An open ended equity scheme", 18),
    ])

    assert detect_scheme_name(page) == "Kotak Special Opportunities Fund"

def test_scheme_name_no_fund():
    page = FakePage(lines=[("PORTFOLIO", 14), ("Disclaimer text", 8)])

    assert detect_scheme_name(page) is None


# =============================================================================
# extract_page_content  (uses FakePage — text + tables)
# =============================================================================
#
# TEST 1 — happy path: plain text first, then tables below it
#   page = FakePage(text="Fund overview text",
#                   tables=[[["Scheme", "NAV"], ["Kotak", "50"]]])
#   result = extract_page_content(page)
#   assert "Fund overview text" in result
#   assert "Scheme | NAV" in result          ← table rendered + appended
#
# TEST 2 — page with no tables (edge)
#   page = FakePage(text="Just prose, no tables", tables=[])
#   assert "Just prose, no tables" in extract_page_content(page)
#
# TEST 3 — fully empty page (missing input)
#   page = FakePage(text="", tables=[])
#   assert extract_page_content(page).strip() == ""
#
# Write the three extract_page_content tests here:

def test_extract_content_happy_path():
    page = FakePage(text="Fund overview text",
                tables=[[["Scheme", "NAV"], ["Kotak", "50"]]])
    result = extract_page_content(page)
    assert "Fund overview text" in result
    assert "Scheme | NAV" in result

def test_extract_content_notables():
    page = FakePage(text="Just prose, no tables", tables=[])
    result = extract_page_content(page)
    assert "Just prose, no tables" in result

def test_extract_content_empty():
    page = FakePage(text="", tables= [])
    result = extract_page_content(page)
    assert result.strip() == ""



# =============================================================================
# chunk_text  (pure function — no PDF needed)
# =============================================================================
#
# TEST 1 — happy path: long text → multiple Chunks with correct metadata
#   text = " ".join(f"word{i}" for i in range(1200))       ← 1200 words
#   chunks = chunk_text(text, page=3, fund_name="Kotak Liquid Fund",
#                       source_file="KotakMF.pdf", chunk_size=500, overlap=50)
#   assert len(chunks) > 1
#   assert all(isinstance(c, Chunk) for c in chunks)
#   For every chunk:
#     - c.page == 3, c.fund_name == "Kotak Liquid Fund", c.source_file == "KotakMF.pdf"
#     - c.embedding == []                    ← embedder fills this LATER
#   assert chunk ids are unique and start with the stem + page,
#     e.g. each c.chunk_id.startswith("KotakMF_p3_c")
#
# TEST 2 — overlap: consecutive windows share `overlap` words (edge)
#   words = [f"word{i}" for i in range(1000)]; text = " ".join(words)
#   chunks = chunk_text(text, page=1, fund_name="F", source_file="x.pdf",
#                       chunk_size=500, overlap=50)
#   step = chunk_size - overlap = 450, so chunk[1] begins at word index 450:
#     assert chunks[1].text.split()[0] == "word450"
#
# TEST 3 — empty text → no chunks (missing input)
#   assert chunk_text("", page=1, fund_name="F", source_file="x.pdf") == []
#
# Write the three chunk_text tests here:

def test_chunk_text_happy():
    text = " ".join(f"word{i}" for i in range(1200))
    chunks = chunk_text(
        text, 
        page=3, 
        fund_name="Kotak Liquid Fund",
        source_file="KotakMF.pdf", 
        chunk_size=500, 
        overlap=50
    )

    assert len(chunks) == 3
    assert all(isinstance(c, Chunk) for c in chunks)
    for c in chunks:
        assert c.page == 3
        assert c.fund_name == "Kotak Liquid Fund"
        assert c.source_file == "KotakMF.pdf"
        assert c.embedding == []
    
    assert all(c.chunk_id.startswith("KotakMF_p3_c") for c in chunks)
    ids = [c.chunk_id for c in chunks]
    assert len(set(ids)) == len(ids)

def test_chunk_text_overlap():
    text = " ".join(f"word{i}" for i in range(1000))
    chunks = chunk_text(
        text, 
        page=3, 
        fund_name="Kotak Liquid Fund",
        source_file="KotakMF.pdf", 
        chunk_size=500, 
        overlap=50
    )

    assert chunks[1].text.split()[0] == "word450"

def test_chunk_text_no_chunks():
    text = ""
    chunks = chunk_text(
        text, 
        page=3, 
        fund_name="Kotak Liquid Fund",
        source_file="KotakMF.pdf", 
        chunk_size=500, 
        overlap=50
    )

    assert chunks == []



# =============================================================================
# Embedder  (REAL model — mark each @pytest.mark.slow; take `embedder` fixture)
# =============================================================================
#
# TEST 1 — embed_one: single string → flat list of 384 NATIVE floats
#   vec = embedder.embed_one("What is the expense ratio?")
#   assert isinstance(vec, list) and len(vec) == 384
#   assert all(isinstance(x, float) for x in vec)   ← not np.float32 (ChromaDB gotcha)
#
# TEST 2 — embed: list of N strings → N vectors, each 384-dim, order preserved (edge)
#   texts = ["alpha", "beta", "gamma"]
#   vecs = embedder.embed(texts)
#   assert len(vecs) == 3 and all(len(v) == 384 for v in vecs)
#   assert vecs == embedder.embed(texts)            ← deterministic / same order
#
# TEST 3 — embed_chunks: fills each chunk.embedding in place, returns None
#   chunks = chunk_text("some real-ish text " * 50, page=1,
#                       fund_name="F", source_file="x.pdf")   ← embedding == [] each
#   ret = embedder.embed_chunks(chunks)
#   assert ret is None                              ← mutates in place
#   assert all(len(c.embedding) == 384 for c in chunks)
#   assert all(isinstance(x, float) for x in chunks[0].embedding)
#
# Write the three Embedder tests here (each decorated @pytest.mark.slow):

@pytest.mark.slow
def test_embed_one(embedder):
    vec = embedder.embed_one("What is the expense ratio?")
    assert isinstance(vec, list) and len(vec) == 384
    assert all(isinstance(x, float) for x in vec)   # native floats, not np.float32


@pytest.mark.slow
def test_embed_batch(embedder):
    texts = ["alpha", "beta", "gamma"]
    vecs = embedder.embed(texts)
    assert len(vecs) == 3
    assert all(len(v) == 384 for v in vecs)
    assert vecs == embedder.embed(texts)            # deterministic / order preserved


@pytest.mark.slow
def test_embed_chunks_mutates(embedder):
    chunks = chunk_text(
        "some real-ish text " * 50,
        page=1,
        fund_name="F",
        source_file="x.pdf",
    )
    ret = embedder.embed_chunks(chunks)
    assert ret is None                              # mutates in place, returns nothing
    assert all(len(c.embedding) == 384 for c in chunks)
    assert all(isinstance(x, float) for x in chunks[0].embedding)