# RAG Concepts — Q&A Log

Questions and discussions specifically about the Retrieval-Augmented Generation
system we're building (architecture, vector stores, embeddings, retrieval logic).

---

## Vector store fundamentals

### What goes in `metadatas` for ChromaDB?
A **list of dicts**, one per chunk, parallel to `ids`/`embeddings`/`documents`.
Each dict holds the structured fields kept with the chunk — `fund_name`,
`source_file`, `page`. Rules:
- Values must be **primitives only** (`str`, `int`, `float`, `bool`) — no nested
  dicts or lists.
- It's optional, but we need it: `has_fund()` filters on `fund_name`, and
  citations need `source_file` + `page`.

### Why a list of `Chunk` objects and not a list of dicts?
A dataclass gives a **contract**: fixed fields, fail-fast on missing/typo,
IDE autocomplete, and swappability (the store interface always receives
`list[Chunk]`). Dicts have no guarantees — a misspelled key fails silently.
The store's `add()` reads `c.chunk_id`, `c.embedding`, etc., so it needs Chunks.

### How does `upsert` prevent duplication?
"Upsert" = update + insert. For each id: if it already exists → update that row
in place; if new → insert. Since our `chunk_id` is deterministic
(`{stem}_p{page}_c{i}`), re-ingesting the same PDF produces the same ids, so
rows are overwritten rather than duplicated — `count()` stays stable.

### Why both `search()` and `has_fund()` in the interface?
Vector `search()` **always** returns top-k nearest neighbours, even for a fund
that isn't in the store (it returns the least-bad matches). So you cannot detect
a missing fund from `search()`. `has_fund()` does an **exact metadata lookup**
(`where={"fund_name": ...}`) — a precise yes/no. This is the gate that decides
whether to raise `FundNotFoundError`.

### How does `has_fund` / the empty-store test help with "fund not found"?
The retriever calls `has_fund()` as a **gate before searching**:
- `False` → raise `FundNotFoundError` → UI prompts for upload (don't search)
- `True`  → run `search()` → synthesize
The empty-store test proves the gate returns `False` correctly, and that
`search()` on an empty store returns `[]` cleanly (crash-guard) rather than
throwing on a cold start.

---

## Embeddings

### Why `.tolist()` on embeddings (the ChromaDB ValueError)?
`model.encode()` returns numpy arrays of `np.float32`. ChromaDB checks
`isinstance(value, float)` — `np.float32` is **not** a Python `float`, so it
rejects them. `list(np_array)` keeps the np.float32 scalars; `.tolist()`
**recursively** converts to native Python floats. Always use `.tolist()`.

### Why share one `Embedder` between ingestion and the retriever (Option A)?
Both ingestion (embedding chunks) and the retriever (embedding the question)
need the model. Loading `all-MiniLM-L6-v2` is expensive (~seconds, ~100MB).
A shared `Embedder` class loads it **once** in `__init__` and reuses it,
avoiding a duplicate model load and duplicated code.

---

## Architecture & pipeline

### How is ingestion triggered?
Two entry points, both calling `ingest_pdf()`:
1. **CLI:** `python -m ingestion.pipeline` → batch-ingest `data/raw_pdfs/*.pdf`.
2. **Streamlit:** on user upload → `ingest_pdf(path, store, embedder, persist=...)`.
   Mode 1 → `persist=True`; Mode 2 → `persist=False`.

### Where/when do we optimize the pipeline for memory?
The spike comes from holding all pages + all chunks + all embeddings at once
(crashed the notebook on the 189-page PDF). The fix is **streaming**: turn
`parse_pdf` into a generator (`yield` per page) and, in `ingest_pdf`, chunk →
embed → store **one page at a time** so each page's data is freed before the
next. Tradeoff: a streaming `ingest_pdf` returns a count, not `list[Chunk]`,
which conflicts with Mode 2 (needs the chunks). Recommendation: build the
simple batching version first; optimize only if it actually hits the wall.

### Two interaction modes (recap)
- **Mode 1 (Q&A):** retrieve from the persistent KB; missing fund → upload →
  ingest with `persist=True` → re-query.
- **Mode 2 (Direct Upload):** ingest with `persist=False`, session-only chunks
  merged with KB at retrieval, never written to the store.

---

## Build status (vector-only slice)
- `store/base.py`, `store/vector_store.py` — done + tested (`tests/test_store.py`)
- `ingestion/pdf_parser.py`, `chunker.py`, `embedder.py`, `pipeline.py` — done
- `tests/test_ingestion.py` (16 component tests), `tests/test_pipeline.py`
  (4 end-to-end tests) — done; pipeline tests pass against the real PDF
- Next: retriever → synthesizer (OpenAI) → Streamlit UI
- Structured store (SQLite) deferred until the vector slice runs end-to-end.

---

## Scheme detection & the fund-name data model

### What is `fund_name` — the scheme (sub-fund) or the fund house?
The **scheme** (e.g. "Kotak Liquid Fund"), not the house ("Kotak"). A consolidated
factsheet PDF covers *many* schemes — usually one per page. `fund_name` is the key
`has_fund()` runs on, and users type the **scheme** name, not the house. If we
stored the house, `has_fund("Kotak Liquid Fund")` would never match → the
FundNotFound gate breaks. So we detect the scheme per page and store both
`fund_name` (scheme) and a `fund_house` fallback.

### Why does "first non-empty line" fail for scheme detection?
`extract_text()` orders text by **vertical position**, so callouts physically
higher than the title ("Investment style", "Scan to Invest Now") come out first
and get mistaken for the scheme name. The signal the eye actually uses is **size**
— the scheme title is the visually dominant text. So we rank lines by **font
size**, not position.

### How `detect_scheme_name` works (font size + keyword + clip)
1. `page.extract_text_lines()` → lines, each with `chars` carrying a `"size"`.
2. **Keyword guard:** keep only lines containing "fund"/"scheme" and `< 80` chars
   → drops the callouts and long paragraphs.
3. **Largest font wins:** `max(candidates, key=font_size)` → the big title beats a
   small "Fund Manager: ..." line.
4. **Clip at "fund":** trim trailing text merged onto the title line ("KOTAK ...
   FUND An open ended scheme" → "KOTAK ... FUND").
5. **No candidate → `None`** → table/disclaimer pages fall through to `fund_house`.

### Why does the parser return `None` and the *pipeline* do the fallback?
Single responsibility. The parser **reports what it sees** ("found a header" /
"none" → `None`) — it shouldn't invent a name. The fallback value is the
`fund_house`, which comes from the **file path** — info the parser (operating on a
page object) doesn't have. The pipeline holds both, so it resolves `scheme or
fund_house` just before chunking. Keeping `None` also keeps the parser **testable**
(a TOC page → `None` deterministically) — collapsing it early destroys that signal.

### Can `fund_name` be `None` in the store?
No. ChromaDB metadata values must be primitives (`str/int/float/bool`) — `None` is
rejected. Every stored chunk needs a real string `fund_name`, which is why
`None`-scheme pages fall back to `fund_house` *before* `store.add`.

---

## Embeddings & pipeline mechanics (continued)

### Why embed chunks in one batch, not one-at-a-time?
`embed_chunks` gathers **all** texts and calls `self.embed(texts)` **once** —
sentence-transformers vectorizes the whole batch, far faster than N separate
`encode` calls. Calling `embed(c.text)` per chunk also passes a `str` to a method
typed `list[str]`; it happens to work by accident today but is fragile to any
tightening of `embed` (see python_concepts).

### Chunks ↔ embeddings ↔ stored rows are 1:1:1
One chunk gets exactly one embedding (`embed_chunks` zips one vector per chunk),
and `store.add` upserts one row per chunk_id. So `len(chunks) == #embeddings ==
store.count()` — *provided chunk_ids are unique* (they are: `{stem}_p{page}_c{n}`).
That's why `store.count() == len(chunks)` quietly also verifies no id collisions.
(Note: `len(embedding) == 384` is the vector's **dimension**, not a count.)

---

## Ingestion memory (the 189-page OOM)

The batch-everything pipeline (parse-all → chunk-all → embed-all → store-all) holds
every stage's full output at once; a 189-page PDF OOM-killed the test ("Terminated"
= SIGKILL, no swap). The **dominant** spike is pdfplumber parsing — it caches each
page's parsed objects on the `PDF` object and never frees them while iterating, so
memory climbs to the whole document's worth. Chunks/embeddings are small by
comparison.

**Interim fix (done):** `page.flush_cache()` in `parse_pdf`'s loop releases each
page's cache after extraction → peak parsing memory ~one page, not 189.
**Deferred:** the per-page streaming pipeline — see
[ingestion_batching.md](ingestion_batching.md) for the full design, tradeoffs, and
the Mode 2 conflict.

---

## Testing the ingestion stack

### Why a `FakePage` stub for the parser tests?
`detect_scheme_name` / `extract_page_content` only call a couple of methods on a
pdfplumber page (`extract_text`, `extract_tables`, `extract_text_lines`). A tiny
`FakePage` returning canned values lets us test our logic without a real PDF — this
substitutes the **external** dependency (allowed by the protocol), not our own
code. For font-size tests it rebuilds the real shape:
`{"text": t, "chars": [{"size": s}]}`.

### End-to-end pipeline tests (real stack, two scenarios)
`test_pipeline.py` runs the real pdfplumber + real `Embedder` + real
`ChromaVectorStore` (tmp_path). **Single file:** `persist=True` → `count ==
len(chunks)`, `has_fund` true for a fund derived from the data, search round-trips;
`persist=False` (Mode 2) → chunks returned but `count == 0`. **Multiple files:**
glob all PDFs in `data/raw_pdfs/`, ingest each, assert counts add up and each fund
is queryable; plus a re-ingest **idempotency** test (count unchanged). Assertions
**derive expected values from the data** (`chunks[0].fund_name`) since we can't
hardcode scheme names from a 189-page PDF.
