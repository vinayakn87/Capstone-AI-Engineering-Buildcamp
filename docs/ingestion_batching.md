# Ingestion Memory & Batching — Design Note

Future-reference note on how `ingestion/pipeline.py` handles memory on large PDFs,
why the current design can OOM, the interim fix already in place, and the
streaming/batching redesign we deferred. Written after a 189-page factsheet got
OOM-killed ("Terminated") during `test_single_file_persist`.

---

## The problem

`ingest_pdf` currently runs the four stages **in full, one after another**, with
every stage's *entire* output alive in memory at the same time:

```
parse_pdf(path)        → list of ALL pages' (page, scheme, content)   ← all pages
  ↓
chunk_text per page    → all_chunks (ALL chunks for the whole document)
  ↓
embedder.embed_chunks  → fills an embedding on EVERY chunk at once
  ↓
store.add(all_chunks)  → writes everything
return all_chunks
```

So at peak we simultaneously hold: every page's parsed content **+** every chunk
**+** every embedding. For a 189-page factsheet that peak exceeds available RAM on
a constrained box (≈3–5 GB free, **no swap**) → the kernel `SIGKILL`s the process.

### Which part actually spikes
Measured/reasoned ranking of the cost:
1. **pdfplumber parsing — the dominant spike.** `extract_tables()` is heavy on
   table-dense factsheet pages, and pdfplumber **caches each page's parsed objects
   on the `PDF` object and never frees them while iterating** — memory climbs
   monotonically to the whole document's worth by the last page.
2. Chunks + embeddings themselves are **small** (~hundreds of chunks × 384 floats
   ≈ a few MB). Not the spike.
3. The embedding model + torch (~1–1.5 GB) is a fixed baseline, loaded once.

So the memory problem is **parsing**, not the vectors.

---

## Interim fix — already applied ✅

One line in `parse_pdf`'s loop, after we've extracted the page's text:

```python
for i, page in enumerate(pdf.pages, start=1):
    scheme_name = detect_scheme_name(page)
    output_parsed.append((i, scheme_name, extract_page_content(page)))
    page.flush_cache()          # ← release this page's cached objects before the next
```

`flush_cache()` drops pdfplumber's per-page object cache once we're done with the
page, so peak parsing memory stays ~**one page** instead of all 189. The extracted
**text** (plain Python strings in `output_parsed`) is unaffected — it's already
copied out and is cheap. This was enough to get the 189-page test passing.

**This fix addresses the dominant cause.** The streaming redesign below is only
needed if memory is still tight (e.g. running the whole multi-PDF corpus, or much
larger documents).

---

## The deferred redesign — streaming / batching

The structural fix is to stop holding every stage's full output at once. Process
**one page (or one batch) at a time**: parse → chunk → embed → store → discard,
then move to the next. Peak memory becomes ~one page's worth regardless of
document size.

### Approach A — generator parser + per-page pipeline (the main idea)

Turn `parse_pdf` into a **generator** that `yield`s a page at a time instead of
building a full list:

```python
def iter_pages(path):                       # generator — one page in flight
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            scheme  = detect_scheme_name(page)
            content = extract_page_content(page)
            page.flush_cache()
            yield (i, scheme, content)        # caller consumes, then we move on
```

Then `ingest_pdf` consumes the stream and flushes each page through immediately:

```python
def ingest_pdf(path, store, embedder, persist=True):
    fund_house = _fund_house_from_path(path)
    source_file = Path(path).name
    total = 0
    for (page_no, scheme, content) in iter_pages(path):
        chunks = chunk_text(content, page=page_no,
                            fund_name=scheme or fund_house, source_file=source_file)
        embedder.embed_chunks(chunks)        # embed just THIS page's chunks
        if persist:
            store.add(chunks)                # write + let chunks be GC'd
        total += len(chunks)
    return total                              # ← a COUNT, not list[Chunk]
```

Now only one page's chunks/embeddings exist at any moment.

### Approach B — batch the embedding (complementary, smaller win)

`embedder.embed` already batches internally via sentence-transformers'
`encode(batch_size=...)`. The only extra memory is holding *all* input texts and
*all* returned vectors in the list — which is small. So batching embedding alone
**doesn't** solve the OOM (parsing is the spike), but if vectors ever dominate,
cap them by embedding in fixed-size groups rather than the whole document.

---

## The tradeoff that makes this non-trivial: Mode 2

Approach A changes the **return type** from `list[Chunk]` → `int` (a count),
because the whole point is to *not* keep the chunks around. That conflicts with
**Mode 2 (Direct Upload)**, which needs the chunks **in memory** to merge into
session retrieval (they're never persisted, so the store can't supply them later).

| Mode | persist | needs the chunks back? | streaming OK? |
|------|---------|------------------------|---------------|
| 1 — Q&A / KB seeding | `True`  | No — they live in the store | ✅ stream, return count |
| 2 — Direct Upload    | `False` | **Yes** — held in `session_chunks` | ❌ must collect |

### Resolution options (pick when we implement)
1. **Yield chunks, let the caller decide.** Make `ingest_pdf` a generator of
   chunks; Mode 1 streams them to the store and discards (`for c in ...: store.add`),
   Mode 2 collects (`list(...)`). Single code path, caller controls memory.
2. **Two return shapes by mode.** `persist=True` streams and returns a count;
   `persist=False` collects and returns `list[Chunk]`. Simple but the signature
   lies depending on the flag.
3. **Keep batch for Mode 2, stream for Mode 1.** Mode 2 is *one user-uploaded*
   document (bounded, smaller); batch is acceptable there. Mode 1 / CLI seeding
   handles the big corpus, so stream only that path.

Leaning toward **option 1** (generator of chunks) — it keeps one code path and
pushes the collect-vs-stream decision to the caller, which is exactly who knows
whether the chunks are needed afterward.

---

## Decision / status

- **Now:** `flush_cache()` in `parse_pdf` is the active mitigation; the 189-page
  test passes. Keep the simple batch `ingest_pdf` returning `list[Chunk]`.
- **Implement streaming when:** routinely ingesting the full multi-PDF corpus
  OOMs even with `flush_cache`, or documents get materially larger.
- **When we do:** prefer Approach A (generator parser + per-page pipeline) with
  resolution **option 1** (generator of chunks) so Mode 2 still works.
- **Don't bother with:** Approach B alone — it doesn't address the parsing spike.

See also: the "Where/when do we optimize the pipeline for memory?" entry in
[rag_concepts.md](rag_concepts.md).
