# Python & Coding Concepts — Q&A Log

General programming concepts that came up while building the RAG system —
classes, OOP, idioms, and language mechanics. Not specific to RAG.

---

## Classes & OOP

### When do we need a class vs a plain function?
**Rule of thumb:** use a class when you have **state to bundle with behavior**;
use a function for a **pure transformation** (input → output, nothing remembered).

| Has expensive/reused state? | Use |
|---|---|
| No — data in, data out | **function** (`chunk_text`, `parse_pdf`, `table_to_text`) |
| Yes — loaded model, DB connection | **class** (`Embedder`, `ChromaVectorStore`) |

Smell test: if your class would have only one `do_it()` method and an empty
`__init__`, it should probably just be a function. Classes earn their keep when
`__init__` sets up something the methods repeatedly reuse.

### What goes into the `__init__` constructor?
Ask: "What does this object need to **remember between method calls**?" Anything
methods reuse and you don't want to recreate each call becomes instance state
(`self.x`). For `ChromaVectorStore`, every method needs the **collection**, so
`self._col` is set up once in `__init__` and reused. `__init__` runs once;
methods run many times — expensive one-time setup belongs in `__init__`.

### When do we write `self.`?
- **Storing** onto the object (in `__init__`): `self._col = ...` → survives the call.
- **Reading** the object's state (in any method): `return self._col.count()`.
- **No `self.`** for local scratch variables that only live inside one method
  (`results`, `chunks`, loop temporaries).
Test: "Does this need to outlive this method call?" Yes → `self.`; no → local.
Also: `self` is always the first parameter, but you never pass it — `store.count()`
passes `store` as `self` automatically.

### Why the leading underscore (`self._col`, `_COLLECTION_NAME`)?
Convention for "internal — don't touch from outside." Python has no true private;
the underscore signals it's an implementation detail that may change. Public
methods (`add`, `search`...) are the stable contract; `_client`/`_col` are
swappable plumbing. (Double underscore `__x` triggers name-mangling — rarely
needed; single underscore is right here.)

### Module-level constant vs constructor parameter vs class attribute
Things that may **vary per instance** → constructor parameter (e.g. `persist_dir`,
and arguably `collection_name` so tests can isolate). Things truly **global to
the program** → module constant. A class attribute groups a constant with its
class. For testability we lean toward parameters with sensible defaults.

---

## Python idioms & mechanics

### "Create a Chunk object" — what does that mean?
Instantiating the dataclass: `Chunk(chunk_id=..., text=..., embedding=[], ...)`.
`embedding` is `[]` at chunk time because the embedder fills it later — a
separation of concerns (chunking is independent of embedding).

### Why `embedding=None` then build inside, not `embedding=[0.1]*384` as default?
**Mutable default argument trap.** A list default is created **once** and shared
across all calls — a classic footgun. The `None`-then-build pattern gives each
call a fresh list.

### How does `enumerate(range(0, len(words), step))` work?
- `range(0, len, step)` → start positions (`0, 450, 900, ...`).
- `enumerate(...)` → pairs each with a counter: `(0, 0), (1, 450), (2, 900)`.
- `for i, start in ...` → unpacks both: `i` is the chunk index (for `chunk_id`),
  `start` is the word position (for slicing). Replaces a manual counter + manual
  `start += step`.

### `all(isinstance(c, Chunk) for c in results)` — what type, and `[]` vs `()`?
The inner piece yields `bool`s; `all(...)` collapses them to a single `bool`.
- `(... for ...)` → lazy **generator** (short-circuits, no intermediate list).
- `[... for ...]` → **list comprehension** → `list[bool]` (built fully first).
Both give the same result. Generator is leaner at scale; the list is easier to
`print()` while debugging. For small test data either is fine.

### Why does `store.add()` work — `store` isn't a `ChromaVectorStore`?
It is — via **pytest dependency injection**. A `@pytest.fixture` named `store`
returns a `ChromaVectorStore`; a test parameter named `store` receives that
return value. The name match is the wiring. You never assign `store` yourself
inside the test (doing so erases the injected object).

### `search()` returns a dict — why `result[0]`?
Two layers: raw `collection.query()` returns a **dict**; but our
`ChromaVectorStore.search()` **unpacks that dict and returns `list[Chunk]`**.
So `result[0]` indexes the list → a `Chunk`; `result[0].text` reads the
attribute. The dict only exists *inside* `search()`, before reconstruction.

### `.pop()` vs `.get()` when rebuilding a Chunk in `search()`
`pop` removes the key as it reads it, so after popping `fund_name`/`source_file`/
`page`, whatever **remains** in the dict is the leftover `metadata`. `get()`
would leave those keys in, duplicating them into `metadata`. This mirrors how
`add()` flattened them out.

---

## Testing concepts

### Why `tmp_path` in the store fixture?
pytest's built-in `tmp_path` gives each test a **fresh, isolated temp directory**.
Pointing `ChromaVectorStore` at it means every test starts with a clean real
ChromaDB — no mocks, no cross-test pollution (per the testing protocol).

### Why vary one embedding element / derive it from `chunk_id`?
Identical vectors all score 1.0 on a search → ties → arbitrary ordering → a
meaningless ranking assertion. Nudging makes vectors distinct so ranking is
deterministic. Deriving the nudge from `chunk_id` (not `random`) keeps it
**reproducible** across runs. (Caveat found in practice: `len(chunk_id)` collides
for same-length ids like `c0`/`c1`/`c2` — pass explicit distinct embeddings when
a test depends on ranking.)

---

## Classes & OOP (continued)

### Which methods take `self` and which don't?
A method takes `self` only if it **uses the object's state**.
- **Instance method** (has `self`) — reads/writes `self.x`. All `Embedder` methods
  qualify (they all touch `self._model`).
- **`@staticmethod`** (no `self`) — logically grouped under the class but touches
  nothing on the instance; pure input→output.
- **Module-level function** — if it's not tied to the class at all (no `self`, no
  reason to be namespaced under it). This is why `table_to_text` /
  `detect_scheme_name` are plain functions, not methods — there's no parser
  *object* with state. Test: "does it read/write `self.something`?" No → it
  shouldn't be an instance method.

### Why load the model in `__init__`, and how is that "less expensive"?
`__init__` runs **once per object**; methods run **many** times. Putting the
~100MB `SentenceTransformer(...)` load in `__init__` means it happens once and
every `embed*` call reuses `self._model`. The alternative (loading inside the
method) reloads 100MB on *every* call. Same load cost, amortized across hundreds
of calls instead of repeated. The efficiency only materializes if you **construct
the object once and hold the reference** — a fresh `Embedder()` per call buys
nothing.

### "Once per process," not "once per session"
With Streamlit's `@st.cache_resource`, the model loads once for the **running
process** and is **shared across all user sessions** hitting that server. 1 user
or 5 users → 1 load. Caching is keyed to the process, not the session, precisely
so a second user doesn't pay the load again. (A short-lived CLI run = 1 load per
run.)

### Why make `embed` private (`_embed`) — public vs private methods
A method is **public** because callers outside the class depend on it (the stable
contract); **private** (`_embed`) because it's internal plumbing you're free to
change. The test is "does anyone *outside* call it?" — not "is it useful?".
`embed` had zero external callers (only `embed_chunks` used it), so `_embed` is
honest. Start private; promote to public the moment a real external caller appears
— widening an interface is painless, narrowing a public one is a breaking change.

### YAGNI and DRY
**YAGNI** ("You Aren't Gonna Need It") — don't build for a caller that doesn't
exist yet; the speculative code is maintenance burden + a promise you must keep.
Build it when the need actually arrives. **DRY** ("Don't Repeat Yourself") — keep
one source of truth (e.g. `embed_one` reusing `_embed` so `.tolist()` lives in one
place). YAGNI says *don't pre-build*; DRY says *don't duplicate*. Both keep the
code no bigger than it needs to be.

---

## Python idioms & mechanics (continued)

### What does `line.strip()` do?
Returns a **copy** with leading/trailing whitespace (spaces, tabs, newlines)
removed — inner whitespace untouched. `"  Kotak Fund \n".strip()` → `"Kotak Fund"`;
`"   ".strip()` → `""` (so a whitespace-only line becomes falsy, which is how
"first non-empty line" logic skips blank lines). Cousins: `.lstrip()` / `.rstrip()`
(one side), `.strip("xy")` (strip those *chars*, not whitespace).

### How do `key=` functions work (`max(candidates, key=font_size)`)?
`key` is a function `max`/`min`/`sorted` call **once per element** to get a number
to rank by; it **returns the element**, not the number. Internally: `for el in
items: score = key(el); keep el with the biggest score`. So `max(lines,
key=font_size)` ranks lines by their font size but hands back the **line dict**.
Without `key`, `max` would compare the dicts directly and raise `TypeError`. Pass
the function (`key=font_size`), not a call (`key=font_size()`). Ties → the
**first** max-scoring element wins (only `>` replaces the best).

### The mutable-default trap, again (`FakePage(tables=[])`)
A default argument is evaluated **once at definition time** and stored on the
function — every call that omits the arg shares that **same** object. For `[]`
that means appends/mutations leak across calls (and across `FakePage` instances:
`a.tables is b.tables`). Fix: default to `None`, build `[]` inside the body
(runs per call → fresh list each time). Use `x if x is not None else []`, not
`x or []`, so an explicitly-passed empty list is respected. Only *mutable*
defaults are dangerous — `text=""` is fine (strings are immutable).

### Checking uniqueness with `set`
`assert len(set(ids)) == len(ids)` — a `set` discards duplicates, so if every id
is unique the set has the same length as the list; a collision makes the set
shorter → assertion fails. Standard "are these all unique?" idiom.

---

## Testing concepts (continued)

### How does `@pytest.fixture` work?
A fixture is a function providing a value a test needs; the test **requests it by
putting its name in its parameter list**, and pytest runs the fixture and injects
the return value (dependency injection — the name match is the wiring). Setup runs
before the `return`; if it `yield`s, code after the yield is teardown.
`pytest.skip(...)` inside a fixture skips any test using it (e.g. `real_pdf` skips
when the PDF is absent).

### Fixture `scope` — session vs function
`scope` controls how often a fixture is rebuilt and how widely it's shared:
- **`function`** (default) — fresh value **per test**. Use for mutable per-test
  state: `store` must be a clean ChromaDB each test, or tests pollute each other.
- **`session`** — built **once for the whole run**, shared. Use for expensive
  read-only state: `embedder` (load the model once).
Gotcha: a **session**-scoped fixture **cannot depend on a function**-scoped one
(e.g. `tmp_path`) → `ScopeMismatch`. That's why a `store(tmp_path)` fixture must
be function-scoped.

### Markers, `@pytest.mark.slow`, and the `-m` flag
Mark slow tests `@pytest.mark.slow`; register the marker in
`[tool.pytest.ini_options]` (pyproject) to silence "unknown mark" warnings.
Run/skip by **marker expression**: `-m "not slow"` skips them, `-m slow` runs only
them. Note the two different `-m`: in `python -m pytest`, `-m` is **Python's**
"run this module"; in `pytest ... -m "not slow"`, `-m` is **pytest's** marker
filter.

### Duplicate test name silently drops a test
Two `def`s with the **same name** in a module → the second overwrites the first in
the namespace, so pytest only collects the second. The first test vanishes — suite
goes green while a real case isn't run. Always give each test a unique name.

### One behavior per test (why not merge TEST 1 & 2?)
You *can* combine asserts, but pytest **stops at the first failed assert** in a
function — so a metadata failure would hide whether the overlap logic is also
broken. Separate tests (1) both run regardless of the other failing, (2) name the
failure for you (metadata vs overlap), (3) make happy/edge coverage explicit. The
rule is "one logical *behavior* per test," not "one assert per test."

### `is None` vs `== None`
Compare to `None` with **identity** (`is None`), not `==`. It's the convention,
linters flag `== None` (E711), and it avoids any custom `__eq__` surprises.
