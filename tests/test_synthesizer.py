# =============================================================================
# tests/test_synthesizer.py — UNIT tests for the synthesizer (isolated)
#
# SCOPE: this file tests the synthesizer's OWN logic in isolation — NO real
# OpenAI call, NO network, NO API key needed. Two testable surfaces:
#
#   1. _format_context(chunks)  → PURE function (chunks in, str out). Test it
#      directly — this is the easy, high-value part.
#   2. Synthesizer.synthesize() → calls OpenAI. We inject a FAKE client (the
#      `client` param exists for exactly this) so the method runs without a real
#      API call. External HTTP may be mocked per the testing protocol.
#
# Run with: python -m pytest tests/test_synthesizer.py -v
# =============================================================================


# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------
# You need:
#   - `pytest`
#   - `Chunk`                       → from store.base (build test chunks)
#   - `Synthesizer`, `_format_context`, `_SYSTEM_PROMPT`
#                                   → from agent.synthesizer (import the privates
#                                     too — tests are allowed to reach in)
#
# Write your imports here:
import pytest
from store.base import Chunk
from agent.synthesizer import Synthesizer, _format_context, _SYSTEM_PROMPT


# -----------------------------------------------------------------------------
# HELPER: make_chunk(...) — a tiny factory so tests aren't full of boilerplate
#
# WHY: Chunk has 6+ fields; most tests only care about a couple. A helper with
# sensible defaults keeps each test focused on what it's actually asserting.
#
# STEPS:
#   Define a function make_chunk(text="t", fund_name="Fund X",
#       source_file="f.pdf", page=1, embedding=None) -> Chunk that:
#     - defaults embedding to [] (or a small vector) when None  ← NOTE the
#       mutable-default trap: default the arg to None, build inside.
#     - returns Chunk(chunk_id=..., text=..., embedding=..., fund_name=...,
#                     source_file=..., page=..., metadata={})
#
# Write make_chunk() here:

def make_chunk(text="t", fund_name="Fund X", source_file="f.pdf", page=1,
               embedding=None, chunk_id="c1") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        embedding=embedding if embedding is not None else [],   # avoid mutable default
        fund_name=fund_name,
        source_file=source_file,
        page=page,
        metadata={},
    )



# -----------------------------------------------------------------------------
# FAKE OPENAI CLIENT — mimics the Responses API shape (NOT chat.completions)
#
# WHY: Synthesizer.synthesize calls `self._client.responses.create(...)` and
# reads `.output_text`. The fake must expose exactly that shape so synthesize
# runs unchanged, records what it was called with, and returns a canned answer.
#
# STEPS — build three tiny classes (or use unittest.mock, your call):
#   class _FakeResponse:      # what .create() returns
#       has attribute `output_text` (a str)
#   class _FakeResponses:     # the `.responses` namespace
#       .create(self, **kwargs) -> _FakeResponse
#         → STORE kwargs on self (e.g. self.last_kwargs = kwargs) so a test can
#           assert model / instructions / input were passed correctly
#         → return _FakeResponse(<canned text>)
#   class _FakeClient:
#       __init__ sets self.responses = _FakeResponses(...)
#
# This is what makes Synthesizer(client=_FakeClient()) skip the real OpenAI()
# (client or OpenAI() short-circuits on a truthy fake → no key needed).
#
# Write the fake client here:

class _FakeResponse:
    """What client.responses.create(...) returns — carries .output_text."""
    def __init__(self, text):
        self.output_text = text


class _FakeResponses:
    """The `.responses` namespace: records the call and returns a canned answer."""
    def __init__(self, text):
        self._text = text
        self.last_kwargs = None            # tests inspect this to assert the prompt

    def create(self, **kwargs):
        self.last_kwargs = kwargs          # remember model/instructions/input/temperature
        return _FakeResponse(self._text)


class _FakeClient:
    """Stand-in for OpenAI() — exposes .responses.create with .output_text."""
    def __init__(self, text="canned answer"):
        self.responses = _FakeResponses(text)



# =============================================================================
# TESTS — _format_context (pure, no fake needed)
# =============================================================================

# -----------------------------------------------------------------------------
# TEST — happy path: multiple chunks become labelled, separated blocks
#
# WHY: proves each chunk's fund/source/page/text land in the context and blocks
# are separated so the model can tell them apart.
#
# STEPS:
#   1. chunks = [make_chunk(text="expense ratio 0.5%", fund_name="Kotak Liquid Fund",
#                           source_file="kotak.pdf", page=42),
#                make_chunk(text="NAV 12.3", fund_name="Edelweiss Liquid Fund",
#                           source_file="edel.pdf", page=7)]
#   2. out = _format_context(chunks)
#   3. assert the fund names, "Page: 42"/"Page: 7", and both texts appear in out
#   4. assert "[1]" in out and "[2]" in out            ← 1-based labels
#   5. assert "\n\n" in out                            ← blocks are separated
#
# Write test_format_context_happy() here:

def test_format_context_happy():
    chunks = [
              make_chunk(text="expense ratio 0.5%", fund_name="Kotak Liquid Fund",
                source_file="kotak.pdf", page=42),
              make_chunk(text="NAV 12.3", fund_name="Edelweiss Liquid Fund",
                source_file="edel.pdf", page=7)
             ]
    out = _format_context(chunks)
    assert "Kotak Liquid Fund" in out and "Edelweiss Liquid Fund" in out
    assert "Page: 42" in out and "Page: 7" in out
    assert "expense ratio 0.5%" in out and "NAV 12.3" in out
    assert "[1]" in out and "[2]" in out
    assert "\n\n" in out

# -----------------------------------------------------------------------------
# TEST — empty chunks → sentinel string (soft absence)
#
# WHY: an empty retrieval must still yield coherent context so the prompt can
# say "I don't have that". _format_context must NOT crash on [].
#
# STEPS:
#   1. out = _format_context([])
#   2. assert out is a non-empty str (e.g. "no relevant context found" in out)
#
# Write test_format_context_empty() here:

def test_format_context_empty():
    out = _format_context([])
    assert out == "no relevant context found"

# -----------------------------------------------------------------------------
# TEST — a single chunk → exactly one block, no trailing separator
#
# WHY: edge case — the join must not add a dangling "\n\n" for one element.
#
# STEPS:
#   1. out = _format_context([make_chunk(text="hello", page=3)])
#   2. assert "hello" in out and "Page: 3" in out
#   3. assert out.count("\n\n") == 0                   ← one block, no separator
#
# Write test_format_context_single() here:

def test_format_context_single():
    chunk = [make_chunk(text="hello", page=3)]
    out = _format_context(chunk)
    assert "hello" in out and "Page: 3" in out
    assert out.count("\n\n") == 0

# =============================================================================
# TESTS — Synthesizer.synthesize (with the FAKE client)
# =============================================================================

# -----------------------------------------------------------------------------
# TEST — happy path: returns the model's text and calls the API correctly
#
# WHY: proves synthesize wires the prompt into the Responses API and returns
# response.output_text — without any real network call.
#
# STEPS:
#   1. fake = _FakeClient(<canned "the answer">)
#   2. synth = Synthesizer(client=fake)         ← injected fake, no OpenAI()/key
#   3. chunks = [make_chunk(text="expense ratio 0.5%", fund_name="Kotak Liquid Fund")]
#   4. answer = synth.synthesize("What is the expense ratio?", chunks)
#   5. assert answer == "the answer"            ← returned .output_text
#   6. Inspect what the client was called with (fake.responses.last_kwargs):
#        - kwargs["model"]        == the synth's model (default _MODEL)
#        - kwargs["instructions"] == _SYSTEM_PROMPT
#        - "expense ratio" in kwargs["input"]        ← the question is in the input
#        - "Kotak Liquid Fund" in kwargs["input"]    ← the context is in the input
#        - kwargs["temperature"] == 0
#
# Write test_synthesize_happy() here:

def test_synthesize_happy():
    fake = _FakeClient(text="canned answer")
    synth = Synthesizer(client=fake)
    chunks = [make_chunk(text="expense ratio 0.5%", fund_name="Kotak Liquid Fund")]
    answer = synth.synthesize(question="What is expense ratio", chunks=chunks) #synthesize -> client -> responses -> create method -> canned response (this canned rsponse is fed throguht client here; in real scenario we just have client = OpenAI() whish does not have any hardcoded text)
    assert answer == "canned answer"
    kwargs =  fake.responses.last_kwargs
    assert kwargs["model"] == synth._model
    assert kwargs["instructions"] == _SYSTEM_PROMPT
    assert "expense ratio" in kwargs["input"]
    assert "Kotak Liquid Fund" in kwargs["input"]
    assert kwargs["temperature"] == 0


# -----------------------------------------------------------------------------
# TEST — empty chunks still calls the API with the sentinel context
#
# WHY: absence is NOT special-cased in Python — synthesize should STILL call the
# model (its prompt handles "I don't have that"). This locks in that design.
#
# STEPS:
#   1. fake = _FakeClient("no data answer")
#   2. synth = Synthesizer(client=fake)
#   3. answer = synth.synthesize("Anything about HDFC?", [])
#   4. assert answer == "no data answer"
#   5. assert fake.responses.last_kwargs is not None         ← the API WAS called
#      assert "no relevant context found" in fake.responses.last_kwargs["input"]
#
# Write test_synthesize_empty_chunks() here:

def test_synthesize_empty_chunks():
    fake = _FakeClient("no data answer")
    synth = Synthesizer(client=fake)
    answer = synth.synthesize("Do pokemons exist?", [])
    assert answer == "no data answer"
    assert fake.responses.last_kwargs is not None
    assert "no relevant context found" in fake.responses.last_kwargs["input"]