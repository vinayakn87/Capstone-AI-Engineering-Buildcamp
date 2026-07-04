# =============================================================================
# agent/agent.py — the orchestrator ("conductor") of the RAG loop
#
# WHY THIS FILE EXISTS:
# The retriever finds chunks; the synthesizer turns chunks into an answer. Neither
# knows about the other. THIS file wires them together: it is the ONLY thing the
# UI talks to. One public entry point — `ask()` — runs the whole query lifecycle:
#
#     UI → agent.ask(question) → retriever.retrieve() → synthesizer.synthesize()
#                                        │                        │
#                                  list[Chunk] ──────────────────►┘
#                                        │
#                                        └──► also kept as `sources` (citation table)
#
# DESIGN (2026-06-30 — see the Design Change Log in the plan):
#   - Pure-RAG: retrieval is unconditional; no has_fund gate; no FundNotFound.
#   - Absence / disambiguation live in the SYNTHESIZER prompt, not here.
#   - Every answer ships with `sources` — the chunks it drew from — so the UI can
#     render a citation table of the actual fund_name / source_file / page used.
#
# WHY A CLASS (not a plain function):
# It holds its two collaborators (retriever, synthesizer), injected once and
# reused for every question. Same construct-once pattern as Retriever / Embedder.
# =============================================================================


# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------
# You need:
#   - `dataclass`           → from dataclasses, to define AgentResponse
#   - `Chunk`               → from store.base (the type held in `sources`)
#   - `Retriever`           → from agent.retriever (collaborator #1)
#   - `Synthesizer`         → from agent.synthesizer (collaborator #2 — NOT BUILT
#                             YET; this import will fail until synthesizer.py
#                             exists. That's expected: agent.py defines the shape
#                             the synthesizer must satisfy, and we build it next.)
#
# Write your imports here:



# -----------------------------------------------------------------------------
# THE SYNTHESIZER CONTRACT (what agent.ask depends on)
#
# We haven't written synthesizer.py yet, but agent.py pins its interface so the
# next step is just "fill this shape":
#
#     class Synthesizer:
#         def synthesize(self, question: str, chunks: list[Chunk]) -> str: ...
#
#   - IN : the user's question + the retrieved chunks (may span several funds,
#          may be empty).
#   - OUT: the final natural-language answer text (str). All grounding,
#          per-fund disambiguation, narrowing follow-ups, and the soft
#          "I don't have this → upload" behaviour happen INSIDE synthesize()'s
#          prompt. The agent does not post-process the text.
#   - Empty chunks: synthesize() is still called; its prompt yields an honest
#     "I don't have data on this" answer. (The agent does NOT special-case [].)
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# DATACLASS: AgentResponse
#
# WHY: `ask()` returns ONE object bundling everything the UI needs to render a
# turn: the prose answer AND the sources behind it. Keeping `sources` as the
# actual Chunks (not pre-formatted text) lets the UI build the citation table
# deterministically in code — we never trust the LLM to reproduce filenames/pages.
#
# FIELDS:
#   text    : str          — the synthesized answer (from synthesizer.synthesize)
#   sources : list[Chunk]  — the chunks the answer drew from (== what retrieve
#                            returned). The UI de-dupes these by
#                            (fund_name, source_file, page) into the citation table.
#
# Write the AgentResponse dataclass here:



# -----------------------------------------------------------------------------
# CLASS: Agent
#
# WHY: Holds the retriever + synthesizer and exposes the single `ask()` the UI
# calls. Thin by design — it orchestrates, it does not compute.
# =============================================================================

# -----------------------------------------------------------------------------
# __init__(self, retriever: Retriever, synthesizer: Synthesizer)
#
# WHY: Inject both collaborators (don't build them inside) so the UI constructs
# them once with a shared store/embedder, and so tests can pass fakes/reals.
#
# STEPS:
#   1. self._retriever   = retriever
#   2. self._synthesizer = synthesizer
#
# Write __init__ here:



# -----------------------------------------------------------------------------
# ask(self, question: str, session_chunks: list[Chunk] | None = None)
#     -> AgentResponse
#
# WHY: The one public method — the whole query lifecycle in three moves. This is
# what chat/app.py calls (Mode 1: session_chunks=None; Mode 2: pass the uploaded
# session chunks straight through).
#
# STEPS:
#   1. chunks = self._retriever.retrieve(question, session_chunks=session_chunks)
#        ← retrieval owns Mode-1-vs-Mode-2; the agent just forwards session_chunks.
#   2. answer = self._synthesizer.synthesize(question, chunks)
#        ← hand the SAME chunks to the synthesizer for grounding.
#   3. return AgentResponse(text=answer, sources=chunks)
#        ← the chunks do double duty: prompt context (step 2) AND citation sources.
#
# NOTE: no branching on empty chunks, no fund detection, no error raising — that
# is deliberate (pure-RAG). The synthesizer's prompt handles "nothing relevant".
#
# Write ask() here:
