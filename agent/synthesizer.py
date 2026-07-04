# =============================================================================
# agent/synthesizer.py — turn retrieved chunks into a grounded answer (the "G")
#
# WHY THIS FILE EXISTS:
# The retriever found the relevant chunks; this file is where they become an
# answer. It builds a prompt from (question + chunks) and calls OpenAI ONCE.
# All the behaviour we deliberately kept OUT of the retriever lives here, in the
# prompt (pure-RAG design, 2026-06-30):
#   - GROUNDING       — answer only from the chunks; never invent figures.
#   - DISAMBIGUATION  — chunks may span several funds; name which fund each fact
#                       belongs to, answer per-fund, ask a narrowing follow-up
#                       when the question is ambiguous.
#   - SOFT ABSENCE    — if the chunks don't cover what was asked, say so, share
#                       what IS available, and invite the user to upload the
#                       factsheet. (This replaces the old FundNotFound gate.)
#
# WHAT THIS FILE DOES *NOT* DO:
#   - It does NOT build the citation table. That is assembled deterministically
#     in code from AgentResponse.sources (we never trust the LLM to reproduce
#     filenames/page numbers). The prompt may mention funds by name in prose,
#     but the structured table is the UI's job.
#   - It does NOT post-process or gate the answer — synthesize() returns the
#     model's text as-is.
#
# WHY A CLASS (not a plain function):
# It holds the OpenAI client + model name — set up once, reused per call. Same
# construct-once pattern as Embedder / Retriever.
# =============================================================================


# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------
# You need:
#   - `OpenAI`              → from openai (the v1 SDK client). Reads the
#                             OPENAI_API_KEY env var automatically.
#   - `Chunk`               → from store.base (the retrieved units we format)
#
# (If `openai` isn't installed yet: add it to requirements.txt and pip install.)
#
# Write your imports here:



# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------
# _MODEL — the OpenAI model id. gpt-4o-mini per the plan (good quality/cost).
#   _MODEL = "gpt-4o-mini"
#
# _SYSTEM_PROMPT — the instruction block that encodes the three behaviours above.
# This is the heart of the synthesizer; write it as a multi-line string. It MUST
# instruct the model to:
#   1. Answer ONLY using the provided context; if a fact isn't in the context,
#      do not state it. Never fabricate NAVs, ratios, returns, dates.
#   2. The context may contain chunks from MULTIPLE funds (each block is labelled
#      with its fund). Identify which fund(s) the question is about; if several
#      funds plausibly match, answer per-fund; if genuinely ambiguous, ask a
#      short clarifying question naming the candidate funds.
#   3. If the context does NOT contain what's asked, say plainly that you don't
#      have that data, share whatever relevant info IS present, and suggest the
#      user upload the fund's factsheet. Do not guess.
#   4. Be concise; refer to funds by name in prose (the app renders the structured
#      source table separately — you don't need to output a table).
#
#   _SYSTEM_PROMPT = """..."""
#
# Write the constants here:



# -----------------------------------------------------------------------------
# _format_context(chunks: list[Chunk]) -> str
#
# WHY: The model reads TEXT, not Chunk objects or vectors. This flattens the
# chunks into labelled blocks so the model can attribute every fact to a fund and
# page — the load-bearing piece that makes grounding + disambiguation possible.
# Module-level + pure (chunks in, str out) so it's unit-testable WITHOUT calling
# OpenAI (same rationale as _cosine_similarity in retriever.py).
#
# STEPS:
#   1. If chunks is empty → return a sentinel like "(no relevant context found)"
#      so the prompt still has something coherent to reason over (soft absence).
#   2. For each chunk, with a 1-based index, build a labelled block:
#        header = f"[{i}] Fund: {c.fund_name} | Source: {c.source_file} | Page: {c.page}"
#        block  = header + "\n" + c.text
#      (Note: we use fund_name / source_file / page — NOT c.embedding.)
#   3. Join the blocks with a blank line ("\n\n") and return the string.
#
# Write _format_context() here:



# -----------------------------------------------------------------------------
# CLASS: Synthesizer
#
# WHY: Wraps the OpenAI call behind synthesize(question, chunks) -> str, the exact
# interface agent.py already depends on.
# =============================================================================

# -----------------------------------------------------------------------------
# __init__(self, model: str = _MODEL, client: OpenAI | None = None)
#
# WHY: Build the client once and reuse it. `client` is injectable (defaults to a
# fresh OpenAI()) so tests can pass a fake client and assert on the prompt without
# a real API call (external HTTP may be mocked, per the testing protocol).
#
# STEPS:
#   1. self._client = client or OpenAI()      ← OpenAI() reads OPENAI_API_KEY
#   2. self._model  = model
#
# Write __init__ here:



# -----------------------------------------------------------------------------
# synthesize(self, question: str, chunks: list[Chunk]) -> str
#
# WHY: The one public method — the contract agent.ask() calls. Formats the
# chunks, builds the two-message prompt, calls OpenAI once, returns the answer.
#
# STEPS:
#   1. context = _format_context(chunks)
#   2. Build the messages list:
#        system → _SYSTEM_PROMPT
#        user   → f"Context:\n{context}\n\nQuestion: {question}"
#   3. response = self._client.chat.completions.create(
#          model=self._model,
#          messages=messages,
#          temperature=0,          ← low temp: factual, reproducible, less drift
#      )
#   4. return response.choices[0].message.content
#
# NOTE: empty chunks are NOT special-cased here — _format_context returns the
# sentinel and the prompt (instruction #3) produces an honest "I don't have that"
# answer. Absence handling is the prompt's job, not a Python branch.
#
# Write synthesize() here:
