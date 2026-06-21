# Agentic Solution — Design Analysis

## What Is Being Compared

| | This Project (RAG Pipeline) | Agentic / Tool-Calling |
|---|---|---|
| Who decides what to retrieve | Python code (always retrieves) | The LLM (decides if/what to retrieve) |
| Number of LLM calls per query | 1 (synthesis only) | 1+ (reasoning + each tool call + final answer) |
| Query formulation | Raw user text → embedding | LLM rewrites query before searching |
| Multi-step retrieval | No | Yes |
| Control flow | `retriever.py` owns the logic | LLM owns the logic |

---

## 1. Architectural Differences

### Current RAG Pipeline

```
User query
    │
    ▼
retriever.py (Python)
    │  1. embed query with sentence-transformers
    │  2. ANN search ChromaDB → top-k chunks
    │  3. filter/query SQLite for metadata
    │  4. merge & rank results
    │  5. check if results empty → raise FundNotFoundError
    ▼
synthesizer.py
    │  build prompt: system + context chunks + user question
    │  call OpenAI once (synthesis only)
    ▼
Answer
```

**Key characteristic:** OpenAI is called exactly once, at the end. It never touches the stores. Retrieval logic lives entirely in Python.

---

### Agentic / Tool-Calling Architecture

```
User query
    │
    ▼
OpenAI (reasoning turn)
    │  receives: user question + tool definitions
    │  decides: which tool(s) to call, and with what arguments
    ▼
Tool execution (Python runs the tool)
    │  e.g. search_knowledge_base("Axis Bluechip expense ratio")
    │  e.g. get_fund_metadata("Axis Bluechip Fund")
    ▼
OpenAI (next reasoning turn)
    │  receives: previous tool results
    │  decides: call another tool OR generate final answer
    │  (loop continues until LLM issues no more tool calls)
    ▼
Answer
```

**Key characteristic:** OpenAI is in the loop at every step. The LLM decides the retrieval strategy; Python only executes what the LLM asks for.

---

### Proposed File Structure for an Agentic Version

```
agent/
  tools.py          ← tool definitions (schemas + Python implementations)
  agent.py          ← agentic loop: call OpenAI → execute tools → repeat
  synthesizer.py    ← (no longer needed separately; OpenAI handles synthesis in-loop)
  retriever.py      ← (demoted: now called only when OpenAI invokes the tool)
```

The `tools.py` file would define something like:

```python
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Search the mutual fund knowledge base using a natural language query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "top_k": {"type": "integer", "default": 5}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_fund_metadata",
            "description": "Retrieve structured data for a fund: NAV, expense ratio, AUM, category.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fund_name": {"type": "string"}
                },
                "required": ["fund_name"]
            }
        }
    }
]
```

And `agent.py` would run the agentic loop:

```python
messages = [{"role": "user", "content": question}]

while True:
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=TOOLS
    )
    if response.choices[0].finish_reason == "stop":
        return response.choices[0].message.content   # final answer
    # else: execute tool calls, append results, loop
    for tool_call in response.choices[0].message.tool_calls:
        result = execute_tool(tool_call)
        messages.append({"role": "tool", "content": result, ...})
```

---

## 2. Functional Difference: Who Decides What to Search

This is the crux of the difference, and your instinct is correct — **retrieval IS a kind of tool call in spirit**. The distinction is whether the LLM or Python code makes that decision.

### Current solution: retrieval is unconditional and mechanical

When a user query arrives in the current RAG pipeline:

1. Python **always** calls the retriever — no reasoning about whether retrieval is needed
2. The query passed to ChromaDB is the **raw user text**, embedded as-is
3. Python decides **top-k** (hardcoded), decides to **merge SQLite results** (hardcoded), decides to raise `FundNotFoundError` (hardcoded condition)

Example:
```
User: "What is the expense ratio of Axis Bluechip Fund?"
Python embeds exactly this sentence → ANN search → returns top 5 chunks
```

The retrieval is dumb: it sends exactly what the user typed, once, unconditionally.

### Agentic solution: retrieval is reasoned, conditional, and can be reformulated

With tool calling:

1. OpenAI **decides** whether retrieval is needed (for a general greeting, it might answer directly)
2. OpenAI **rewrites** the query for better semantic match: `"expense ratio Axis Bluechip"` instead of the full sentence
3. OpenAI can call the tool **multiple times** with different sub-queries if needed
4. OpenAI decides **when it has enough context** to stop retrieving and answer

Example:
```
User: "Compare the 3-year returns and expense ratios of Axis Bluechip and Mirae Asset Large Cap"
OpenAI calls: search_knowledge_base("Axis Bluechip 3-year returns")
OpenAI calls: search_knowledge_base("Mirae Asset Large Cap 3-year returns")
OpenAI calls: get_fund_metadata("Axis Bluechip Fund")
OpenAI calls: get_fund_metadata("Mirae Asset Large Cap Fund")
OpenAI synthesizes using all 4 results → answer
```

The current RAG pipeline would embed the entire comparison question as one vector and return 5 mixed chunks — potentially missing precise data on one or both funds.

---

## 3. Capability and Performance Differences

### What the Agentic Approach Can Do That the Current Cannot

| Capability | RAG Pipeline | Agentic |
|---|---|---|
| Decompose multi-fund comparison questions | No — one embedding, mixed results | Yes — separate tool calls per fund |
| Reformulate a vague query | No — raw text goes to ChromaDB | Yes — LLM rewrites for semantic precision |
| Skip retrieval for trivial questions | No — always retrieves | Yes — LLM decides |
| Call structured + semantic stores independently | No — Python always calls both | Yes — LLM calls whichever is appropriate |
| Iterative retrieval (read, decide, read again) | No | Yes |
| Explain its reasoning (chain-of-thought before tool call) | No | Yes (visible in tool call arguments) |
| Handle "fund not found" gracefully without custom code | No — requires `FundNotFoundError` | Yes — LLM reasons about empty results |

### Performance Trade-offs

| Dimension | RAG Pipeline | Agentic |
|---|---|---|
| Latency | Low — 1 LLM call | Higher — 2-5+ LLM calls per query |
| Cost | Low — synthesis prompt only | Higher — reasoning tokens + tool result tokens in every turn |
| Predictability | High — same code path every time | Lower — LLM decides; output varies |
| Debuggability | Easy — deterministic Python flow | Harder — must inspect tool call sequences |
| Accuracy on simple questions | Sufficient | Similar or marginally better |
| Accuracy on multi-hop / comparison questions | Often poor | Significantly better |

### When Each Approach Is Better

**Use the RAG pipeline (current design) when:**
- Questions are mostly single-fund lookups
- Latency and cost matter more than multi-hop reasoning
- You want predictable, auditable behavior
- The corpus is large and retrieval quality is paramount (tune the chunking/embedding instead)

**Use the agentic / tool-calling approach when:**
- Users ask comparison, ranking, or multi-fund questions frequently
- You want the LLM to route to the right store (semantic vs. structured) intelligently
- You are building toward a conversational agent that maintains state across turns
- You want the model's reasoning to be observable (tool call arguments show intent)

---

## Summary

The fundamental architectural shift is this:

> In the RAG pipeline, **Python orchestrates** and OpenAI only speaks last.
> In the agentic design, **OpenAI orchestrates** and Python only executes what it's told.

The current design is cheaper, faster, and simpler to reason about — a good foundation. The agentic design would require replacing `agent/retriever.py` + `agent/synthesizer.py` + `agent/agent.py` with a tool-definition file (`tools.py`) and an agentic loop in `agent.py` that calls OpenAI with `tools=` and processes `tool_calls` responses in a loop until the model returns `finish_reason: "stop"`.

Neither is universally better. The RAG pipeline is the right starting point for learning the fundamentals of retrieval and chunking before adding the complexity of an agentic loop.
