# Week 10 — LangChain Basics for Data and RAG Tools: Summary Report

## Goal

Learn how an LLM can call a few well-defined tools without building a complex agent: chains, retrievers, tools, output
parsers, tool calling, basic debugging/logging. Create 2–3 tools and connect them in a simple LangChain flow; deliverable:
a simple LangChain data-aware assistant and readable tool logs.

The plan's example tools map to this project as follows: `search_documents` (Week 8 retriever) → `search_documents`;
`query_cost_table` → `get_sales_summary` (SQL figures from the cleaned retail table); `summarize_result` → a **step** of the
flow, not a tool the model calls (the model would have to copy a previous result by hand: more tokens, more room for error).

## What was built

```
question ─► choose: prompt | Gemini.bind_tools(tools) ──► (no tool needed: the model answers in words)
                │
                ▼
        clean arguments (backend) ─► validate_call (Week 9) ─► StructuredTool ─► tool result
                                                                 │  search_documents: RagRetriever over the Week 8 retriever
                                                                 │  get_sales_summary: Week 9 SQL facts
                                                                 ▼
        summarize: plain text + [n] citations ─► output parser (numbers/citations checked, one repair) ─► answer
                                                                 │  failure of the model, or output rejected twice
                                                                 ▼
                                      fallback text built from the tool result itself (no model)
every model call goes through ONE request layer: shared Week 9 rate limiter, retries, time budget, request counter
every step is written to logs/week10/tool_calls.log (callback handler), one question id per question
```

| Piece | File |
|---|---|
| LangChain chat model from the Week 9 settings (`max_retries=0`), provider errors unwrapped | `src/llm/week10/model_factory.py` |
| One request layer: shared limiter, retries, budget, request counter | `src/llm/week10/request_layer.py` |
| The two Week 9 tools as `StructuredTool` + `run_validated` (validate, run, never raises) | `src/llm/week10/tools.py` |
| Week 8 retriever as a LangChain `BaseRetriever` | `src/rag/week10/langchain_retriever.py` |
| Cleaning of the model's search arguments (project form, month/year) | `src/llm/week10/arguments.py` |
| `summarize_result`: prompt, output parser with grounding checks, one repair, fallback | `src/llm/week10/summarize.py` |
| The assistant: choose → clean → validate → run → summarize, `--tool` without a model | `src/llm/week10/assistant.py` |
| Readable tool log (LangChain callback handler) | `src/llm/week10/tool_logging.py` |
| Evaluation: cases, scoring, reply cache, report | `src/llm/week10/evaluation.py` |
| Filter/top-k diagnostics: re-runs retrieval only, no model | `src/llm/week10/filter_check.py` |
| Sales questions (expected figures computed from SQL at run time) | `data/eval/week10/sales_questions.json` |
| Command line | `run_langchain_assistant.py`, `run_langchain_eval.py`, `run_filter_check.py` |
| Tests (scripted fake model, no network, no key): 177 | `tests/week10/` |

## Key decisions

- **Week 9 stays the source of truth.** Tool names, descriptions and argument schemas come from `tool_specs.py`; the LangChain tools are wrappers. LangChain does not check arguments when the schema is a plain dict (verified), so `validate_call` still runs before every tool.
- **One retry layer.** `ChatGoogleGenerativeAI` retries by itself (default 6); it is built with `max_retries=0` and the Week 9 `call_with_retry` + shared rate limiter is the only place a request is repeated. The package wraps API errors (`ChatGoogleGenerativeAIError … from <google error>`); the status code Week 9 reads is on the original error, so it is unwrapped first.
- **Summaries are plain text, not JSON.** Citations are written `[1]` and read back; every number must exist in the data the model was shown (Week 9 check); one repair request, then a fallback built from the tool result. Tool data goes in the user message, never in the system message.
- **Gemini extracts, the backend validates.** The model understands the question; the backend writes the filters in the stored form. Rules in `arguments.py`: `Project Beta` / `Beta` → `projectBeta`; an unknown project is never dropped nor mapped to another one (the filter matches nothing → "not found"); a month the question does not mention is dropped; an explicit year is never removed; a month **without a year is dropped** (the index stores month and year as one value, `2024-03`, and a year is never guessed). The list of valid projects is deliberately **not** put in the tool description: the model would map an unknown project to the nearest valid one and answer about the wrong project.
- **Calls and requests are counted apart.** `model_requests` = model calls made for an answer (failed ones included); `attempts` = requests actually sent, retries included, taken from the layer; a budget (`--max-requests`) stops a run before the limit request is sent.
- **Evaluation rules.** Replies are cached outside the request layer (a hit costs nothing); infrastructure failures are `not_evaluated`, never scored; a guardrail failure of the model's own output is scored. Three document questions that depend on API filters the assistant never sees (ids 8, 36, 49) are left out. `holdout` is only read when asked for.
- **`top_k` stays 3** (see Results): the gain at 5 or 8 is 1–3 questions of 47, its effect on the final answers was not measured, and more passages mean more tokens and more noise.
- **No change to prompts, the retriever or the chunker this week.** What the evidence pointed at is listed under Known limitations.

## Results

Evaluation on 87 questions (70 documents + 17 sales: dev, test, calib), `gemini-3.5-flash-lite`. Full tables: `data/eval/week10/*_results.json`.

**Tool choice** (one request per question)

| Group | Questions | Right | No tool used | Tool for an off-topic question | Wrong tool | Wrong arguments |
|---|---|---|---|---|---|---|
| documents / answerable | 47 | 47 | 0 | 0 | 0 | 0 |
| documents / near topic, fact missing | 14 | 11 | 3 | 0 | 0 | 0 |
| documents / off topic | 6 | 5 | 0 | 1 (Q47) | 0 | 0 |
| documents / unknown entity | 3 | 3 | 0 | 0 | 0 | 0 |
| sales | 17 | 16 | 1 (Q1016) | 0 | 0 | 0 |
| **All** | **87** | **82** | **4** | **1** | **0** | **0** |

When a tool was needed the model used one in 77 of 81 questions and never chose the wrong one.

**Arguments written by the model for `search_documents`** (checked against the question text): 14 projects written `Project Beta` / `Beta` instead of the stored `projectBeta`, 4 months with a year the question never gave (2025 three times; all the documents are from 2024). An exact filter that matches nothing returns no passage, so the model was never asked: **10 answerable questions failed this way** and 5 "correct rejections" were refusals by accident.

**Whole flow, documents answerable (47)**

| | Correct | Right source, fact not found | Refused; the source file was retrieved but not the fact | Expected source not retrieved |
|---|---|---|---|---|
| Before the argument cleaning | 31 | 4 | 2 | 10 |
| After the argument cleaning | **41** | 4 | 2 | **0** |

The cleaning needed 16 new requests (the choice replies were cached). The 5 questions that were refused by accident (Q9, Q37, Q39, Q81, Q82) now reach the 3 passages of the setting and are **still** refused: genuine rejections, no leak. The other groups did not change: near-topic 11 rejected / 3 answered in words, off-topic 5 answered in words / 1 rejected, unknown entity 3/3 rejected, sales 14/15 correct (Q1007 refused), the profit-margin question answered in words (Q1016), the missing category ended as a tool error (as expected).

**The six that are not `correct`, checked against the source documents**

| Question | Automatic outcome | Verdict |
|---|---|---|
| Q54, Q56 | right source, fact missing | **correct**: Arabic answers ("not allowed under any circumstances / VPN", "6 March") for an English source; the fact check wants the English literal. With them: 43/47 |
| Q31, Q60 | right source, fact missing | **wrong**: answered with the total (70 hours) instead of the chat-interface step (15 hours); the passage with that step was not in the text the model received |
| Q53, Q66 | refused | the fact was **not in the retrieved text** (the chunk holding it is not retrieved); the refusal is right for what the model had |

**Where the retrieval stands** (47 answerable questions, expected fact looked for in the passages of the expected source; filters as the backend writes them)

| k (passages) | fact in the text, cleaned filters | no filters | filters as the model chose them |
|---|---|---|---|
| 3 (the setting) | 43 | 41 | 33 |
| 5 | 44 | 44 | 34 |
| 8 | 46 | 46 | 36 |

The expected source file is in the top 8 for 47/47 with cleaned filters (37/47 as the model chose them). Gains: Q31 from k=5, Q53 and Q60 from k=8; **Q66 never** (the chunk holding "Ribbon" is separate from the heading it belongs to). The filters help at k=3 (Q14, Q57 are lost without them).

**Holdout** (11 questions, read once, before the cleaning): 9 evaluated; tool choice 9/9; answerable 4 correct + 1 fact-check miss (Q58, correct in meaning: "company credit card"); near-topic 3/3 rejected; off-topic 1 answered in words (Q69). Q52 and Q55 were not evaluated (`not_evaluated` is always an infrastructure failure; the message is in the `error` field of the results file); they are not counted as wrong.

**Cost** (whole evaluation: 183 real requests, retries included)

| | |
|---|---|
| Choice request | ~504 input + 24 output tokens |
| Summary request, documents | ~644 input + 19 output tokens |
| Summary request, sales (full facts, compact JSON) | ~2,040 input + 29–44 output tokens; no trimming limit was chosen without a measurement |
| Model time per request | median 5.7–5.8 s, 95th percentile 8.8–10.3 s |
| Per question | 2 requests; the first search of a process also loads the embedding model (~30 s, once) |

## Bugs found and fixed

- **Provider errors lost their status code** inside LangChain's wrapper → every API failure looked unknown and was not retried; the cause is unwrapped before the Week 9 mapping (tested with the real `google.genai` error classes).
- **LangChain's own retries (6 by default) would have multiplied every request** → `max_retries=0`, one layer.
- **Failed requests were not counted** (`model_requests` was 0 after three failed sends) → counted when made; `attempts` added from the layer; checked on five scenarios (retry in the choice, failing summary, failing choice, repair with a retry, first try).
- **Exact filters matched nothing** (see Results) → the backend cleaning; 31 → 41 correct.
- **`rank` is the rank of the FILE**: Q66 was "rank 1" while the chunk with the answer was not retrieved → the evaluation records whether the expected fact is in the text the model received, and separates `retrieval_miss` (not in the text) from `false_refusal` (it was).
- **A second run overwrote the first** (`full_results.json`) → the results file is named after the run (mode, splits, domain, `--tag`).
- **The filter-check summary printed "top 3" for numbers measured at k=8** → the label follows the k that was measured.
- **JSON in `--args` breaks in the Windows command line** → `--arg KEY=VALUE`.

## Known limitations

- **Small evaluation:** 70 document + 17 sales questions, groups of 3–6, holdout 11 (2 not evaluated). One question is 1–2 %; a difference of 1–3 questions is not evidence (this includes the choice of `top_k`).
- **Chunks are at most ~300 characters** (Week 7): a fact is often separated from its heading (Q66 "Ribbon" not retrieved even with 8 passages; Q53, Q60 need 8). Larger chunks, overlap or neighbouring chunks are the likely next step; they need re-indexing and are measured with `run_filter_check.py` without any model request.
- **Q31/Q60:** when the asked quantity is missing from the text the model answered with a related one (the total). The numbers check cannot see it (70 is in the text). A prompt rule was considered and **not** tested; a change of prompt invalidates the whole reply cache (one fingerprint for all prompts).
- **The expected fact "15 hours" occurs for three steps of the same PDF**, so the fact check cannot tell them apart (Q31, Q60). The fact check also compares literally, so a correct Arabic answer to an English source (Q54, Q56, Q58) is flagged; their raw outcomes were left as the rule produced them and the verdicts are in this report.
- **Q1007** ("which payment method brings the most revenue") was refused although the data reaches the model (the path from SQL to the prompt was checked). One sample: not explained.
- **Answers in words without a tool:** the model answers general knowledge (Q45, Q46, Q70, Q73) and, for company-sounding questions, sometimes too (Q44 "Cloud Run price"; Q71 returned the question itself). Allowed by the Week 9 design; checking it belongs to Week 11.
- **Sales figures are checked against the same SQL function the tool uses**: a shared error would not show. Internal consistency was checked (categories, payment methods and locations each add up to the total revenue).
- **The cleaning has no catalog by default:** a month without a year is dropped. `MetadataCatalog` (resolve the month from the indexed months, mark ambiguity) is implemented and tested but not wired, because the index holds one year.
- **The holdout was read once before the cleaning;** running it again would be a second look, not an independent test.
- **Cached replies:** wording is not re-sampled, so run-to-run variation was not measured. `gemini-3.5-flash-lite` ignores `temperature` (the package warning is hidden). Free-tier limits depend on the model.

## Reproduce

```bash
python -m pytest tests/week10 -v                                       # 177 tests, no key needed
python run_langchain_assistant.py "Which category has the highest revenue?" --verbose
python run_langchain_assistant.py "Within how many days must the Project Alpha invoice be paid?"
python run_langchain_assistant.py --tool get_sales_summary --arg category=Furniture      # no model request
python run_langchain_eval.py --mode choose --max-requests 100          # 1 request per question
python run_langchain_eval.py --mode full --max-requests 100            # choose + run + summarize
python run_langchain_eval.py --mode full --max-requests 40 --tag after_cleaning
python run_filter_check.py data/eval/week10/full_calib-dev-test_after_cleaning_results.json --all-answerable --top-ks 3,5,8
```

The readable log of every step is `logs/week10/tool_calls.log`.
