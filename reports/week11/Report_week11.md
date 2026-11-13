# Week 11 — Intro to Agentic AI + a simple LangGraph router

**Task (plan):** build a small graph — classify the question → choose RAG or SQL → run one tool → validate the result → answer.
Keep only two main routes and one fallback. **Deliverable:** a beginner LangGraph workflow that routes between document and data questions.

## 1. What was built

The Week 10 assistant (`choose → clean → validate → run → summarize`) became a **stateful graph** (`langgraph==1.2.20`). No new
endpoint, no loop, no memory between questions: one graph run answers one question.

```
question ─► check_input ──(empty / too long)──────────────────────────────► fallback ─► END
               ▼
            classify        (the Week 10 tool choice IS the classification: one model request)
      ┌────────┼──────────────────────────────┐
 search_documents     get_sales_summary    no tool / unknown tool / model failed ─► fallback ─► END
      ▼                     ▼
  rag_route             sql_route           (clean arguments, validate_call, run the tool: the Week 10 step)
      └──────────┬──────────┘
                 ▼
          validate_result ──(rejected call / tool error / no sales data)──► fallback ─► END
                 ▼
              answer        (Week 10 summarize_result: numbers and citations checked, one repair)  ─► END
```

| File | Role |
|---|---|
| `src/llm/week11/state.py` | `RouterState`: the one object that travels through the graph (question, route, tool, args, result, tool_status, valid, reason, outcome, summary, trace) |
| `src/llm/week11/validation.py` | the rules, as plain functions without a model: input guard, result check, fixed fallback messages |
| `src/llm/week11/nodes.py` | `make_nodes(assistant)`: the seven nodes, built around a Week 10 `Assistant` (its chooser, tools, cleaner, log and summarizer are reused) |
| `src/llm/week11/graph.py` | nodes and edges (three conditional edges) |
| `src/llm/week11/placeholders.py` | nodes without a model, the defaults of the graph tests |
| `src/llm/week11/evaluation.py`, `run_router_eval.py` | route + ending evaluation |
| `run_langgraph_router.py` | the router from the command line (`--verbose`, `--json`, `--show-graph`) |
| `data/eval/week11/router_cases.json` | 10 designed cases: what each one tests, expected route, expected ending, reason |

## 2. Design decisions

- **Classification costs no extra request.** `classify` is the Week 10 `choose` step: the tool the model picks decides the route
  (`search_documents` → `rag`, `get_sales_summary` → `sql`). A separate "label only" request would have cost a third request per question.
- **Fallback reasons are short codes**, the same in the state, the log and the evaluation: `empty_question`, `question_too_long`,
  `no_tool` (the model answered in words), `unknown_tool`, `model_unavailable`, `model_failed`, `rejected_call`, `tool_error`, `no_data`.
- **"Not found" is an answer, not a fallback.** A document search with no passages, or a summary that replies `NOT_FOUND`, ends as
  `not_found` on the RAG/SQL route. Only a question that cannot be answered *by the tools at all* ends in the fallback.
- **Fallback messages are fixed per reason**, in Arabic when the question contains Arabic script, otherwise English. The technical
  detail (the tool's own error) is never shown to the user: it is in `logs/week11/router.log` and in the state (`--json`).
- **Input limit = 1000 characters**, the limit the API already uses for a question (`rag_controller.py`). It protects against huge
  inputs; it is not tuned on the evaluation set (the longest evaluation question has 93 characters).
- **Request budget per question:** normally 2 (choice + summary); 3 in the worst case (one repair of a rejected summary, the Week 10 rule);
  a fallback costs 0 or 1. An empty or too long question never reaches the model.
- **Week 10 was not modified.** The nodes call `Assistant.choose()` and, for the tool step, `Assistant._run()` (a private method that
  already does clean → validate → run → log); a public wrapper in `Assistant` is the alternative if this ever matters.

## 3. Results

### Tests
42 tests, no key and no network (`tests/week11`): graph wiring, every fallback path, the limit (1000 accepted, 1001 rejected), the
language of the message, the nodes around the **real** Week 10 `Assistant` with a scripted model (rejected call, answer in words),
the evaluation's verdicts, and the 15 selected questions against the project's real question files.

### Live runs (Gemini 3.5 Flash Lite)
| Question | Path | Ending | Model calls |
|---|---|---|---|
| Which category has the highest revenue? | classify → sql_route → validate → answer | answer (Butchers, 208118.0) | 2 |
| ما هي الفئة الأعلى مبيعاً؟ | same | answer, in Arabic | 2 |
| What is the late payment fee? | classify → rag_route → validate → answer | not_found (the passages do not contain a late fee) | 2 |
| Total amount due on the Project Gamma invoice? | rag_route; `Project Gamma` cleaned to `projectGamma`, no passage | not_found, no summary request | 1 |
| What is the weather today? / the same in Arabic | classify → fallback | `no_tool`, message in the question's language | 1 |
| (empty question) | check_input → fallback | `empty_question` | 0 |
| Search the documents … top 50 passages | rag_route → rejected (`top_k` must be 1..10) | `rejected_call` | 1 |
| Revenue of the Toys category | sql_route → tool error | `tool_error` | 1 |

### Evaluation: 25 cases (15 existing questions + 10 designed), no holdout
Two things are checked for every case and they are different: **route** (what `classify` chose; a call that is rejected later still
counts as classified) and **ending** (`answer`, `not_found` or `fallback` with its reason). The 15 existing questions are also scored
with the Week 10 rules (facts, sources, citations).

```
                          n   route   ending   reason   content   pass
ALL                      25    23/25   22/25     9/10      9/9     22
```
- **All nine answerable questions pass, with the right content** (5 document, 4 sales; two of them in Arabic).
- **All the designed behaviours pass except the ambiguous "best category" (R4):** three out-of-scope questions, an unclear question,
  an empty question, an over-long question, a rejected tool call, a document question with no matching document, a sales question
  with no data.
- **Cost:** ended `answer` 2.0 model calls per question (9 cases), `not_found` 1.7 (6), `fallback` 0.8 (10). 9 requests were sent
  (budget 60), 27 replies came from the shared Week 10 cache. The 9 new requests are exactly what the designed cases that call the
  model need, which is consistent with the 15 existing questions having reproduced the Week 10 model replies.
- **Week 10 rules on the 15 existing questions:** 9 correct, 4 correct rejection, 2 answered in words (no tool).

### The three misses (the expectations were **not** changed after the run)
| Case | Expected | Got | Cause |
|---|---|---|---|
| Q47 (annual-leave policy, Arabic) | fallback `no_tool` | rag → `not_found` | The model chose `search_documents` for a "policy" question the documents do not cover (the Week 10 "tool for an off-topic question"). The ending is harmless: the user is told the information is not available. |
| Q1016 (profit margin) | sql → `not_found` | `no_tool` fallback | The model answered in words instead of calling the tool (the known Week 10 behaviour). The router turns it into a clean message; the message was reworded (see section 4). |
| R4 ("Find the best category.") | sql or rag → answer | sql → `not_found` | "Best" is undefined (revenue? transactions? target?) and the summary step refused to guess. The case design asked for *either route*; the ending `answer` was a stricter expectation of mine than the design needed. |

All three are the **model's tool choice or a stricter-than-needed expectation**, not a router mechanism failing. Q47 and Q1016 are the
two Week 10 findings that the plan assigned to Week 11's validation; the router now ends them in a controlled way (a clear
`not_found` / fallback message) but cannot change what the model chooses.

## 4. Bugs found and fixed during Week 11
- **The fallback node overwrote `route`**, hiding what `classify` had chosen → the evaluation reads the classification from `tool`.
- **The command line rejected an empty question itself** (argparse), so the input guard never ran → the question reaches the graph.
- **The `no_tool` message said "outside that scope" for a sales question about data we do not have** (profit margin) → reworded:
  out of scope, *asks for something those sources do not contain*, or unclear.
- **A designed case had the wrong route:** R8 (an invalid call) was written as `fallback`; the live run showed the model chooses a tool and
  the call is rejected later → the expected route is `rag`, the expected ending `fallback`. (Corrected before the evaluation.)
- **A request-count claim was imprecise:** "at most 2 requests per question" is 2 normally and 3 with the one summary repair.

## 5. Known limitations
- **Small set:** 25 cases (15 + 10). A difference of one or two cases is not evidence, and no threshold is set from it. A rule that
  needs a threshold needs a bigger set first. The holdout was not read.
- **The tool choice stays the model's.** The router cannot fix a model that answers in words (Q1016) or uses a tool for a policy
  question about a missing topic (Q47). Changing the tool prompt would invalidate the Week 10 cache and results, so it was not done.
- **Ambiguous questions** end in `not_found` or a rephrase request instead of a guess; the evaluation can only judge their route
  automatically, their answers are read by a person.
- **`no_data` is only unit-tested:** no live case produced an empty sales result.
- **The 15 existing questions used replies from the Week 10 cache** (same prompts), so the evaluation shows that the graph keeps the Week 10
  behaviour; run-to-run variation of the model is not measured.
- **Not built, on purpose:** loops, memory between questions, a checkpointer, an API endpoint.
- **Language of the fallback** is detected from Arabic script only (English or Arabic).
- **The first document search in a process loads the embedding model** (20–40 s); later ones do not.

## 6. Run
```bash
python -m pytest tests/week11 -v
python run_langgraph_router.py "Which category has the highest revenue?" --verbose
python run_langgraph_router.py "What is the weather today?" --json
python run_langgraph_router.py --show-graph
python run_router_eval.py --max-requests 60          # results: data/eval/week11/router_results.json
```
