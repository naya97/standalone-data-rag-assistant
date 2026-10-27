# Week 9 — LLM Application Basics (Gemini): Summary Report

## Goal

Add reliable LLM responses on top of the Week 8 RAG and the cleaned retail data: prompt design, structured JSON outputs,
system/user messages, the function-calling concept, retries and timeouts, basic hallucination control, simple
summarization and insight functions over the cleaned dataset and the retrieved documents, and a fallback for every
function. Plus the Gemini / Vertex AI concepts (notes only: no Google Cloud account was available).

## What was built

```
question ─► retrieve ─► gate (lenient) ─► Gemini (system + user, JSON schema) ─► guardrails ─► answer + citations
                                               │  error, or output rejected twice
                                               ▼
                         extractive answer (Week 8), judged AGAIN by the stricter gate
```

| Piece | File |
|---|---|
| Settings (.env) | `src/llm/week9/config.py` |
| LLM client: Gemini, scripted fake for tests, error types, one shared client | `src/llm/week9/client.py` |
| Retries, time budget, request throttle | `src/llm/week9/retry.py` |
| Structured outputs (RAG answer, sales insights, document summary) | `src/llm/week9/schemas.py` |
| JSON/schema checks, citations, numbers must exist in the source | `src/llm/week9/guardrails.py` |
| Prompts for summaries and insights (the RAG prompt is `build_messages` in `src/rag/week8/prompt_builder.py`) | `src/llm/week9/prompts.py` |
| One way to ask for a small JSON object: send, check, repair once | `src/llm/week9/runner.py` |
| LLM answer for `/rag/ask`, with the extractive fallback | `src/rag/week9/llm_generator.py` |
| Sales figures computed by SQL | `src/llm/week9/data_facts.py` |
| Sales summary and insights | `src/llm/week9/insights.py` |
| Document summary | `src/rag/week9/doc_summary.py` |
| Function-calling concept: two tools, validation, execution | `src/llm/week9/tool_specs.py` |
| Endpoints | `src/api/rag_controller.py` (`generation`), `src/api/llm_controller.py` |
| Evaluation: gate experiment, LLM vs extractive | `src/rag/week9/gate_experiment.py`, `src/rag/week9/evaluator.py` |
| Command line | `run_llm_insights.py`, `run_llm_tools.py`, `run_llm_eval.py`, `run_gate_experiment.py`, `run_rag_query.py --mode` |
| Tests (scripted fake model, no network, no key) | `tests/week9/` |

## Key decisions

- **Extractive stays as the fallback.** The LLM is the main path; any model failure returns the Week 8 answer and says why (`generation.fallback_reason`).
- **Two-tier gate.** The gate in front of the LLM can be lenient (0.33) because the model reads the context and can say "not found". A verbatim chunk has no such reader, so it is judged again with the Week 8 value (0.50), also when it is the fallback.
- **The model never computes.** Sales numbers come from SQL; the model only words them, and every number it writes must exist in those facts. Document summaries and answers must cite passages that exist and use numbers that appear in them.
- **A guardrail failure gets one repair request**, with the reasons; then the fallback.
- **One shared client.** The RAG answer and the insight endpoints share one rate limiter and one request count (the free tier is measured in requests).
- **Decisions come from the evaluation, not from guesses.** The lexical threshold was compared on the saved questions (gate-only, no quota) and then checked end to end with the model on the questions it affects.
- **Function calling is a concept here.** The model chooses a tool and arguments; we validate them and run the tool. Sending results back and routing are Weeks 10–11.

## Results

Full tables: `reports/week9/llm_evaluation.md`.

| | |
|---|---|
| Week 8's 10 predefined questions, extractive → LLM (Gemini 3.5 Flash Lite) | answerable 7/8 → **8/8** correct; unanswerable 1/2 → **2/2** rejected |
| All questions tried with the LLM (18 different) | answerable 11 of 12 correct (1 false refusal); unanswerable 6 of 6 rejected (5 by the model, 1 by the gate) |
| Lexical threshold 0.50 → 0.33, gate only | 2 valid questions rescued, 2 more near-topic questions let through; the model refused both |
| Cost of a RAG answer | about 850 input + 45 output tokens; no hidden thinking tokens on Flash Lite |
| Sales insights | about 2,560 input + 350 output tokens; first version rejected twice, fixed version passed at once |
| Model time | median 5.8 s, up to 22 s |

## Known limitations

- **Small evaluation:** 18 different questions with the LLM, 10 of them in the required set. A difference of one question is not evidence. The leaks the model caught are 5 of about 14 in the evaluation sets.
- **Numbers check:** small numbers (years, months, counts like 2 or 5) always appear in the source, so an invented small number is not caught. Large figures and percentages are protected. The check does not judge whether a sentence is *sensible*.
- **Free tier:** limits depend on the model (a quota error reported 20 requests/day for one model; AI Studio shows 500/day for Flash Lite); an evaluation that does not fit the daily quota has to be spread over days (replies are cached).
- **Answer wording changes between runs** (temperature 0.1 is not 0); the evaluation uses cached replies for stable numbers.
- **Q27 is a false refusal**: the fact sits in a chunk outside the retrieved top 3 (a larger `top_k` was not tried).
- **The fallback texts are English only** (template and summary).
- **The function-calling demo** has only been unit-tested with a scripted model unless `python run_llm_tools.py --demo` was run.
- **Vertex AI** is documented, not executed.

## Reproduce

```bash
python -m pytest tests/week9 -v
python run_llm_insights.py sales --focus "Which categories are behind their revenue target?"
python run_llm_insights.py docs --filter project=projectBeta --filter document_type=invoice
python run_llm_eval.py --variants extractive@0.50,llm@0.33 --ids 1,2,3,4,5,6,7,8,9,10
python run_gate_experiment.py
python run_llm_tools.py --demo
uvicorn src.api.health_controller:app --reload      # /docs: /rag/ask, /llm/sales-insights, /llm/summarize-documents
```

Design-decision log, including every bug found and how it was fixed: `reports/week9/llm_design_walkthrough.txt`.
