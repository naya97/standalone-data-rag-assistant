# Week 9 — LLM Evaluation Notes

How the LLM answer, the insights and the summaries were checked, and what each check changed. Model: Gemini 3.5 Flash
Lite unless stated; `top_k = 3`. Everything is reproducible with the commands in `Report_week9.md`; per-question output:
`data/eval/week9/llm_eval_results.json` and `data/eval/week9/gate_experiment.json`.

---

## 1. The 10 predefined questions (task requirement), same as Week 8

| # | Question | Week 8 (extractive, gate 0.50) | Week 9 (LLM, gate 0.33) |
|---|---|---|---|
| 1 | Total amount due, Project Alpha, Jan 2024 | correct | correct |
| 2 | Vulnerabilities in Project Beta's March scan | correct | correct |
| 3 | Minimum password length, Project Alpha | correct | correct |
| 4 | Who approves spending above Project Beta's budget | correct | correct |
| 5 | Required fields of the email ingestion payload | correct | correct |
| 6 | Unit price of the n2-highmem-8 instance | correct | correct |
| 7 | متى موعد إجراء المقابلات؟ | **false refusal** (gate) | **correct** |
| 8 | Total amount due, filters project=projectBeta, document_type=invoice | correct | correct |
| 9 | Refund policy for Project Beta (does not exist) | **leak** | **rejected by the model** |
| 10 | Cost of Project Gamma in Jan 2024 (project does not exist) | rejected (gate) | rejected (gate; the model was not called) |

Answerable 7/8 → 8/8; unanswerable 1/2 → 2/2. The extractive column reproduced the Week 8 report exactly, so nothing in
the extractive path changed. Caveat: these ten questions were part of what the Week 8 gate was designed on, so they are
optimistic; and Q7 passes the 0.33 gate narrowly (1 of 3 key words = 0.333).

---

## 2. Which lexical threshold? Gate only, no model (`run_gate_experiment.py`)

83 questions, retrieval once per question, every setting judged on the same chunks. The 0.50 row reproduces the Week 8
numbers (51 correct, 2 wrong source, 3 false refusals, 13 leaks), which validates the tool.

| Threshold | Correct | False refusals | Wrong source | Leaks |
|---|---|---|---|---|
| 0.25 / 0.33 | 53 | 1 | 2 | 15 |
| 0.40 | 52 | 2 | 2 | 14 |
| **0.50** (Week 8) | 51 | 3 | 2 | 13 |
| 0.60 | 48 | 6 | 2 | 11 |

Every question rescued costs one leak: the lexical signal does not separate the two cases at the margin. 0.50 → 0.33
changes the decision of 4 questions only: rescued Q7 and Q78 (holdout), newly let through Q38 and Q48. The other 12–13
leaks pass at every threshold, so catching them is about the model, not the threshold. The light Arabic mode (spelling
normalisation and function words) changed no decision on this set; it stays off. The Arabic-to-Arabic group has only 6
questions (4 answerable, 2 not): too few to decide anything on it.

---

## 3. LLM end to end (`run_llm_eval.py`)

### 3.1 Questions that exercise the threshold and the leaks

| Variant | Answerable (6) | Unanswerable (4) |
|---|---|---|
| extractive @ 0.50 | 3 correct, 1 fact missing, 2 false refusals | 2 rejected (gate), 2 leaked |
| LLM @ 0.33 | **5 correct**, 1 false refusal (Q27) | **4 rejected** (all by the model) |

Questions: Q1, Q7, Q27, Q57, Q63, Q78 (answerable); Q38, Q48, Q47, Q39 (unanswerable). Q27: the extractive answer
returned text without the fact; the model said "not found" (a false refusal, but not a wrong answer).

### 3.2 Everything tried with Gemini 3.5 Flash Lite (18 different questions)

| | Result |
|---|---|
| Answerable (12) | 11 correct, 1 false refusal (Q27) |
| Unanswerable (6) | 6 rejected: 5 by the model (Q9, Q38, Q48, Q47, Q39), 1 by the gate (Q10) |

A count, not a rate. The model has seen 5 of the roughly 14 leaking questions of the evaluation sets.

### 3.3 First run, on another model (stopped by the quota)

A 73-question run on `gemini-3.5-flash` stopped after 31 questions: Google's quota error reported a limit of 20 requests per day. On the 29
answerable questions it reached, the LLM and the extractive answer both had 26 correct; the only difference was Q27
(fact missing → "not found"). It also showed the cost of that model: about 240 hidden thinking tokens per request, median
9.3 s and 95th percentile 52.8 s, and about a third of the request attempts (23 attempts for 15 new answers, derived) spent on failures and retries. See sections 6 and 7.

---

## 4. Cost and latency

| Request | Input tokens | Output tokens | Thinking tokens |
|---|---|---|---|
| RAG answer | ~850 | ~45 | 0 |
| Sales insights (24 months of figures) | ~2,560 | ~350 | 0 |
| Document summary (2 passages) | ~410 | 190–250 | 0 |
| (other model) RAG answer | ~840 | ~40 | ~240 |

Model time per request (Flash Lite): median 5.8 s, 95th percentile 11.3 s, one summary took 21.9 s for 188 output
tokens, so latency is not explained by answer length. With 15 requests/minute and 500/day, the limit that matters on the
free tier is the number of requests and the latency, not the tokens.

---

## 5. Sales insights and document summary on the real data

**Sales insights** (11,971 transactions, 37 months, 8 categories). The first version was rejected twice: the honest
answer to "which categories are behind their target?" is *none* (all between 562.85% and 650.37%), and saying so needs the
reference number 100, which was not in the facts. After the facts included definitions, the second version passed on the
first request; all four figures quoted matched the computed values, and the caveats mentioned that the targets are
hypothetical and that the last month is incomplete.

**Document summary** (Beta invoice, 2 passages): passed first time, twice; the facts (company, date, 50 TB, $1,200,
automatic card payment) are all in the passages and the source numbers are right. The wording differed between the two
runs.

---

## 6. Failure scenarios, checked against the real API

| Scenario | Observed |
|---|---|
| Wrong API key | `LLMAuthError` (Google answers 400 "API key not valid"), 1 attempt, no retry, the key does not appear in the output, fallback returned |
| Wrong model name | `LLMConfigError` 404 with Google's full message, 1 attempt, no retry |
| Timeout 0.5 s (`ConnectTimeout`) | 2 attempts (1 + 1 retry as configured), then the fallback; a slow answer (`ReadTimeout`) was seen in the first run |
| Daily quota exhausted (429) | detected as permanent (no pointless retries); the evaluation stopped, did not score the unanswered questions, and keeps its replies in the cache |
| Model down + a question only the lenient gate accepts | answered `no_relevant_context` with "stricter gate ... only 33% ... (minimum 50%)" instead of quoting a chunk |
| Model down + a question that passes the strict gate | `answered` with the extractive answer, `generation.mode = extractive_fallback` and the reason |

---

## 7. How the evaluation changed the system

| Finding | Action | Effect |
|---|---|---|
| With the 0.33 gate, a failed model call would quote a chunk for a question only the lenient gate accepted (Q48) | the stricter gate judges every verbatim answer | closed; confirmed with the real API |
| 429 = quota exhausted was retried three times (15–30 s each) | permanent quota errors are not retried; the provider's retry delay is honoured | no wasted attempts |
| 20 s timeout cut valid answers (median 9 s, up to 53 s); a retry restarts the slow answer | timeout 45 s, total 100 s, one retry | in the first run about a third of the attempts (23 for 15 new answers) were failures and retries |
| The model could not say "all above target" without the number 100 | definitions (including "100 = on target") in the facts | insights passed at once |
| The last month (2025-01) holds 202 transactions against ~327 in a normal one; the template called it the weakest month | incomplete months are marked and left out of best/worst/growth, with a note | no false "decline" |
| The targets are 6× below revenue | the facts say they are hypothetical figures added for this project | the model says so |
| The summary fallback showed only the header lines of an invoice and lost the amount due | a small selection is shown passage by passage | the amount is kept |
| Cached replies showed a median latency of 20 ms | latency is the model's own time, stored with the reply | 5.8 s / 11.3 s |
| `include_prompt` showed the old Week 8 prompt | it returns what was really sent (system + user) | debugging shows the truth |

---

## 8. Known limits (not fixed on purpose)

1. **Small samples.** See the counts above; the holdout questions were not used to choose anything (Q78 appears only as an illustration).
2. **The numbers check misses small numbers** that always occur in the source, and does not judge sense. Read the insights.
3. **Q27** needs a fact that is not in the three retrieved chunks. `top_k = 5` might fix it; not tried.
4. **Wording is not deterministic;** use the cache for stable evaluation numbers.
5. **Free-tier data may be used by Google;** only synthetic documents and a public sample dataset are sent.
6. **The sales targets are hypothetical** (added in Week 6), so "target achievement" is a demonstration, not a business result.

## 9. Settings used

`GEMINI_MODEL = gemini-3.5-flash-lite`, `LLM_TEMPERATURE = 0.1`, `LLM_TIMEOUT_SECONDS = 45`, `LLM_DEADLINE_SECONDS = 100`,
`LLM_MAX_RETRIES = 1`, `LLM_MAX_RPM = 10`, `RAG_TOP_K = 3`, `RAG_MIN_LEXICAL_COVERAGE = 0.33` (in front of the LLM),
`RAG_EXTRACTIVE_MIN_LEXICAL_COVERAGE = 0.50`, `RAG_MIN_SCORE_CROSS_LINGUAL = 0.30`, `RAG_ARABIC_LIGHT = False`.
Prompt versions: `rag_answer_v1`, `sales_insights_v2`, `doc_summary_v1`.
