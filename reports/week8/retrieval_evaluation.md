# Week 8 — Retrieval Evaluation Notes

Evaluation of the basic RAG assistant (`POST /rag/ask`) over the 10 documents prepared in Week 7
(60 chunks, 60 vectors in ChromaDB). Everything below is reproducible with:

```bash
python run_rag_eval.py                       # all 84 questions
python run_rag_eval.py --set original        # the first 49 questions (includes the 10 predefined ones)
python run_rag_eval.py --split holdout       # only the questions never used to tune anything
```

Full per-question output (every score, every signal, every answer): `data/eval/week8/eval_results.json`.

---

## 1. The 10 predefined questions (task requirement)

For each question: was the **correct source returned**? "Retrieved" = the expected document appeared at rank 1 of the
vector search (independent of the answerability gate). "Answer" = what the full system returned.

| # | Question | Expected source | Retrieved @1 | System answer | Result |
|---|---|---|---|---|---|
| 1 | Total amount due, Project Alpha, Jan 2024 | `invoice_2024-01_projectAlpha_computeEngine` (.txt or .pdf) | yes | answered, cites the `.txt` | correct |
| 2 | Vulnerabilities in Project Beta's March scan | `report_2024-03_projectBeta_security.txt` | yes | answered, cites it twice (best + supporting chunk), contains "SSL" | correct |
| 3 | Minimum password length, Project Alpha | `policy_2024-01_projectAlpha_general.md` | yes | answered ("16 characters") | correct |
| 4 | Who approves spending above Project Beta's budget | `policy_2024-03_projectBeta_billing.md` | yes | answered ("CTO") | correct |
| 5 | Required fields of the email ingestion payload | `email_payload_spec.pdf` | yes | answered (`mailbox_id`, …) | correct |
| 6 | Unit price of the n2-highmem-8 instance | `invoice_2024-01_projectAlpha_computeEngine.pdf` (table) | yes | answered ("$8.50", table preserved) | correct |
| 7 | متى موعد إجراء المقابلات؟ | the Arabic job-brief PDF | yes | **refused** ("not found") | **false refusal** |
| 8 | Total amount due, filters `project=projectBeta`, `document_type=invoice` | `invoice_2024-02_projectBeta_cloudStorage.txt` | yes | answered ("$1,200") | correct |
| 9 | Refund policy for Project Beta (no such policy exists) | must be refused | – | answered with an invoice chunk | **leak** |
| 10 | Cost of Project Gamma in Jan 2024 (project does not exist) | must be refused | – | refused (unknown entity "Gamma") | correct |

**Summary:** the correct source was retrieved for **8/8** answerable questions; the full system answered **7/8** of them
correctly and refused 1 (Q7, Arabic: the question says "إجراء" while the document says "تجرى", so only 33% of its key
words match and the lexical gate refuses it). Of the 2 unanswerable questions, 1 was refused (Q10) and 1 leaked (Q9,
see "Known limits").

---

## 2. Wider evaluation (84 questions)

### 2.1 Question set

| Set | File | Questions | Role |
|---|---|---|---|
| original | `data/eval/week8/eval_questions.json` | 49 (`dev` 33, `test` 16) | used to design the answerability rule |
| extended | `data/eval/week8/eval_questions_extended.json` | 35 (`calib` 24, `holdout` 11) | cross-language questions + fresh same-language ones; `calib` fits the cross-language threshold, `holdout` is only used to judge |

56 answerable and 28 unanswerable. Unanswerable questions are typed: `near_topic_missing` (17: the topic exists, the
fact does not), `off_topic` (7), `unknown_entity` (3), `filter_no_match` (1). Every answerable question lists the
expected source(s) and `answer_contains`: facts that must appear in the returned text, so "right document" and "right
information" are checked separately. The question file itself is validated against the stored documents before each run.

### 2.2 What is measured

| Measure | Meaning |
|---|---|
| Hit@k, MRR | is a chunk of the expected document in the top-k (independent of the gate) |
| correct | answered, first citation is the expected document, and the expected fact is in the answer |
| right source, fact missing | right document, but the fact sits in a neighbouring chunk |
| wrong source | answered from another document |
| false refusal | an answerable question was refused |
| correct rejection / leak | an unanswerable question was refused / answered anyway |

### 2.3 Results (final run)

| | Result |
|---|---|
| Retrieval | Hit@1 **94.6%**, Hit@3 **98.2%**, Hit@5 **100%**, MRR **0.969** |
| Answerable (56) | **47 correct (83.9%)**, 4 right source / fact missing, 2 wrong source, 3 false refusals (5.4%) |
| Unanswerable (28) | **15 rejected (53.6%)**, 13 leaked |
| Latency (service process, embedding model already loaded) | p50 **17 ms**, p95 **25 ms**, max 120 ms (first request) |

By language relation between the question and the best chunk:

| | Answerable | Unanswerable |
|---|---|---|
| Same language (59 questions) | 33 correct, 2 missing fact, 1 wrong source, 3 false refusals (of 39) | 12 rejected, 8 leaked (of 20) |
| Cross-language (25 questions) | 14 correct, 2 missing fact, 1 wrong source, **0 false refusals** (of 17) | 3 rejected, 5 leaked (of 8) |

By type of unanswerable question:

| Type | Rejected / total |
|---|---|
| unknown entity ("Project Gamma", "Delta", …) | 3 / 3 |
| filters match nothing | 1 / 1 |
| off topic | 6 / 7 (the leak is Q47, an Arabic HR question that is close to the billing policy) |
| **near topic, fact missing** | **5 / 17** |

By split (small samples, read with care):

| Split | Answerable correct | Unanswerable rejected |
|---|---|---|
| dev (33) | 20 / 23 | 6 / 10 |
| test (16) | 11 / 11 | 2 / 5 |
| calib (24) | 11 / 15 | 4 / 9 |
| **holdout (11), never used to tune anything** | **5 / 7** | **3 / 4** |

`dev` and `test` were used to design the rule, so their numbers are optimistic. `holdout` is the honest estimate, and it
has only 11 questions.

---

## 3. How the evaluation changed the system

| Finding | Action | Effect |
|---|---|---|
| Hit@1 = 1.0 on the first 10 questions, but top-1 cosine scores of answerable and unanswerable questions overlap (5 of 8 answerable scored below the best unanswerable one) | stop using one cosine threshold; gate on lexical coverage of the best chunk (≥ 50%) + unknown-entity check | rejection works for off-topic and unknown-entity questions |
| First run of the 35 extended questions: **0 of 17 valid cross-language questions answered** (retrieval was fine, Hit@3 = 100%; the lexical gate compares words, and a question and chunk in different languages share none) | cross-language pairs skip the lexical gate and need a cosine floor of 0.30, fitted on `calib` only (it sits between the off-topic questions, ≤ 0.12, and the lowest valid one, 0.48) | **17/17 answered**, holdout 5/5; the price: rejection of unanswerable questions fell from 71% to 54%, because cross-language questions are no longer refused wholesale. The prediction made before the change (15 rejected, 13 leaked, 3 false refusals) matched the run exactly |
| "How do **I** bake bread?" was flagged for the unknown entity "I" | stop words are never entities | no false refusal for questions containing "I" |
| 4 answers came from the right document but the neighbouring chunk | add one supporting chunk of the same document (score ≥ 65% of the best) to the extractive answer | fixed 2 (Q2, Q30); 4 remain (Q27, Q31, Q60, Q66), so the net count did not change |

---

## 4. Known limits (not fixed on purpose)

1. **Near-topic questions with a missing fact leak** (12 of 17, 5 of them cross-language). The retrieved chunk is
   topically close ("refund policy for Project Beta" → Beta's invoice), shares enough words with the question, and no
   cheap signal separates it from a valid question. Deciding "does this chunk actually contain the answer" needs a
   reader (an LLM, planned for Week 9) or a cross-encoder. Checking that the project/month/service combination exists
   would, by inspection of the 12 leaks, catch only 2 (Q40, Q41); this was not implemented or measured.
2. **The cross-language gate only removes off-topic questions.** The 0.30 floor rejects questions that are unrelated
   (cosine ≈ 0.12) but not near-topic ones; cosine alone cannot separate those (AUC 0.735 on the 25 cross-language
   questions).
3. **Lexical gate is literal.** It caused all 3 false refusals: Q7 (Arabic morphology), Q78 (question words such as
   "specification", "say" do not appear in the chunk), Q79 (the right chunk was ranked 4th, so the best chunk was unrelated).
4. **Retrieval misses at rank 1:** Q51 and Q75 (right document at rank 2), Q79 (rank 4).
5. **Extractive answers are verbatim chunks.** 4 answers have the right source but not the fact (Q27, Q31, Q60, Q66):
   the fact is in another chunk. Adding the chunk right after the best one would fix Q27 and Q66 but not Q31/Q60, whose
   fact is far away in the same PDF; not implemented to keep the system simple.
6. **Arabic PDF extraction** drops spaces between some words (Week 7 limitation), e.g. "مثلRibbon".
7. **Small evaluation set.** 84 questions, 11 in the holdout: differences of one or two questions are not conclusive.

## 5. Settings used for this run

`paraphrase-multilingual-MiniLM-L12-v2` (text only), cosine, `top_k = 3`, `RAG_MIN_LEXICAL_COVERAGE = 0.5`,
unknown-entity check on, `RAG_MIN_SCORE_CROSS_LINGUAL = 0.30`, `RAG_SUPPORT_RATIO = 0.65`, `RAG_MIN_SCORE = 0.0`
(a plain cosine gate adds nothing for same-language questions).
