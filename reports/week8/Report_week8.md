# Week 8 — Basic RAG Assistant: Summary Report

## Goal

Turn the document collection prepared in Week 7 into a working question-answering flow: embeddings, a local vector
store, retrieval with metadata filtering, a grounded prompt, citations, handling of missing information, an evaluation
set, and a `POST /rag/ask` endpoint. No LLM is used yet (Week 9): the answer is extractive.

## What was built

```
rag_chunks (SQLite, source of truth)
      │  run_rag_indexing.py  (idempotent sync, not a rebuild)
      ▼
embeddings (sentence-transformers, local) ──► ChromaDB (cosine) ──► POST /rag/ask
                                                                         │
 question ► embed ► top-k (+ metadata filters) ► answerability gate ► grounded prompt ► extractive answer + citations
```

| Piece | File |
|---|---|
| Settings from `.env` | `src/rag/week8/config.py` |
| Embedding model wrapper (loaded once, thread-safe) | `src/rag/week8/embedder.py` |
| ChromaDB wrapper (metadata cleaning, filters, model/variant guard) | `src/rag/week8/vector_store.py` |
| SQLite → vectors sync | `src/rag/week8/indexer.py`, `run_rag_indexing.py` |
| Retrieval + filters + stale-index detection | `src/rag/week8/retriever.py` |
| Can the best chunk answer the question? | `src/rag/week8/answerability.py` |
| Grounded prompt, fixed messages (English / Arabic) | `src/rag/week8/prompt_builder.py` |
| Extractive answer + citations (replaceable by an LLM in Week 9) | `src/rag/week8/answer_generator.py`, `rag_service.py` |
| HTTP endpoint | `src/api/rag_controller.py` (mounted from `health_controller.py`) |
| Evaluation | `src/rag/week8/evaluator.py`, `run_rag_eval.py`, `data/eval/week8/` |

Responses carry one of three statuses: `answered`, `no_relevant_context`, `no_documents_match_filter`. Every refusal
returns its reasons and the measured signals, so each decision is explainable.

## Key decisions

- **SQLite stays the source of truth; Chroma is only an index.** Text and metadata are read from SQLite by `chunk_id`.
- **Local multilingual embedding model** (`paraphrase-multilingual-MiniLM-L12-v2`), because the corpus mixes English and Arabic and no API key is needed.
- **No single similarity threshold.** On the evaluation set the cosine scores of answerable and unanswerable questions overlap, so a threshold would have been arbitrary. The gate uses lexical coverage of the best chunk, an unknown-entity check, and, for cross-language questions, a cosine floor fitted on a separate calibration split.
- **Re-ingestion replaces documents** (it used to append duplicates); every chunk gets a new id, so the index syncs by id, never by count.
- **Evaluation drove the design.** The first evaluation exposed that every valid cross-language question was refused; the fix was measured on a holdout.

## Results (84 questions, final run)

| | |
|---|---|
| Retrieval Hit@1 / Hit@3 / MRR | 94.6% / 98.2% / 0.969 |
| Answerable questions answered correctly (source and fact) | 47 / 56 (83.9%) |
| Answerable questions wrongly refused | 3 / 56 (5.4%) |
| Unanswerable questions correctly refused | 15 / 28 (53.6%) |
| Latency p50 / p95 | 17 ms / 25 ms |

The 10 predefined questions: the correct source was retrieved for 8/8 answerable ones, 7/8 were answered correctly,
1 of 2 unanswerable ones was refused. Details, tables and the iteration history: `reports/week8/retrieval_evaluation.md`.

## Known limitations

- Questions about an existing topic whose specific fact is missing mostly pass the gate (12 of 17 leak). No cheap signal separates them; an LLM or cross-encoder is needed.
- The cross-language gate removes off-topic questions only.
- The lexical gate is literal (3 false refusals).
- Extractive answers are verbatim chunks; 4 right-source answers lack the fact because it sits in another chunk.
- The evaluation set is small (84 questions, 11 in the holdout).
- After every ingestion the index must be rebuilt (`python run_rag_indexing.py`); the API detects a stale index and answers 503. Restart the API after changing the documents (the vocabulary of the unknown-entity check is built at startup).

## Reproduce

```bash
python run_rag_ingestion.py
python run_rag_indexing.py
python run_rag_eval.py
uvicorn src.api.health_controller:app --reload      # http://127.0.0.1:8000/docs
```

Design-decision log, including every bug found and how it was fixed: `reports/week8/rag_design_walkthrough.txt`.
