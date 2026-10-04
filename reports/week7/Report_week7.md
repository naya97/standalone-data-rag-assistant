# Report — Week 7: RAG Foundations & Document Preparation

## 1. Scope

Prepared a small, RAG-ready document collection with metadata, covering the RAG
foundation topics: PDF/text parsing, noise cleaning, chunking, metadata tagging,
and table handling. No embedding generation or vector store is included - that
is Week 8's scope. This week produces the clean, chunked, metadata-tagged input
Week 8 will embed and index.

This report covers the final, accepted state of the pipeline after several
rounds of testing against real documents uncovered and fixed real bugs along
the way. For the full step-by-step reasoning behind every design decision
(including the ones that were tried and reverted), see the companion file
`reports/week7/parsing_and_chunking_walkthrough.txt`.

## 2. Input Dataset

10 documents, deliberately mixing formats and provenance:

| Type | Format | File | Classified? |
|---|---|---|---|
| Invoice | .pdf | `invoice_2024-01_projectAlpha_computeEngine.pdf` | yes |
| Invoice | .txt | `invoice_2024-01_projectAlpha_computeEngine.txt` | yes |
| Invoice | .txt | `invoice_2024-02_projectBeta_cloudStorage.txt` | yes |
| Policy | .md | `policy_2024-01_projectAlpha_general.md` | yes |
| Policy | .md | `policy_2024-03_projectBeta_billing.md` | yes |
| Report | .txt | `report_2024-02_projectAlpha_computeEngine.txt` | yes |
| Report | .txt | `report_2024-03_projectBeta_security.txt` | yes |
| (external) | .pdf | `Task-3.pdf` (a real trainee-task brief) | no - unclassified |
| (external) | .pdf | `email_payload_spec.pdf` (a real JSON payload spec) | no - unclassified |
| (external) | .pdf | `الوظيفة_الثانية__1_.pdf` (a real Arabic job-task brief) | no - unclassified |

The first 7 were authored for this project and follow the naming convention
`<type>_<YYYY-MM>_<project>_[service].<ext>`. The last 3 are real external
documents, deliberately added to stress-test the pipeline against content we
don't control the formatting of - they live under `data/raw/week7/misc/`,
**unrenamed**, and are ingested as `unclassified` (see section 4).

## 3. Naming Convention - Opt-In Enrichment, Not a Gate

Early in this project, filename-derived metadata (`project`/`month`/`service`)
was treated as a requirement for ingestion. This was deliberately changed
after testing against the 3 real external documents above, for two reasons:

1. **It isn't actually required by the task** - the requirement is that
   metadata be *attached* to chunks, not that it be *derived exclusively from
   the filename*.
2. **Forcing a match is actively harmful.** An early, lenient version of
   `extract_metadata_from_filename()` accepted a *partial* match - if the
   filename had the right number of underscore-separated segments but one
   segment failed validation (e.g. the month), it still returned the other
   fields as if they were valid. This produced a confirmed, concrete bug:
   `email_payload_spec.pdf` was parsed as `document_type='email'`,
   `project='spec'` - plausible-looking, completely wrong, and silent (no
   warning). The fix: validation is now a single, whole-string, anchored
   regex with **no partial credit** - if any part fails, the entire filename
   is `unclassified`, with every field `NULL`/`None`.

Classification now never blocks ingestion. An `unclassified` document is
still fully loaded, cleaned, chunked, and stored - just without
`project`/`month`/`service` values to filter on later. `original_filename` is
always preserved verbatim, and `document_id`/`chunk_id` are app-generated
UUIDs rather than being derived from the filename at all, so retrieval
references stay stable even if a source file is later renamed.

## 4. Cleaning - `TextCleaner`

Raw `pypdf` text was found, by direct inspection, to contain: stray control
characters standing in for mangled bullet glyphs (`\x7f`, `\uf0b7`), almost no
preserved paragraph structure (`0` occurrences of `"\n\n"` in 2 of 3 real test
PDFs), and line-wrap artifacts (hyphenated words split across lines, near
one-word-per-line fragmentation in one document). Feeding this directly into a
chunker that relies on `"\n\n"` as a paragraph delimiter is close to useless.

`TextCleaner` runs between loading and chunking: Unicode NFKC normalization,
bullet-symbol unification, control-character stripping, de-hyphenation, and -
**for PDF-sourced text only** - reconstruction of real paragraph/sentence
breaks from sentence-ending punctuation. This last, "aggressive" set of rules
is explicitly **never** applied to `.md`/`.txt`: verified directly that
running it against our own clean `policy_2024-01...md` file destroyed its
markdown headers (a header with no trailing punctuation got merged onto the
next line). `aggressive=True` is gated strictly to `source_format == "pdf"`.

Two real bugs were found and fixed during this stage by testing against
actual documents (not just reasoning about the regex in the abstract):
regex-backtracking silently undid a just-inserted bullet paragraph break in
one case, and the same class of newline-collapsing rule initially destroyed a
spliced-in Markdown table's row structure in another. Both are detailed in
the walkthrough file.

## 5. Chunking - One Universal Algorithm

A single `DocumentChunker` is used for every document type and format - no
per-type chunking strategies (an earlier proposal to route `invoice`/`policy`/
`report` to three different chunkers was tested and rejected: real PDF
numbering doesn't reliably start at a true line start, and `"\n\n"` is nearly
absent from raw PDF text, so type-based header-detection rules proved
unreliable in practice).

**Algorithm:** text is split into whole `"\n\n"`-delimited units (a sentence, a
bullet, a short paragraph) and packed greedily into a chunk until the next
unit would exceed `chunk_size` (default 300 characters - a hard, verified
safety cap, never exceeded regardless of what precedes it: a single header
followed by ~5,000 words of unbroken text is still safely split into many
chunks, not kept whole). A unit is never split internally unless it alone
exceeds the cap. "Overlap" (default 50 characters) carries forward at most one
complete prior unit for light context continuity - never a raw character
fragment landing mid-word or mid-item.

This replaced an earlier character-sliding-window design after it was found,
via testing against a real bullet-heavy document, to reproduce whole list
items verbatim across chunk boundaries (very frequent paragraph breaks in
cleaned text, closer together than the overlap window, caused the character
window to back up into the previous item rather than a fresh one). A
regression was also caught and fixed during the rebuild: an early version of
the "carry one unit forward" rule could itself produce a chunk larger than
`chunk_size`, which directly violated the safety-cap guarantee - fixed by
making the overlap carry conditional on it still fitting under the cap.

Every chunk also carries `section_header` - the header text of the section it
belongs to, propagated to **every** chunk of that section (not just the
first one). This exists because an embedding model only "sees" the text it's
actually given, never a separate metadata column - a chunk several positions
into a long section, with no mention of the section's topic in its own text,
risks not being judged relevant to a query about that topic. `section_header`
is stored once, separately, and is meant to be combined with `chunk_text`
only at the point of actual use (constructing the embedding input in Week 8,
or a retrieved-context prompt) rather than duplicated into permanent storage -
the same underlying idea as "contextual retrieval."

## 6. Table Handling

`pypdf` has no concept of a table - a PDF table extracts as flat,
column-less text (verified on a generated invoice: `pypdf` returns
`"High-Memory Instance (n2-highmem-8) 500 hrs $8.50"` as one plain line, with
no delimiter between the service name, quantity, and price).

**Decision: keep `pypdf` as the primary extractor for everything, and use
`pdfplumber` narrowly, only for table detection**, rather than switching the
whole pipeline to `pdfplumber`. `pdfplumber.extract_tables()` correctly
recovered the exact row/column structure of the test invoice. Verified that
`pdfplumber`'s extracted cell values, joined with a single space in reading
order, reproduce `pypdf`'s flattened text for that table *exactly* - so the
garbled version can be safely found-and-replaced **in place** with a clean
Markdown table, with zero duplication and no change to surrounding reading
order.

**Markdown was chosen over alternative representations** after a direct
token-count comparison for this table:

| Representation | ~tokens | Note |
|---|---|---|
| Markdown table, no separator row | **50** | chosen - LLMs parse markdown tables reliably |
| Markdown table, with pipe-dash separator row | 54 | separator is purely decorative, dropped |
| CSV-like (commas, no padding) | 44 | cheapest, but less visually explicit |
| One `key: value` sentence per row, repeating each column name | 68 | most expensive - repeats fixed info on every row |

For a **small** table (fits in one chunk, as here), the whole table is kept
as a single Markdown block so all rows stay retrievable together (a query
comparing two rows needs both in the same chunk). A **large** table that
doesn't fit in one chunk would reuse the exact same `section_header`
mechanism from section 5 - header row as the shared context, each data row as
an independent packable unit - with no new architecture required.

A manual alternative (using `pypdf`'s `extraction_mode="layout"` plus a
whitespace-gap heuristic to split columns) was also tested for comparison: it
correctly reconstructed the simple invoice table, but produced **18-40 false
positive "table-like" lines per page** on real documents that have no table
at all (any indented bullet or hanging-indent line matches a "2+ spaces"
heuristic). `pdfplumber`, tested on the same non-table documents, reported
**zero** false positives, because it reasons from actual page geometry rather
than a whitespace guess. This is the concrete evidence behind using
`pdfplumber` for table detection rather than a hand-rolled heuristic.

Table extraction is a best-effort enhancement, not a hard requirement: if
`pdfplumber` fails to open a file or a specific row's text can't be located
verbatim for a safe swap, the failure is logged and that document still
ingests using `pypdf`'s original (uncleaned-table) text, rather than the
whole document failing.

## 7. Pipeline Run - Results

```
Files found:  10
Processed:    10 | Failed: 0
Classified:   7  | Unclassified: 3
Total chunks: 66
By type:   {invoice: 3, policy: 2, report: 2, unclassified: 3}
By format: {pdf: 4, txt: 4, md: 2}
```

Full step-by-step trace (Scan -> Load -> Metadata -> Clean -> Chunk -> Store per
file) is recorded in `logs/week7/ingestion.log`.

## 8. Failure Handling - Verified Scenarios

Following the same approach as Week 4 (fail loud, fail specific, keep the
rest of the batch going):

1. **No input documents found (fatal):**
   `CRITICAL - Ingestion Terminated: No supported documents (.txt/.md/.pdf) found in <dir>`
2. **Filename doesn't match the naming convention (non-fatal):** logged as
   unclassified; the document is still loaded, cleaned, chunked and stored.
3. **Corrupted/unreadable PDF (non-fatal):**
   `ERROR - Load Phase FAILED for <file>: ...` - attributed to the correct
   phase; the rest of the batch still completes.
4. **Table extraction failure on an otherwise-readable PDF (non-fatal):**
   logged as a warning; the document still ingests using the original
   (uncleaned) table text rather than failing entirely.

## 9. Storage Design

Stored in the same `data/db/local_storage.sqlite` used by Task 1, in two
tables fully independent of `rag_topics`: `rag_documents` (one row per
document - `document_id` UUID primary key, `original_filename`,
classification fields, `classification_status`) and `rag_chunks` (one row per
chunk - `chunk_id` UUID primary key, `section_header`, and the document's
metadata denormalized onto each chunk for direct filtering without a JOIN).
Chosen over a separate `.db` file so the collection stays easy to inspect
today, while remaining a drop-in source for a real vector store in Week 8.

## 10. Known Technical Debt / Limitations

- **PDF table trap - solved for detected tables, not eliminated in general.**
  A table `pdfplumber` fails to detect (e.g. no clear ruling/whitespace
  structure) still extracts as flat text via `pypdf` alone, with the same
  column-ambiguity problem described in section 6.
- **Arabic PDF word-spacing.** `pypdf` can drop the space between two Arabic
  words entirely during extraction - confirmed directly in the raw text,
  before any of our own cleaning code runs, and confirmed to happen at both
  Arabic-Arabic and Arabic-English boundaries. A second extraction library
  (`PyMuPDF`) was tested as a potential fix on the same real document: the
  result was **mixed** - it corrected some fused word pairs but not others,
  and introduced a different, new corruption elsewhere. This is not
  considered reliably fixable by swapping libraries and is documented as
  accepted technical debt; if this specific content's accuracy becomes
  important, OCR (rendering the page to an image and extracting text
  visually, bypassing the PDF's internal text layer entirely) would be the
  next thing to try, but was judged out of scope for the current batch.
- **Header/footer stripping** is implemented generically (drops a page's
  first/last line when it repeats across a large-enough fraction of a
  multi-page document's pages, or matches a pure page-number pattern) but is
  currently a no-op: none of the real sample PDFs (up to 3 pages each)
  contained any repeated running header or footer to strip. Kept in as a
  defensive measure for future documents that do have this problem.
- **A minor, intentionally-accepted cosmetic duplication.** A fix was
  prototyped to stop a bare section title (no bullet marker, no terminal
  punctuation) from ever being "carried forward" alone as a duplicated
  overlap unit - it worked, but was reviewed and deliberately **not kept**:
  the extra complexity wasn't judged worthwhile for the current scope. The
  known, accepted effect: a short section title can occasionally appear
  twice - once closing one chunk, once opening the next - never a
  correctness problem (no chunk exceeds the safety cap, no unit is ever
  split mid-way), just a small, bounded redundancy.

## 11. Next Steps (Week 8 scope, out of scope here)

- Generate embeddings for each `rag_chunks` row (combining `chunk_text` with
  `section_header` as the embedding input, per section 5) and load them into
  a vector store.
- Implement top-k similarity retrieval with metadata filtering
  (project/month/service/classification_status).
- Scale the input batch from 10 documents toward the originally-planned
  6-8 documents per type, now that the pipeline has been validated against
  both authored and real external content.
