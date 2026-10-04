from __future__ import annotations

import logging
from pathlib import Path

import pdfplumber

logger = logging.getLogger(__name__)


def to_markdown_table(rows: list[list[str]]) -> str:
    """Convert a 2D list of strings into a markdown table string."""
    clean = lambda cell: (cell or "").strip().replace("\n", " ")
    lines = ["| " + " | ".join(clean(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def splice_clean_tables_into_text(pdf_path: str | Path, pypdf_text: str) -> str:
    """Detect tables in a PDF and replace the garbled pypdf text with clean markdown tables."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                for table in page.extract_tables():
                    if not table or len(table) < 2:
                        continue  # need a header row + at least one data row
                    pypdf_text = _replace_one_table(pypdf_text, table, pdf_path)
    except Exception as exc:
        # Table extraction is a best-effort enhancement, not a hard
        # requirement - a pdfplumber failure must never break ingestion
        # of a document that pypdf can still read.
        logger.warning("Table extraction failed for %s (continuing without "
                        "it): %s", pdf_path, exc)
    return pypdf_text


def _replace_one_table(text: str, table: list[list[str]], pdf_path) -> str:
    clean = lambda cell: (cell or "").strip()
    reconstructed_lines = [" ".join(clean(c) for c in row) for row in table]
    reconstructed_block = "\n".join(reconstructed_lines)
    markdown = to_markdown_table(table)

    if reconstructed_block in text:
        logger.info("Table found and cleaned (%d data row(s)) in %s",
                     len(table) - 1, pdf_path)
        return text.replace(reconstructed_block, markdown, 1)

    # Whole-block match failed (e.g. other text was interleaved between
    # rows in pypdf's version) - try swapping each row line independently.
    swapped_any = False
    md_lines = markdown.split("\n")
    for line, md_line in zip(reconstructed_lines, md_lines):
        if line in text:
            text = text.replace(line, md_line, 1)
            swapped_any = True
        else:
            logger.warning(
                "Could not locate table row text verbatim in %s - leaving "
                "that row as pypdf originally extracted it (known 'table "
                "trap' limitation, not fixed for this row): %r",
                pdf_path, line,
            )
    if swapped_any:
        logger.info("Table partially cleaned (row-by-row fallback) in %s", pdf_path)
    return text
