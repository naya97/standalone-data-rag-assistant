"""
Pipeline order (order matters - see inline comments):
  1. Unicode NFKC normalization       (universal, safe)
  2. Bullet symbol normalization      (universal, safe)
  3. Stray control-character removal  (universal, safe)
  4. De-hyphenation                   (universal, safe)
  5. De-fragmentation of fake newlines   (PDF-only)
  6. Force paragraph breaks on real sentence ends (PDF-only)
  7. Whitespace normalization         (universal, safe, always last)
"""

from __future__ import annotations

import re
import unicodedata

# Symbols observed in real extracted PDFs standing in for a bullet point:
# \uf0b7 and \x7f are font-mapping artifacts (Wingdings-style custom fonts),
# the rest are genuine Unicode bullet-like symbols used directly.
BULLET_CHARS_PATTERN = re.compile(r"[\uf0b7\x7f\u25cf\u25cb\u25a0\u25aa\u2022]")

# Sentence-ending punctuation (Arabic + Latin) used as the single source of
# truth for where a real line/paragraph break belongs.
SENTENCE_END_CHARS = ".:؟!"

# word-\nword -> word  (only letter...letter, so we don't merge across
# numeric patterns like "2024-\n01" or a genuine em-dash break)
DEHYPHEN_PATTERN = re.compile(r"([A-Za-z\u0600-\u06FF])-\n\s*([A-Za-z\u0600-\u06FF])")


class TextCleaner:
    def clean(self, text: str, aggressive: bool) -> str:
        """
        Args:
            text: raw extracted text.
            aggressive: True ONLY for PDF-sourced text. Enables the
                fake-newline de-fragmentation and forced-paragraph rules.
                Must be False for .md/.txt files we control - see module
                docstring for why.
        """
        text = self._normalize_unicode(text)
        text = self._normalize_bullets(text)
        text = self._strip_control_chars(text)
        text = self._dehyphenate(text)

        if aggressive:
            text = self._protect_markdown_tables(text)
            text = self._defragment_newlines(text)
            text = self._force_paragraphs(text)
            text = self._restore_markdown_tables(text)

        text = self._normalize_whitespace(text)
        return text.strip()

    # -- steps ----------------------------------------------------------

    @staticmethod
    def _normalize_unicode(text: str) -> str:
        # Fixes ligatures (fi, fl glyphs) and other compatibility variants.
        return unicodedata.normalize("NFKC", text)

    @staticmethod
    def _normalize_bullets(text: str) -> str:
        # Consume any surrounding whitespace/newlines together with the
        # bullet glyph itself, and re-emit a clean "start of a new line"
        # bullet marker, so list items are never glued to the previous line.
        return re.sub(r"\s*" + BULLET_CHARS_PATTERN.pattern + r"\s*", "\n\n* ", text)

    @staticmethod
    def _strip_control_chars(text: str) -> str:
        return "".join(
            ch for ch in text
            if ch in ("\n", "\t") or unicodedata.category(ch)[0] != "C"
        )

    @staticmethod
    def _dehyphenate(text: str) -> str:
        return DEHYPHEN_PATTERN.sub(r"\1\2", text)

    _TABLE_ROW_SEP = "\uE001"   # restored to a single "\n" (between table rows)
    _TABLE_BOUNDARY = "\uE002"  # restored to "\n\n" (separates table from prose)

    @classmethod
    def _protect_markdown_tables(cls, text: str) -> str:
        """A table in pypdf's column-less text is a series of lines that
        all start with a pipe character. We protect that block from the
        de-fragmentation and forced-paragraph rules by replacing the
        newlines between the rows with a placeholder, and wrapping the
        whole block in a second placeholder. The placeholders are
        restored to their original newlines after those rules have run.
        """
        lines = text.split("\n")
        out: list[str] = []
        i = 0
        while i < len(lines):
            if lines[i].lstrip().startswith("|"):
                block = [lines[i]]
                i += 1
                while i < len(lines) and lines[i].lstrip().startswith("|"):
                    block.append(lines[i])
                    i += 1
                protected_block = cls._TABLE_ROW_SEP.join(block)
                out.append(cls._TABLE_BOUNDARY + protected_block + cls._TABLE_BOUNDARY)
            else:
                out.append(lines[i])
                i += 1
        return "\n".join(out)

    @classmethod
    def _restore_markdown_tables(cls, text: str) -> str:
        text = text.replace(cls._TABLE_ROW_SEP, "\n")
        text = text.replace(cls._TABLE_BOUNDARY, "\n\n")
        return text

    @staticmethod
    def _defragment_newlines(text: str) -> str:
        """A line break that does NOT follow sentence-ending punctuation is
        a fake line-wrap (PDF-extraction artifact) - merge it into the
        previous line. A line break that DOES follow sentence-ending punctuation is a real paragraph/sentence boundary - leave it alone.
        """
        placeholder = "\uE000"
        text = text.replace("\n\n* ", placeholder)
        pattern = re.compile(r"(?<![" + re.escape(SENTENCE_END_CHARS) + r"])\n+")
        text = pattern.sub(" ", text)
        return text.replace(placeholder, "\n\n* ")

    @staticmethod
    def _force_paragraphs(text: str) -> str:
        """A line break that DOES follow sentence-ending punctuation is a
        real paragraph/sentence boundary - promote it to a double newline
        so the chunker's paragraph-first rule can find it."""
        pattern = re.compile(r"([" + re.escape(SENTENCE_END_CHARS) + r"])\n+")
        return pattern.sub(r"\1\n\n", text)

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        text = re.sub(r"[ \t]+", " ", text)          # collapse repeated spaces/tabs
        text = re.sub(r"\n{3,}", "\n\n", text)        # cap blank-line runs at one
        text = re.sub(r"[ \t]+\n", "\n", text)        # strip trailing spaces per line
        return text


def strip_repeated_header_footer(pages: list[str], min_repeat_ratio: float = 0.5) -> list[str]:
    """Defensive, general-purpose header/footer stripper: drops a page's
    first/last non-empty line if that same line (or a pure page-number
    line) repeats across a large enough fraction of the document's pages.

    A no-op on the current week7 sample set (no repeated boilerplate was
    found there), but kept in as a general safeguard for future PDFs that
    DO carry running headers or "Page X of Y" footers.
    """
    if len(pages) < 2:
        return pages

    def edges(p: str) -> tuple[str, str]:
        lines = [l.strip() for l in p.split("\n") if l.strip()]
        return (lines[0] if lines else "", lines[-1] if lines else "")

    firsts = [edges(p)[0] for p in pages]
    lasts = [edges(p)[1] for p in pages]
    threshold = max(2, int(len(pages) * min_repeat_ratio))

    def is_boilerplate(line: str, pool: list[str]) -> bool:
        if not line:
            return False
        if re.fullmatch(r"(page\s*)?\d+(\s*/\s*\d+)?", line, re.IGNORECASE):
            return True  # pure page number, always stripped regardless of count
        return pool.count(line) >= threshold

    cleaned_pages = []
    for page, first, last in zip(pages, firsts, lasts):
        lines = page.split("\n")
        if lines and is_boilerplate(first, firsts):
            lines = [l for l in lines if l.strip() != first][0:] if False else lines
        # remove the exact first/last matching lines only (not every
        # occurrence in the body), to avoid accidentally stripping a
        # legitimately repeated sentence elsewhere in the text.
        stripped = list(lines)
        if stripped and stripped[0].strip() == first and is_boilerplate(first, firsts):
            stripped = stripped[1:]
        if stripped and stripped[-1].strip() == last and is_boilerplate(last, lasts):
            stripped = stripped[:-1]
        cleaned_pages.append("\n".join(stripped))
    return cleaned_pages
