"""
ONE general algorithm for all document types and all source formats.
No per-document-type chunker classes. Structure-awareness (headers,
paragraphs) is handled as an optional PRE-SEGMENTATION step before the
same universal, size-safety-capped engine runs over each segment.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 300
DEFAULT_OVERLAP = 50

# Used to recognize a "bare section title" unit (see _bond_bare_titles_to_first_item).
SENTENCE_END_CHARS = ".:؟!"

# A header line is only trusted as a real structural boundary when the
# source format is authored by us (.md/.txt) and it starts at a true line
# start (^ with MULTILINE). We deliberately do NOT rely on this for PDFs:
# real-world testing showed pypdf frequently glues numbered headers onto
# the end of the previous line, so the ^ anchor would silently never match.
HEADER_PATTERN = re.compile(r"^(#{1,6}\s+.+|\d{1,2}\.\s+.+)$", re.MULTILINE)


class DocumentChunker:
    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be >= 0 and smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    # -- public API ---------------------------------------------------

    def chunk(self, text: str, allow_header_presplit: bool = False) -> list[dict]:

        text = text.strip()
        if not text:
            return []

        if allow_header_presplit and HEADER_PATTERN.search(text):
            sections = self._split_on_headers(text)  # list[(header|None, text)]
        else:
            sections = [(None, text)]

        results: list[dict] = []
        for header, section_text in sections:
            for piece in self._chunk_with_safety_cap(section_text):
                results.append({"chunk_text": piece, "section_header": header})
        return results

    # -- header pre-segmentation (optional, .md/.txt only) -------------

    def _split_on_headers(self, text: str) -> list[tuple[str | None, str]]:
        matches = list(HEADER_PATTERN.finditer(text))
        sections: list[tuple[str | None, str]] = []
        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            section = text[start:end].strip()
            header = m.group(0).strip()
            if section:
                sections.append((header, section))
        # Text before the first header (e.g. a title/intro line) has no
        # header of its own - tagged as None, not dropped.
        if matches and matches[0].start() > 0:
            preamble = text[: matches[0].start()].strip()
            if preamble:
                sections.insert(0, (None, preamble))
        return sections or [(None, text)]

    # -- universal size-safety-capped engine ---------------------------

    def _chunk_with_safety_cap(self, text: str) -> list[str]:

        units = [u.strip() for u in text.split("\n\n") if u.strip()]
        if not units:
            return []

        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        def flush():
            if current:
                chunks.append("\n\n".join(current))

        for unit in units:
            if len(unit) > self.chunk_size:
                flush()
                current.clear()
                current_len = 0
                chunks.extend(self._split_oversized_unit(unit))
                continue

            joiner_len = 2 if current else 0  # "\n\n" between units
            if current and current_len + joiner_len + len(unit) > self.chunk_size:
                flush()
                # Unit-based overlap: carry the single last unit forward
                # for light context continuity - but ONLY if doing so still
                # leaves room for the new unit within the cap. The safety
                # cap is a hard, non-negotiable guarantee (confirmed
                # explicitly earlier); overlap is a best-effort nicety that
                # must yield to it. An earlier version of this got this
                # backwards and produced a 340-char chunk against a
                # 300-char cap - caught by the safety-cap regression test.
                carried = current[-1]
                if self.overlap > 0 and len(carried) + 2 + len(unit) <= self.chunk_size:
                    current = [carried]
                    current_len = len(carried)
                else:
                    current = []
                    current_len = 0

            joiner_len = 2 if current else 0
            current.append(unit)
            current_len += joiner_len + len(unit)

        flush()
        return chunks

    def _split_oversized_unit(self, text: str) -> list[str]:
       
        chunks: list[str] = []
        start = 0
        n = len(text)

        while start < n:
            end = min(start + self.chunk_size, n)
            if end < n:
                end = self._find_best_cut(text, start, end)

            piece = text[start:end].strip()
            if piece:
                chunks.append(piece)

            if end >= n:
                break

            raw_next_start = end - self.overlap
            next_start = self._snap_to_word_start(text, raw_next_start, hard_limit=end)
            start = next_start if next_start > start else end

        return chunks

    def _find_best_cut(self, text: str, start: int, end: int) -> int:
        # 1) paragraph break
        boundary = text.rfind("\n\n", start, end)
        if boundary > start:
            return boundary
        # 2) sentence end
        boundary = text.rfind(". ", start, end)
        if boundary > start:
            return boundary + 1
        # 3) nearest whitespace (any kind - handles PDFs that use single \n
        #    as their only separator, and Arabic/RTL text)
        boundary = self._rfind_whitespace(text, start, end)
        if boundary > start:
            return boundary
        # 4) last resort - logged, and should be rare in practice
        logger.warning(
            "No natural boundary found in chars %d-%d; falling back to a hard cut.",
            start, end,
        )
        return end

    def _snap_to_word_start(self, text: str, pos: int, hard_limit: int) -> int:
       
        if pos <= 0:
            return 0
        if text[pos - 1].isspace():
            return pos  # already sitting right at a word start

        # Walk forward to the next whitespace, then start right after it.
        m = re.compile(r"\s").search(text, pos, hard_limit)
        if m:
            return m.end()

        # No whitespace between pos and hard_limit (one very long unbroken
        # token, e.g. a URL) -> give up on overlap for this step rather
        # than risk a mid-word start.
        return hard_limit

    @staticmethod
    def _rfind_whitespace(text: str, start: int, end: int) -> int:
        for i in range(end - 1, start, -1):
            if text[i].isspace():
                return i
        return -1
