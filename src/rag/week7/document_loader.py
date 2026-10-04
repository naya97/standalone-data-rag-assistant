from pathlib import Path
from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


class DocumentLoaderError(Exception):
    """Base class for any error raised while loading a document. Callers
    that just want to catch "this file couldn't be loaded, skip it and
    move on" can catch this single type."""


class UnsupportedFileFormatError(DocumentLoaderError):
    """Raised when a file extension is not supported by the loader."""


class EmptyDocumentError(DocumentLoaderError):
    """Raised when a file is readable but contains no usable text."""


def load_document(file_path: str | Path) -> str:
    """Load a document and return its raw text content.

    Args:
        file_path: path to a .txt, .md or .pdf file.

    Returns:
        The raw extracted text (not chunked, not cleaned).

    Raises:
        FileNotFoundError: if the path does not exist.
        UnsupportedFileFormatError: if the extension isn't .txt/.md/.pdf.
        EmptyDocumentError: if the file has no extractable text.
    """
    path = Path(file_path)
    if not path.exists():
        raise DocumentLoaderError(f"Document not found: {path}")

    ext = path.suffix.lower()
    if ext in (".txt", ".md"):
        return _load_text(path)
    elif ext == ".pdf":
        return _load_pdf(path)
    else:
        raise UnsupportedFileFormatError(
            f"Unsupported file format '{ext}' for file: {path.name} "
            f"(supported: {sorted(SUPPORTED_EXTENSIONS)})"
        )


def _load_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise EmptyDocumentError(f"File is empty: {path}")
    return text


def _load_pdf(path: Path) -> str:
    try:
        reader = PdfReader(str(path))
        if len(reader.pages) == 0:
            raise EmptyDocumentError(f"PDF has no pages: {path}")

        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
    except DocumentLoaderError:
        raise
    except Exception as exc:
        # pypdf can raise several different internal error types for a
        # corrupted/invalid file (EOFError, PdfReadError, ...). We normalize
        # all of them to DocumentLoaderError so the ingestion orchestrator
        # can reliably attribute the failure to the Load phase.
        raise DocumentLoaderError(f"Could not read PDF '{path.name}': {exc}") from exc

    text = "\n".join(text_parts).strip()
    if not text:
        raise EmptyDocumentError(
            f"No extractable text found in PDF (possibly scanned/image-based): {path}"
        )

    # Narrow, best-effort enhancement (see table_extractor.py): pypdf stays
    # the primary extractor for everything above. pdfplumber is only
    # invoked here to detect tables and splice in a clean Markdown
    # replacement for the garbled, column-less text pypdf produces for
    # them. A no-op for any document with no table (the common case).
    from src.rag.week7.table_extractor import splice_clean_tables_into_text
    text = splice_clean_tables_into_text(path, text)

    return text


def scan_documents(root_dir: str | Path) -> list[Path]:
    root_dir = Path(root_dir)
    return sorted(
        p for p in root_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
