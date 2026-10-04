
from __future__ import annotations

import logging
import os
import re

logger = logging.getLogger(__name__)

# Whole-filename (minus extension) match, no partial credit:
#   type_YYYY-MM_project[_service]
CONVENTION_PATTERN = re.compile(
    r"^(?P<type>[A-Za-z]+)_(?P<month>\d{4}-\d{2})_(?P<project>[^_]+)(?:_(?P<service>[^_]+))?$"
)


def extract_metadata_from_filename(filename: str) -> dict:
    """Extract metadata from a filename based on the naming convention."""
    
    base, ext = os.path.splitext(filename)
    ext = ext.lstrip(".").lower()

    metadata = {
        "document_name": filename,
        "original_filename": filename,
        "source_format": ext,
        "document_type": None,
        "month": None,
        "project": None,
        "service": None,
        "classification_status": "unclassified",
    }

    match = CONVENTION_PATTERN.match(base)
    if not match:
        logger.info(
            "'%s' does not match the naming convention - ingesting as unclassified "
            "(this does not block loading/chunking/storage).",
            filename,
        )
        return metadata

    metadata.update(
        {
            "document_type": match.group("type").lower(),
            "month": match.group("month"),
            "project": match.group("project"),
            "service": match.group("service") or "general",
            "classification_status": "classified",
        }
    )
    return metadata
