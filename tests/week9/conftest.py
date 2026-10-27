from dataclasses import dataclass

import pytest


@dataclass
class FakeChunk:
    """Same fields as src.rag.week8.retriever.RetrievedChunk (the generator and the service only read them)."""
    text: str
    original_filename: str = "invoice_2024-02_projectBeta_cloud.txt"
    document_id: str = "doc-1"
    chunk_id: str = "c-1"
    chunk_index: int = 0
    score: float = 0.9
    section_header: str | None = None
    document_type: str | None = "invoice"
    month: str | None = "2024-02"
    project: str | None = "projectBeta"
    service: str | None = "cloud"


@pytest.fixture
def chunks():
    return [
        FakeChunk("Total amount due: 1,250.00 USD. Payment is due within 30 days.", chunk_id="c-1", score=0.91),
        FakeChunk("Company: CloudTech Solutions. Invoice date: 2024-02-10.", chunk_id="c-2", chunk_index=1, score=0.55),
    ]
