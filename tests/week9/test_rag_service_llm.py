"""RagService + LLMGenerator end to end, with a fake retriever and gate (no embedding model, no index, no API)."""

import json
from types import SimpleNamespace

from src.llm.week9.client import FakeLLMClient, LLMTimeoutError
from src.rag.week8.answer_generator import ExtractiveGenerator
from src.rag.week8.prompt_builder import NO_ANSWER
from src.rag.week8.rag_service import RagService
from src.rag.week9.llm_generator import LLMGenerator


class FakeRetriever:
    def __init__(self, chunks, status="retrieved"):
        self.chunks, self.status = chunks, status

    def retrieve(self, question, top_k=None, filters=None):
        chunks = [] if self.status == "no_documents_match_filter" else self.chunks
        return SimpleNamespace(status=self.status, chunks=chunks, applied_filters=filters or {}, stale_chunks=0)


class FakeChecker:
    def __init__(self, answerable=True, threshold=0.33):
        self.answerable, self.min_lexical_coverage, self.calls = answerable, threshold, 0

    def check(self, question, chunks):
        self.calls += 1
        decision = {"answerable": self.answerable, "reasons": [] if self.answerable else ["lexical coverage too low"],
                    "signals": {"top1_score": 0.9}}
        return SimpleNamespace(answerable=self.answerable, reasons=list(decision["reasons"]),
                               to_dict=lambda: dict(decision, reasons=list(decision["reasons"])))


def reply(**kw):
    base = {"status": "answered", "answer": "The total amount due is 1250 USD.", "citations": [1], "language": "en"}
    return json.dumps({**base, **kw})


def service(chunks, script, answerable=True, status="retrieved", strict=None):
    client = FakeLLMClient(script)
    generator = LLMGenerator(client, ExtractiveGenerator())
    return RagService(FakeRetriever(chunks, status), FakeChecker(answerable), generator, strict), client


def test_answered_with_llm(chunks):
    svc, _ = service(chunks, [reply()])
    r = svc.ask("What is the total amount due?", include_prompt=True)
    assert r.status == "answered" and r.answer == "The total amount due is 1250 USD."
    assert [c["citation_id"] for c in r.citations] == [1] and r.citations[0]["chunk_id"] == "c-1"
    assert r.generation["mode"] == "llm" and r.generation["fallback_reason"] is None and r.generation["model"] == "fake-model"
    assert r.prompt is not None and r.retrieved is not None  # the Week 8 debug switch still works


def test_llm_not_found_becomes_no_relevant_context(chunks):
    svc, _ = service(chunks, [reply(status="not_found", answer="x", citations=[])])
    r = svc.ask("Who approved the invoice?")
    assert r.status == "no_relevant_context" and r.answer == NO_ANSWER["en"] and r.citations == []
    assert any(reason.startswith("llm_not_found") for reason in r.answerability["reasons"])
    assert r.answerability["answerable"] is True  # that is what the gate said; the LLM overruled it
    assert r.generation["mode"] == "llm"


def test_llm_failure_still_answers_like_week8(chunks):
    svc, _ = service(chunks, [LLMTimeoutError("slow")])
    r = svc.ask("What is the total amount due?")
    assert r.status == "answered" and r.answer.startswith(chunks[0].text.strip())
    assert r.generation["mode"] == "extractive_fallback" and "LLMTimeoutError" in r.generation["fallback_reason"]


def test_gate_rejection_never_calls_the_llm(chunks):
    svc, client = service(chunks, [reply()], answerable=False)
    r = svc.ask("How do I bake bread?")
    assert r.status == "no_relevant_context" and r.generation is None and client.calls == []


def test_filter_without_match_never_calls_the_llm(chunks):
    svc, client = service(chunks, [reply()], status="no_documents_match_filter")
    r = svc.ask("anything", filters={"project": "nope"})
    assert r.status == "no_documents_match_filter" and r.generation is None and client.calls == []


def test_extractive_service_reports_its_mode(chunks):
    svc = RagService(FakeRetriever(chunks), FakeChecker(), ExtractiveGenerator())
    assert svc.ask("What is the total amount due?").generation["mode"] == "extractive"


def test_response_serialises(chunks):
    svc, _ = service(chunks, [reply()])
    payload = svc.ask("What is the total amount due?").to_dict()
    assert payload["generation"]["input_tokens"] > 0 and json.dumps(payload)


# ----------------------------------------------------------------------------------------------- two-tier gate
def test_a_fallback_answer_is_judged_by_the_stricter_gate(chunks):
    """The hole: lenient gate lets a question through because the LLM will read it; the LLM then fails (429/504),
    and the extractive fallback must NOT quote a chunk for a question only the lenient gate accepted."""
    strict = FakeChecker(answerable=False, threshold=0.5)
    svc, _ = service(chunks, [LLMTimeoutError("504")], strict=strict)
    r = svc.ask("كم راتب المدرب في المنصة؟")
    assert r.status == "no_relevant_context" and r.citations == []
    assert r.generation["mode"] == "extractive_fallback" and "LLMTimeoutError" in r.generation["fallback_reason"]
    assert r.answerability["answerable"] is False
    assert r.answerability["reasons"][0].startswith("stricter gate for a verbatim answer (mode=extractive_fallback)")


def test_a_fallback_answer_is_returned_when_it_also_passes_the_stricter_gate(chunks):
    svc, _ = service(chunks, [LLMTimeoutError("504")], strict=FakeChecker(answerable=True, threshold=0.5))
    r = svc.ask("What is the total amount due?")
    assert r.status == "answered" and r.generation["mode"] == "extractive_fallback" and r.answer.startswith(chunks[0].text.strip())


def test_the_stricter_gate_is_not_used_when_the_llm_answers(chunks):
    strict = FakeChecker(answerable=False, threshold=0.5)
    svc, _ = service(chunks, [reply()], strict=strict)
    assert svc.ask("What is the total amount due?").status == "answered" and strict.calls == 0


def test_the_stricter_gate_is_not_used_when_the_llm_says_not_found(chunks):
    strict = FakeChecker(answerable=False, threshold=0.5)
    svc, _ = service(chunks, [reply(status="not_found", citations=[])], strict=strict)
    assert svc.ask("Who approved it?").status == "no_relevant_context" and strict.calls == 0


def test_extractive_mode_is_judged_by_the_stricter_gate_too(chunks):
    strict = FakeChecker(answerable=False, threshold=0.5)
    svc = RagService(FakeRetriever(chunks), FakeChecker(), ExtractiveGenerator(), strict)
    r = svc.ask("What is the total amount due?")
    assert r.status == "no_relevant_context" and r.generation["mode"] == "extractive" and strict.calls == 1


def test_no_stricter_gate_means_week8_behaviour(chunks):
    svc, _ = service(chunks, [LLMTimeoutError("504")], strict=None)
    assert svc.ask("What is the total amount due?").status == "answered"


# ----------------------------------------------------------------------------------------------- include_prompt
def test_include_prompt_shows_what_was_really_sent_to_the_llm(chunks):
    svc, client = service(chunks, [reply()])
    r = svc.ask("What is the total amount due?", include_prompt=True)
    assert r.prompt.startswith("### SYSTEM") and "<passage id=\"1\"" in r.prompt
    assert r.prompt == f"### SYSTEM\n{client.calls[0]['system']}\n\n### USER\n{client.calls[0]['user']}"


def test_include_prompt_is_also_returned_when_the_llm_says_not_found(chunks):
    svc, _ = service(chunks, [reply(status="not_found", citations=[])])
    r = svc.ask("Who approved it?", include_prompt=True)
    assert r.status == "no_relevant_context" and r.prompt.startswith("### SYSTEM") and r.retrieved


def test_include_prompt_without_an_llm_keeps_the_week8_prompt(chunks):
    svc = RagService(FakeRetriever(chunks), FakeChecker(), ExtractiveGenerator())
    assert "Answer:" in svc.ask("What is the total amount due?", include_prompt=True).prompt


def test_nothing_is_shown_when_nothing_was_sent(chunks):
    svc, client = service(chunks, [reply()], answerable=False)
    assert svc.ask("How do I bake bread?", include_prompt=True).prompt is None and client.calls == []


def test_prompt_is_off_by_default(chunks):
    svc, _ = service(chunks, [reply()])
    assert svc.ask("What is the total amount due?").prompt is None
