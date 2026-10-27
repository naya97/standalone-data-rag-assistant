"""The lexical gate on Arabic: what the Week 8 behaviour does, what the light mode changes, what it must not change."""

import pytest

from src.rag.week8.answerability import (
    AnswerabilityChecker, content_stems, lexical_coverage, normalize_arabic,
)
from src.rag.week9.gate_experiment import (
    build_variants, changed_vs, classify, compare_variants, format_report, tally, variant_name,
)

# the real failing case: the chunk that answers the question, and the question as typed
CHUNK = ("* تجرى المقابلات يوم الأحد بتاريخ 6/7على أن يتم إرسال تقرير مختصرعن الوظيفة قبل نهاية 5/7.\n\n"
         "* يجب العمل ضمن مجموعات الوظيفة الأولى ذاتها. بالتوفيق... د.باسم قصيبة م.آية معطي م.أميرأبو الشعر")
QUESTION = "متى سيتم اجراء المقابلات؟"
FILE = "الوظيفة الثانية (1).pdf"


# ----------------------------------------------------------------------------------------------- the helpers
def test_week8_behaviour_is_unchanged_by_default():
    assert content_stems(QUESTION) == {"مقابل", "اجراء", "سيتم"}
    assert lexical_coverage(QUESTION, CHUNK) == pytest.approx(1 / 3)  # the false refusal, documented


def test_light_mode_drops_the_future_marker_word_and_nothing_else():
    assert content_stems(QUESTION, arabic_light=True) == {"مقابل", "اجراء"}
    assert lexical_coverage(QUESTION, CHUNK, arabic_light=True) == pytest.approx(0.5)  # passes only AT the threshold


@pytest.mark.parametrize("a, b", [("اجراء", "إجراء"), ("المقابلة", "المقابلـة"), ("المُقابَلات", "المقابلات"),
                                  ("مدرسة", "مدرسه"), ("على", "علي"), ("أحمد", "احمد"), ("آلية", "الية")])
def test_spelling_variants_become_one_word(a, b):
    assert normalize_arabic(a) == normalize_arabic(b)
    assert content_stems(a, True) == content_stems(b, True)


def test_normalisation_is_off_unless_asked():
    assert content_stems("إجراء") != content_stems("اجراء")
    assert content_stems("إجراء", True) == content_stems("اجراء", True)


@pytest.mark.parametrize("text", ["When are the interviews held?", "What is the total amount due?",
                                  "Project Beta storage cost February 2024"])
def test_english_is_identical_in_both_modes(text):
    assert content_stems(text, True) == content_stems(text, False)


def test_existing_arabic_stop_words_still_work_in_light_mode():
    assert content_stems("متى إلى الى عن", True) == set()


# ----------------------------------------------------------------------------------------------- the checker
class Chunk:
    def __init__(self, text, filename=FILE, score=0.64):
        self.text, self.original_filename, self.score = text, filename, score


def checker(light):
    return AnswerabilityChecker(vocabulary=set(), unknown_entity_check=False, arabic_light=light)


def test_the_failing_question_is_refused_by_default_and_accepted_in_light_mode():
    chunks = [Chunk(CHUNK)]
    refused = checker(False).check(QUESTION, chunks)
    assert not refused.answerable and refused.signals["lexical_coverage"] == pytest.approx(0.3333)
    accepted = checker(True).check(QUESTION, chunks)
    assert accepted.answerable and accepted.signals["thresholds"]["arabic_light"] is True


@pytest.mark.parametrize("question", ["كيف أخبز الخبز؟", "من وافق على الفاتورة؟", "كم سعر الذهب اليوم؟"])
def test_off_topic_arabic_questions_are_still_refused_in_light_mode(question):
    assert not checker(True).check(question, [Chunk(CHUNK)]).answerable


# ----------------------------------------------------------------------------------------------- the experiment
def questions():
    return [
        {"id": 1, "question": QUESTION, "should_reject": False, "expected_sources": ["الوظيفة*"], "split": "dev", "kind": "paraphrase"},
        {"id": 2, "question": "كيف أخبز الخبز؟", "should_reject": True, "split": "dev", "kind": "off_topic"},
        {"id": 3, "question": "What is the total amount due?", "should_reject": False, "expected_sources": ["invoice_*"], "split": "test", "kind": "direct"},
        {"id": 4, "question": "anything", "should_reject": False, "expected_sources": ["x"], "filters": {"project": "none"}},
    ]


def fake_retrieve(q):
    return {1: [Chunk(CHUNK)], 2: [Chunk(CHUNK)],
            3: [Chunk("Total amount due: 1,250.00 USD. Payment due within 30 days.", "invoice_2024-02_projectBeta_cloud.txt", 0.8)],
            4: None}[q["id"]]


@pytest.fixture
def experiment():
    cfg = type("Cfg", (), {"min_score": 0.0, "min_lexical_coverage": 0.5, "unknown_entity_check": False,
                           "min_score_cross_lingual": 0.30})()
    variants = build_variants(cfg, set())
    return compare_variants(questions(), fake_retrieve, variants), variant_name(0.5, False), variant_name(0.5, True)


def test_grid_contains_the_current_threshold_and_both_modes(experiment):
    result, base, cand = experiment
    assert base in result["variants"] and cand in result["variants"] and len(result["variants"]) == 10


def test_questions_without_documents_are_skipped_not_judged(experiment):
    assert experiment[0]["skipped"] == [4] and len(experiment[0]["rows"]) == 3


def test_outcomes_per_variant(experiment):
    result, base, cand = experiment
    by_id = {r["id"]: r for r in result["rows"]}
    assert by_id[1]["outcomes"][base] == "false_refusal" and by_id[1]["outcomes"][cand] == "ok"
    assert by_id[2]["outcomes"][base] == by_id[2]["outcomes"][cand] == "correct_rejection"
    assert by_id[3]["outcomes"][base] == by_id[3]["outcomes"][cand] == "ok"
    assert by_id[1]["group"] == "ar->ar" and by_id[3]["group"] == "en->en"


def test_tally_and_changed_decisions(experiment):
    result, base, cand = experiment
    focus = tally(result["rows"], base, "ar->ar")
    assert (focus["answerable"], focus["unanswerable"], focus["false_refusal"], focus["leak"]) == (1, 1, 1, 0)
    assert tally(result["rows"], cand, "ar->ar")["false_refusal"] == 0
    assert [(c["id"], c["from"], c["to"]) for c in changed_vs(result["rows"], base, cand)] == [(1, "false_refusal", "ok")]


def test_a_lower_threshold_can_leak_and_the_table_shows_it():
    chunk = Chunk("تجرى المقابلات يوم الأحد", FILE)
    q = {"id": 9, "question": "متى موعد المقابلات في الجامعة الاردنية؟", "should_reject": True, "split": "dev"}
    cfg = type("Cfg", (), {"min_score": 0.0, "min_lexical_coverage": 0.5, "unknown_entity_check": False,
                           "min_score_cross_lingual": 0.30})()
    result = compare_variants([q], lambda _: [chunk], build_variants(cfg, set()))
    outcomes = result["rows"][0]["outcomes"]
    assert outcomes[variant_name(0.5, False)] == "correct_rejection" and outcomes[variant_name(0.25, False)] == "leak"


def test_the_report_states_how_small_the_evidence_is(experiment):
    result, base, cand = experiment
    text = format_report(result, base, cand)
    assert "<- current" in text and "ar->ar" in text and "not evidence" in text
    assert "Q1" in text and "false_refusal -> ok" in text


def test_classify_flags_a_pass_from_the_wrong_document():
    class D: answerable = True
    q = {"should_reject": False, "expected_sources": ["invoice_*"]}
    assert classify(q, D(), "invoice_2024.txt") == "ok" and classify(q, D(), "other.pdf") == "wrong_source"


# ----------------------------------------------------------------------------------------------- two-tier gate
def test_the_question_the_lenient_gate_accepts_and_the_strict_gate_refuses():
    lenient = AnswerabilityChecker(set(), min_lexical_coverage=0.33, unknown_entity_check=False)
    strict = lenient.with_lexical_coverage(0.5)
    chunks = [Chunk(CHUNK)]
    assert lenient.check(QUESTION, chunks).answerable          # 1 of 3 key words: enough to reach the LLM
    refused = strict.check(QUESTION, chunks)
    assert not refused.answerable and "minimum 50%" in refused.reasons[0]  # not enough to quote a chunk by itself


def test_the_strict_copy_keeps_every_other_setting():
    lenient = AnswerabilityChecker({"a"}, 0.1, 0.33, False, 0.4, True)
    strict = lenient.with_lexical_coverage(0.5)
    assert strict.vocabulary is lenient.vocabulary and strict.min_lexical_coverage == 0.5
    assert (strict.min_score, strict.unknown_entity_check, strict.min_score_cross_lingual, strict.arabic_light) == (0.1, False, 0.4, True)
    assert lenient.min_lexical_coverage == 0.33  # the original is untouched


class _Store:
    def get_all_chunks(self):
        return [{"chunk_text": "Total amount due"}]


def _cfg(main, extractive):
    from types import SimpleNamespace
    return SimpleNamespace(min_score=0.0, min_lexical_coverage=main, unknown_entity_check=True,
                           min_score_cross_lingual=0.3, arabic_light=False, extractive_min_lexical_coverage=extractive)


@pytest.mark.parametrize("main, extractive, expected_strict", [
    (0.33, 0.5, 0.5),    # lenient in front of the LLM, strict for quoting a chunk
    (0.5, 0.5, None),    # same value: one gate is enough
    (0.6, 0.5, None),    # the strict gate is never looser than the main one
    (0.33, 0.33, None),
])
def test_build_checkers(main, extractive, expected_strict):
    from src.rag.week8.rag_service import build_checkers
    checker, strict = build_checkers(_cfg(main, extractive), _Store())
    assert checker.min_lexical_coverage == main
    assert (strict.min_lexical_coverage if strict else None) == expected_strict


def test_extractive_threshold_is_read_from_the_environment_and_validated(monkeypatch):
    from src.rag.week8 import config as rag_config
    monkeypatch.setattr(rag_config, "load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("RAG_EXTRACTIVE_MIN_LEXICAL_COVERAGE", raising=False)
    assert rag_config.get_config().extractive_min_lexical_coverage == 0.5
    monkeypatch.setenv("RAG_EXTRACTIVE_MIN_LEXICAL_COVERAGE", "0.4")
    assert rag_config.get_config().extractive_min_lexical_coverage == 0.4
    monkeypatch.setenv("RAG_EXTRACTIVE_MIN_LEXICAL_COVERAGE", "1.5")
    with pytest.raises(ValueError, match="RAG_EXTRACTIVE_MIN_LEXICAL_COVERAGE"):
        rag_config.get_config()
