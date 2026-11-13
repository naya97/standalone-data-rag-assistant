from src.llm.week11.validation import FALLBACK_MESSAGES, MAX_QUESTION_CHARS, check_question, check_result, fallback_message


def test_check_question():
    assert check_question("What is the total revenue?") is None
    assert check_question("x" * MAX_QUESTION_CHARS) is None
    assert check_question("x" * (MAX_QUESTION_CHARS + 1)) == "question_too_long"
    for empty in ("", "  \n ", None, 5):
        assert check_question(empty) == "empty_question"


def test_check_result():
    sales = {"overall": {"transactions": 5}}
    assert check_result("get_sales_summary", sales, "ok") is None
    assert check_result("get_sales_summary", {"overall": {"transactions": 0}}, "ok") == "no_data"
    assert check_result("get_sales_summary", {"scope": {}}, "ok") == "tool_error"
    assert check_result("search_documents", {"passages": [{"id": 1}]}, "ok") is None
    assert check_result("search_documents", {"passages": []}, "ok") is None            # 'not found' is an answer, not a failure
    assert check_result("search_documents", {"error": "x"}, "ok") == "tool_error"
    assert check_result("search_documents", {"error": "x", "rejected": True}, "rejected") == "rejected_call"
    assert check_result("get_sales_summary", {"error": "Unknown category"}, "error") == "tool_error"
    assert check_result("get_sales_summary", None, "ok") == "tool_error"


def test_every_reason_has_both_languages():
    for reason in ("empty_question", "question_too_long", "no_tool", "unknown_tool", "rejected_call", "tool_error", "no_data",
                   "model_unavailable", "model_failed"):
        assert FALLBACK_MESSAGES[reason]["en"] and FALLBACK_MESSAGES[reason]["ar"]


def test_message_language_and_unknown_reason():
    assert fallback_message("no_data", "total sales?").startswith("The sales data")
    assert fallback_message("no_data", "كم المبيعات؟").startswith("لا توجد")
    assert fallback_message(None, "hi") == fallback_message("no_tool", "hi")
    assert "limit is 1000" in fallback_message("question_too_long", "x")
