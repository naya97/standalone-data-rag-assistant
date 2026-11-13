"""Stage 3: the router's rules, as plain functions (no model, no I/O). Each returns a short REASON CODE or None (= fine).
The nodes call them; the evaluation compares the same codes with the expected ones in data/eval/week11/router_cases.json."""
from __future__ import annotations

import re

# The limit the API already uses for a question (src/api/rag_controller.py: Field(..., max_length=1000)). A protection against
# huge inputs, not a value tuned on the evaluation set: the longest question of the evaluation sets is 93 characters.
MAX_QUESTION_CHARS = 1000

_ARABIC = re.compile(r"[\u0600-\u06FF]")


def check_question(question) -> str | None:
    """The input guard: runs BEFORE the model, so a bad input never costs a request."""
    if not isinstance(question, str) or not question.strip():
        return "empty_question"
    if len(question) > MAX_QUESTION_CHARS:
        return "question_too_long"
    return None


def check_result(tool: str | None, result, status: str | None) -> str | None:
    """Is what the tool returned usable for an answer?
    An EMPTY list of passages is valid on purpose: 'the documents do not contain it' is a correct answer (stage 4 words it),
    not a failure. A sales result with no transactions has nothing to say, so it is 'no_data'."""
    if status == "rejected":
        return "rejected_call"
    if status != "ok" or not isinstance(result, dict):
        return "tool_error"
    if tool == "get_sales_summary":
        overall = result.get("overall")
        if not isinstance(overall, dict):
            return "tool_error"
        if not overall.get("transactions"):
            return "no_data"
    elif tool == "search_documents":
        if not isinstance(result.get("passages"), list):
            return "tool_error"
    return None


# ----------------------------------------------------------------------------------------------- the fallback message
_REQUEST = {"en": "I could not turn this question into a valid request. Please rephrase it with more detail.",
            "ar": "لم أتمكن من تحويل هذا السؤال إلى طلب صالح. الرجاء إعادة صياغته بتفصيل أكثر."}
_UNAVAILABLE = {"en": "The language model is not available right now, so I cannot answer. Please try again later.",
                "ar": "نموذج اللغة غير متاح حالياً، لذلك لا أستطيع الإجابة. الرجاء المحاولة لاحقاً."}

FALLBACK_MESSAGES = {
    "empty_question": {"en": "Please type a question.", "ar": "الرجاء كتابة سؤال."},
    "question_too_long": {"en": f"The question is too long (the limit is {MAX_QUESTION_CHARS} characters). Please shorten it.",
                          "ar": f"السؤال طويل جداً (الحد الأقصى {MAX_QUESTION_CHARS} حرف). الرجاء اختصاره."},
    "no_tool": {"en": "I can only answer questions about the project documents (invoices, policies, reports) or about the sales "
                      "data. This question is outside that scope, asks for something those sources do not contain, or is not clear enough: "
                      "please rephrase it.",
                "ar": "أستطيع فقط الإجابة عن الأسئلة المتعلقة بمستندات المشروع (الفواتير والسياسات والتقارير) أو ببيانات المبيعات. "
                      "هذا السؤال خارج هذا النطاق، أو يطلب شيئاً لا تحتويه هذه المصادر، أو غير واضح بما يكفي: الرجاء إعادة صياغته."},
    "unknown_tool": _REQUEST,
    "rejected_call": _REQUEST,
    "tool_error": {"en": "The lookup failed for this request (for example, a category or a name that does not exist). "
                         "Please check the names in your question and try again.",
                   "ar": "فشل البحث عن هذا الطلب (مثلاً فئة أو اسم غير موجود). الرجاء التحقق من الأسماء في سؤالك والمحاولة مرة أخرى."},
    "no_data": {"en": "The sales data has no transactions for this request.", "ar": "لا توجد معاملات في بيانات المبيعات لهذا الطلب."},
    "model_unavailable": _UNAVAILABLE,
    "model_failed": {"en": "The model request failed. Please try again later.", "ar": "فشل طلب النموذج. الرجاء المحاولة لاحقاً."},
}


def fallback_message(reason: str | None, question: str | None) -> str:
    """A fixed message per reason, in the language of the question. The technical detail (the tool's error) is never shown
    here: it is in the log and in the state."""
    language = "ar" if _ARABIC.search(question or "") else "en"
    return FALLBACK_MESSAGES.get(reason or "", FALLBACK_MESSAGES["no_tool"])[language]
