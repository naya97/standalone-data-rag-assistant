from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:  # pragma: no cover
    from src.rag.week7.db_store import RagStore
    from src.rag.week8.config import RagConfig
    from src.rag.week8.retriever import RetrievedChunk

EN_STOP = set('''a an the of in on at to for from by with and or is are was were be been being what which who whom
whose when where why how do does did can could should would will shall may might must it its this that these those
there their they them we our you your i me my as if than then so not no into over under about per any all each have
has had'''.split())
AR_STOP = set('متى ما ماذا من كم هل هو هي في على إلى الى عن مع كيف أين لماذا التي الذي ذلك هذا هذه'.split())
STOP = EN_STOP | AR_STOP

# ----------------------------------------------------------------------------------------------- Week 9: light Arabic mode
# Off by default (RAG_ARABIC_LIGHT=False): the Week 8 behaviour is unchanged until the gate experiment
# (run_gate_experiment.py) shows that turning it on removes false refusals without adding leaks.
#   1. spelling normalisation: removes diacritics/tatweel and unifies alef/hamza forms, ى/ي, ة/ه
#      ("اجراء" and "إجراء" become the same word)
#   2. function words: auxiliaries, future/passive markers, negation, dialect question words. Chosen by CATEGORY
#      (the Arabic counterparts of the English "will, be, do, not, what" already in EN_STOP), NOT by picking words
#      that rescue particular questions.
_AR_MARKS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u0640]")
_AR_LETTERS = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه"})
AR_FUNCTION_WORDS = set('''سيتم يتم تم سوف قد كان كانت يكون تكون لم لن لا اي كل بين حتى عند لدى ثم ان انه هنا هناك
الذين اللذين شو ايش وين ليش قديش مين'''.split())


def normalize_arabic(text: str) -> str:
    return _AR_MARKS.sub("", text).translate(_AR_LETTERS)


STOP_LIGHT = {normalize_arabic(w) for w in STOP} | {normalize_arabic(w) for w in AR_FUNCTION_WORDS}


def content_stems(text: str, arabic_light: bool = False) -> set[str]:
    if arabic_light:
        text, stop = normalize_arabic(text), STOP_LIGHT
    else:
        stop = STOP
    stems = set()
    for tok in re.findall(r'\w+', text.lower()):
        if tok in stop or len(tok) < 2:
            continue
        if tok.startswith('ال') and len(tok) > 3:
            tok = tok[2:]
        stems.add(tok[:5])
    return stems


def lexical_coverage(question: str, chunk_text: str, arabic_light: bool = False) -> float:
    q = content_stems(question, arabic_light)
    return len(q & content_stems(chunk_text, arabic_light)) / len(q) if q else float('nan')


_ARABIC_LETTER = re.compile(r"[\u0600-\u06FF]")
_LATIN_LETTER = re.compile(r"[A-Za-z]")


def script_of(text: str) -> str:
    """'ar' if Arabic letters outnumber Latin letters, else 'en'. Used to detect cross-lingual question/chunk pairs."""
    return "ar" if len(_ARABIC_LETTER.findall(text)) > len(_LATIN_LETTER.findall(text)) else "en"


def build_vocabulary(texts: Iterable[str]) -> set[str]:
    return {t for text in texts for t in re.findall(r'\w+', text.lower())}


def unknown_entities(question: str, vocabulary: set[str]) -> list[str]:
    toks = re.findall(r'\w+', question)
    return [t for i, t in enumerate(toks)
            if i > 0 and t[:1].isupper() and t.lower() not in STOP and t.lower() not in vocabulary]


@dataclass
class AnswerabilityDecision:
    answerable: bool
    reasons: list[str] = field(default_factory=list)  # why it was rejected (empty when answerable)
    signals: dict = field(default_factory=dict)       # always filled: makes every decision explainable

    def to_dict(self) -> dict:
        return asdict(self)


class AnswerabilityChecker:
    def __init__(self, vocabulary: set[str], min_score: float = 0.0,
                 min_lexical_coverage: float = 0.5, unknown_entity_check: bool = True,
                 min_score_cross_lingual: float = 0.30, arabic_light: bool = False):
        self.vocabulary = vocabulary
        self.min_score = min_score
        self.min_lexical_coverage = min_lexical_coverage
        self.unknown_entity_check = unknown_entity_check
        self.min_score_cross_lingual = min_score_cross_lingual
        self.arabic_light = arabic_light  # Week 9

    @classmethod
    def from_config(cls, config: "RagConfig", rag_store: "RagStore") -> "AnswerabilityChecker":
        """The vocabulary is the text of the whole corpus. It is built once: restart the service after re-ingesting."""
        vocabulary = build_vocabulary(c["chunk_text"] for c in rag_store.get_all_chunks())
        return cls(vocabulary, config.min_score, config.min_lexical_coverage, config.unknown_entity_check,
                   config.min_score_cross_lingual, getattr(config, "arabic_light", False))

    def with_lexical_coverage(self, value: float) -> "AnswerabilityChecker":
        """Same checker, another lexical threshold (Week 9: the stricter gate for extractive answers)."""
        return AnswerabilityChecker(self.vocabulary, self.min_score, value, self.unknown_entity_check,
                                    self.min_score_cross_lingual, self.arabic_light)

    def check(self, question: str, chunks: list["RetrievedChunk"]) -> AnswerabilityDecision:
        thresholds = {"min_score": self.min_score, "min_lexical_coverage": self.min_lexical_coverage,
                      "unknown_entity_check": self.unknown_entity_check,
                      "min_score_cross_lingual": self.min_score_cross_lingual,
                      "arabic_light": self.arabic_light}
        if not chunks:
            return AnswerabilityDecision(False, ["no chunks were retrieved"], {"thresholds": thresholds})

        top1 = chunks[0].score
        gap = top1 - chunks[1].score if len(chunks) > 1 else None
        lex = lexical_coverage(question, chunks[0].text, self.arabic_light)
        entities = unknown_entities(question, self.vocabulary)
        cross_lingual = script_of(question) != script_of(chunks[0].text)

        reasons = []
        if top1 < self.min_score:
            reasons.append(f"top-1 score {top1:.3f} is below the minimum {self.min_score:.3f}")
        if cross_lingual:
            # no shared words across scripts: lexical coverage cannot be measured, so it is not applied
            if top1 < self.min_score_cross_lingual:
                reasons.append(f"cross-language question and best chunk: similarity {top1:.3f} is below the "
                               f"cross-language minimum {self.min_score_cross_lingual:.3f}")
        elif not math.isnan(lex) and lex < self.min_lexical_coverage:
            reasons.append(f"only {lex:.0%} of the question's key words appear in the best chunk "
                           f"(minimum {self.min_lexical_coverage:.0%})")
        if self.unknown_entity_check and entities:
            reasons.append(f"the question mentions {entities} which appear nowhere in the documents")

        signals = {
            "top1_score": round(top1, 4),
            "score_gap": None if gap is None else round(gap, 4),
            "lexical_coverage": None if math.isnan(lex) else round(lex, 4),
            "lexical_gate_applied": not cross_lingual,
            "cross_lingual": cross_lingual,
            "unknown_entities": entities,
            "thresholds": thresholds,
        }
        return AnswerabilityDecision(not reasons, reasons, signals)
