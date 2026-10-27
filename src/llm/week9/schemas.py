"""Structured outputs the LLM must return (Pydantic). Kept flat and simple on purpose: Gemini turns these into a
response schema, and unions/nesting are where providers differ."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RagAnswer(BaseModel):
    """Answer to one question from numbered context passages."""

    status: Literal["answered", "not_found"] = Field(
        description="'answered' only if the passages contain the answer; otherwise 'not_found'.")
    answer: str = Field(description="The answer in the language of the question (the fixed 'not found' sentence if not_found).")
    citations: list[int] = Field(description="Numbers of the passages the answer relies on; empty when not_found.")
    language: Literal["en", "ar"] = Field(description="Language of the question and of the answer.")


class SalesInsights(BaseModel):
    """A short summary and a few insights about retail sales. Every number must come from the FACTS given to the model."""

    summary: str = Field(description="2 to 3 sentences.")
    insights: list[str] = Field(description="3 to 5 insights, each supported by the FACTS.")
    caveats: list[str] = Field(description="Limits of the conclusion (short period, partial data...). May be empty.")


class KeyPoint(BaseModel):
    text: str = Field(description="One point stated in the passages.")
    sources: list[int] = Field(description="Numbers of the passages the point relies on (never empty).")


class DocumentSummary(BaseModel):
    """A summary of numbered passages."""

    summary: str = Field(description="2 to 4 sentences; says so when the passages do not cover the question.")
    key_points: list[KeyPoint] = Field(description="3 to 6 points, each with its source passages. May be empty only if the passages do not cover the question.")
