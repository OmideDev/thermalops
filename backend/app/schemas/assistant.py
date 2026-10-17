"""Pydantic schemas for the experimental grounded assistant API."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.rag.ask import MAX_QUESTION_CHARS


class AssistantAskRequest(BaseModel):
    """POST /api/v1/assistant/ask request body."""

    question: str = Field(..., description="User question (informational only).")

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("question must be a string")
        stripped = value.strip()
        if not stripped:
            raise ValueError("question must be a non-empty string")
        if len(stripped) > MAX_QUESTION_CHARS:
            raise ValueError(
                f"question exceeds maximum length of {MAX_QUESTION_CHARS} characters"
            )
        return stripped


class AssistantSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    title: str
    page: int


class AssistantAskResponse(BaseModel):
    """Matches the Phase 4A AskResponse public contract."""

    model_config = ConfigDict(extra="forbid")

    answer: str
    sources: list[AssistantSource]
