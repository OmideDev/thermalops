"""Pydantic schemas for admin authentication."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AdminLoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    username: str = Field(..., min_length=1, max_length=128)
    password: str = Field(..., min_length=1, max_length=256)

    @field_validator("username", "password")
    @classmethod
    def strip_nonempty(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("must be a string")
        stripped = value.strip()
        if not stripped:
            raise ValueError("must be non-empty")
        return stripped


class AdminLoginResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    username: str
    csrf_token: str
    expires_at: str


class AdminMeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    username: str
    session_id: str
    expires_at: str
    # Session-scoped CSRF secret for post-refresh mutations (memory only in UI).
    csrf_token: str


class AdminLogoutResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool = True
