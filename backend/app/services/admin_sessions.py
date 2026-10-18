"""Admin session create / lookup / revoke (hashed session tokens only)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.security_tokens import (
    constant_time_equals,
    new_opaque_token,
    sha256_hex,
)
from app.models_rag import AdminSession


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def create_admin_session(
    db: Session,
    *,
    ttl_seconds: int,
) -> tuple[AdminSession, str, str]:
    """
    Persist a new session.

    Returns (row, raw_session_token, raw_csrf_token).
    Only the session cookie value is hashed at rest; CSRF is stored server-side
    so GET /auth/me can re-supply it after browser refresh (not in localStorage).
    """
    if ttl_seconds < 60:
        raise ValueError("ttl_seconds must be >= 60")
    raw_session = new_opaque_token()
    raw_csrf = new_opaque_token()
    now = _utcnow()
    row = AdminSession(
        session_token_hash=sha256_hex(raw_session),
        csrf_token=raw_csrf,
        created_at=now,
        expires_at=now + timedelta(seconds=ttl_seconds),
        last_seen_at=now,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row, raw_session, raw_csrf


def get_valid_session_by_raw_token(
    db: Session,
    *,
    raw_session_token: str,
    touch_last_seen: bool = False,
) -> AdminSession | None:
    """
    Lookup a valid session by opaque cookie value.

    Safe GET authentication must call with touch_last_seen=False so reads do
    not mutate session state.
    """
    if not raw_session_token or not raw_session_token.strip():
        return None
    token_hash = sha256_hex(raw_session_token.strip())
    row = db.scalar(
        select(AdminSession).where(AdminSession.session_token_hash == token_hash)
    )
    if row is None:
        return None
    now = _utcnow()
    expires = row.expires_at
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=timezone.utc)
    if row.revoked_at is not None:
        return None
    if expires <= now:
        return None
    if touch_last_seen:
        row.last_seen_at = now
        db.add(row)
        db.commit()
        db.refresh(row)
    return row


def csrf_matches(session: AdminSession, raw_csrf: str | None) -> bool:
    if not raw_csrf or not raw_csrf.strip():
        return False
    return constant_time_equals(session.csrf_token, raw_csrf.strip())


def revoke_session(db: Session, session: AdminSession) -> None:
    session.revoked_at = _utcnow()
    db.add(session)
    db.commit()


def purge_expired_sessions(db: Session, *, now: datetime | None = None) -> int:
    """Delete expired or long-revoked sessions (operational cleanup helper)."""
    moment = now or _utcnow()
    rows = list(db.scalars(select(AdminSession)).all())
    removed = 0
    for row in rows:
        expires = row.expires_at
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        revoked = row.revoked_at
        if revoked is not None and revoked.tzinfo is None:
            revoked = revoked.replace(tzinfo=timezone.utc)
        if expires <= moment or (revoked is not None and revoked <= moment):
            db.delete(row)
            removed += 1
    if removed:
        db.commit()
    return removed


def session_public_view(session: AdminSession) -> dict[str, Any]:
    return {
        "session_id": str(session.id),
        "expires_at": session.expires_at.isoformat(),
        "revoked": session.revoked_at is not None,
    }


def get_session_by_id(db: Session, session_id: UUID) -> AdminSession | None:
    return db.get(AdminSession, session_id)
