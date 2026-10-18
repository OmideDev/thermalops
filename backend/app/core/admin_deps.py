"""
FastAPI dependencies for admin authentication and authorization.

Authentication = proving session validity via HttpOnly cookie.
Authorization = single admin role (require_admin). Kept separate for clarity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.core.config import Settings, admin_auth_configured, settings
from app.db.session import get_db
from app.models_rag import AdminSession
from app.services import admin_sessions as sessions_svc

_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


@dataclass(frozen=True)
class AdminPrincipal:
    """Authenticated admin identity for request handlers."""

    username: str
    session_id: str


def get_settings() -> Settings:
    return settings


def require_admin_auth_configured(
    cfg: Annotated[Settings, Depends(get_settings)],
) -> Settings:
    if not admin_auth_configured(cfg):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Admin authentication is not configured",
        )
    return cfg


def _cookie_name(cfg: Settings) -> str:
    return cfg.admin_cookie_name


def admin_allowed_origins(cfg: Settings) -> set[str]:
    raw = (cfg.admin_allowed_origins or "").strip()
    if not raw:
        return {
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        }
    return {part.strip() for part in raw.split(",") if part.strip()}


def origin_allowed_for_admin_mutation(request: Request, cfg: Settings) -> bool:
    """
    Cookie-authenticated mutations must come from an allowed Origin.

    Browsers send Origin on cross-site and most same-site fetch POSTs.
    Missing Origin is rejected for unsafe methods (fail closed for CSRF class).
    Does not trust X-Forwarded-For or spoofable client IPs for this check.
    """
    origin = (request.headers.get("origin") or "").strip()
    if not origin:
        return False
    return origin in admin_allowed_origins(cfg)


def authenticate_admin_session(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    cfg: Annotated[Settings, Depends(require_admin_auth_configured)],
) -> AdminSession:
    """Authenticate: validate opaque session cookie against hashed DB row."""
    raw = request.cookies.get(_cookie_name(cfg))
    touch = request.method.upper() not in _SAFE_METHODS
    session = sessions_svc.get_valid_session_by_raw_token(
        db,
        raw_session_token=raw or "",
        touch_last_seen=touch,
    )
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )
    return session


def require_csrf_for_unsafe_methods(
    request: Request,
    session: Annotated[AdminSession, Depends(authenticate_admin_session)],
    cfg: Annotated[Settings, Depends(get_settings)],
    x_csrf_token: Annotated[str | None, Header(alias="X-CSRF-Token")] = None,
) -> AdminSession:
    """
    Gate state-changing admin requests: CSRF + allowed Origin.

    Safe methods (GET/HEAD/OPTIONS) skip CSRF/Origin mutation checks.
    """
    if request.method.upper() in _SAFE_METHODS:
        return session

    if not origin_allowed_for_admin_mutation(request, cfg):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Origin not allowed for admin mutation",
        )

    header_name = cfg.admin_csrf_header_name
    raw = x_csrf_token
    if raw is None:
        raw = request.headers.get(header_name)
    if not sessions_svc.csrf_matches(session, raw):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="CSRF validation failed",
        )
    return session


def require_admin(
    session: Annotated[AdminSession, Depends(require_csrf_for_unsafe_methods)],
    cfg: Annotated[Settings, Depends(get_settings)],
) -> AdminPrincipal:
    """
    Authorization: caller is the configured single admin.

    Separated from cookie authentication so future RBAC can wrap this.
    """
    username = (cfg.admin_username or "").strip()
    if not username:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Admin authentication is not configured",
        )
    return AdminPrincipal(username=username, session_id=str(session.id))


AdminSessionDep = Annotated[AdminSession, Depends(authenticate_admin_session)]
RequireAdminDep = Annotated[AdminPrincipal, Depends(require_admin)]
