"""
Admin session authentication endpoints (Phase 6B.1).

Secrets stay server-side. Session id is HttpOnly cookie; CSRF token is returned
in the login JSON body for subsequent mutating admin requests.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy.orm import Session

from app.core.admin_deps import (
    AdminPrincipal,
    authenticate_admin_session,
    get_settings,
    require_admin,
    require_admin_auth_configured,
    require_csrf_for_unsafe_methods,
)
from app.core.admin_passwords import verify_admin_password
from app.core.config import Settings
from app.core.login_throttle import LoginThrottle, default_login_throttle
from app.core.security_tokens import constant_time_equals
from app.db.session import get_db
from app.models_rag import AdminSession
from app.schemas.admin_auth import (
    AdminLoginRequest,
    AdminLoginResponse,
    AdminLogoutResponse,
    AdminMeResponse,
)
from app.services import admin_sessions as sessions_svc

router = APIRouter(prefix="/api/v1/admin/auth", tags=["Admin Auth"])


def get_login_throttle() -> LoginThrottle:
    return default_login_throttle


def _client_throttle_key(request: Request) -> str:
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def _set_session_cookie(
    response: Response,
    *,
    raw_token: str,
    cfg: Settings,
    max_age: int,
) -> None:
    samesite = (cfg.admin_cookie_samesite or "lax").lower()
    if samesite not in {"lax", "strict", "none"}:
        samesite = "lax"
    response.set_cookie(
        key=cfg.admin_cookie_name,
        value=raw_token,
        httponly=True,
        secure=bool(cfg.admin_cookie_secure),
        samesite=samesite,
        max_age=max_age,
        path="/api/v1/admin",
    )


def _clear_session_cookie(response: Response, cfg: Settings) -> None:
    response.delete_cookie(
        key=cfg.admin_cookie_name,
        path="/api/v1/admin",
    )


@router.post(
    "/login",
    response_model=AdminLoginResponse,
    summary="Admin login (session cookie + CSRF token)",
)
def admin_login(
    payload: AdminLoginRequest,
    request: Request,
    response: Response,
    db: Annotated[Session, Depends(get_db)],
    cfg: Annotated[Settings, Depends(require_admin_auth_configured)],
    throttle: Annotated[LoginThrottle, Depends(get_login_throttle)],
) -> AdminLoginResponse:
    key = _client_throttle_key(request)
    if throttle.is_blocked(key):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many login attempts. Try again later.",
        )

    expected_user = (cfg.admin_username or "").strip()
    password_hash = (cfg.admin_password_hash or "").strip()
    user_ok = constant_time_equals(payload.username, expected_user)
    pass_ok = verify_admin_password(
        password_hash=password_hash,
        password=payload.password,
    )
    if not (user_ok and pass_ok):
        throttle.record_failure(key)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )

    throttle.clear(key)
    row, raw_session, raw_csrf = sessions_svc.create_admin_session(
        db,
        ttl_seconds=int(cfg.admin_session_ttl_seconds),
    )
    _set_session_cookie(
        response,
        raw_token=raw_session,
        cfg=cfg,
        max_age=int(cfg.admin_session_ttl_seconds),
    )
    return AdminLoginResponse(
        username=expected_user,
        csrf_token=raw_csrf,
        expires_at=row.expires_at.isoformat(),
    )


@router.get(
    "/me",
    response_model=AdminMeResponse,
    summary="Current admin session",
)
def admin_me(
    session: Annotated[AdminSession, Depends(authenticate_admin_session)],
    cfg: Annotated[Settings, Depends(get_settings)],
) -> AdminMeResponse:
    """
    Rehydrate admin UI after browser refresh.

    Returns the session-scoped CSRF token so React can continue mutating
    without storing auth secrets in localStorage. Same-origin only via cookie
    + CORS allowlist (cross-origin callers without credentials cannot read this).
    """
    return AdminMeResponse(
        username=(cfg.admin_username or "").strip(),
        session_id=str(session.id),
        expires_at=session.expires_at.isoformat(),
        csrf_token=session.csrf_token,
    )


@router.get(
    "/ping",
    summary="Authorization probe for require_admin (GET; no CSRF)",
)
def admin_ping(
    principal: Annotated[AdminPrincipal, Depends(require_admin)],
) -> dict[str, object]:
    return {"ok": True, "username": principal.username}


@router.post(
    "/logout",
    response_model=AdminLogoutResponse,
    summary="Revoke admin session",
)
def admin_logout(
    response: Response,
    db: Annotated[Session, Depends(get_db)],
    session: Annotated[AdminSession, Depends(require_csrf_for_unsafe_methods)],
    cfg: Annotated[Settings, Depends(get_settings)],
) -> AdminLogoutResponse:
    # CSRF enforced for POST via require_csrf_for_unsafe_methods.
    sessions_svc.revoke_session(db, session)
    _clear_session_cookie(response, cfg)
    return AdminLogoutResponse(ok=True)
