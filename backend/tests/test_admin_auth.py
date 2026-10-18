"""Offline admin session authentication tests (SQLite + Fake settings)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.admin_passwords import hash_admin_password, verify_admin_password
from app.core.config import Settings, admin_auth_configured
from app.core.login_throttle import LoginThrottle
from app.core.security_tokens import sha256_hex
from app.db.base import Base
from app.db.session import get_db
from app.models_rag import AdminSession
from main import app

ADMIN_ORIGIN = "http://localhost:5173"
MUTATION_HEADERS = {"Origin": ADMIN_ORIGIN}


@pytest.fixture
def password() -> str:
    return "correct-horse-test-password"


@pytest.fixture
def password_hash(password: str) -> str:
    return hash_admin_password(password)


@pytest.fixture
def admin_settings(password_hash: str) -> Settings:
    return Settings(
        _env_file=None,
        assistant_enabled=False,
        admin_username="admin",
        admin_password_hash=password_hash,
        admin_session_ttl_seconds=3600,
        admin_cookie_secure=False,
        admin_cookie_samesite="lax",
        admin_allowed_origins=ADMIN_ORIGIN,
        admin_login_max_failures=5,
        admin_login_window_seconds=300,
        database_password=None,
    )


@pytest.fixture
def session_factory():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(
        bind=engine, autoflush=False, autocommit=False
    )
    Base.metadata.create_all(bind=engine)
    yield TestingSessionLocal
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


@pytest.fixture
def throttle() -> LoginThrottle:
    return LoginThrottle(max_failures=5, window_seconds=300)


@pytest.fixture
def client(session_factory, admin_settings: Settings, throttle: LoginThrottle):
    def override_get_db():
        db = session_factory()
        try:
            yield db
        finally:
            db.close()

    from app.api.v1 import admin_auth as admin_auth_routes
    from app.core import admin_deps

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[admin_deps.get_settings] = lambda: admin_settings
    app.dependency_overrides[admin_deps.require_admin_auth_configured] = (
        lambda: admin_settings
    )
    app.dependency_overrides[admin_auth_routes.get_login_throttle] = lambda: throttle

    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.clear()


def _login(client: TestClient, *, username: str, password: str):
    return client.post(
        "/api/v1/admin/auth/login",
        json={"username": username, "password": password},
    )


def test_argon2id_password_verification(password: str, password_hash: str):
    assert password_hash.startswith("$argon2id$")
    assert verify_admin_password(password_hash=password_hash, password=password)
    assert not verify_admin_password(
        password_hash=password_hash, password="wrong-password"
    )


def test_admin_auth_configured_helper():
    assert admin_auth_configured(
        Settings(_env_file=None, admin_username="a", admin_password_hash="x")
    )
    assert not admin_auth_configured(
        Settings(_env_file=None, admin_username=None, admin_password_hash=None)
    )


def test_login_success_sets_httponly_cookie(
    client: TestClient, password: str, session_factory
):
    response = _login(client, username="admin", password=password)
    assert response.status_code == 200
    body = response.json()
    assert body["username"] == "admin"
    assert body["csrf_token"]
    assert "expires_at" in body

    cookie = response.cookies.get("thermalops_admin_session")
    assert cookie
    set_cookie = response.headers.get("set-cookie", "")
    assert "HttpOnly" in set_cookie or "httponly" in set_cookie.lower()
    assert "Path=/api/v1/admin" in set_cookie or "path=/api/v1/admin" in set_cookie.lower()
    assert "SameSite=lax" in set_cookie or "samesite=lax" in set_cookie.lower()

    with session_factory() as db:
        rows = list(db.scalars(select(AdminSession)).all())
        assert len(rows) == 1
        assert rows[0].session_token_hash == sha256_hex(cookie)
        # CSRF stored server-side for /me re-supply; session cookie remains hashed.
        assert rows[0].csrf_token == body["csrf_token"]
        assert rows[0].session_token_hash != cookie


def test_login_wrong_password(client: TestClient):
    response = _login(client, username="admin", password="wrong-password")
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid credentials"


def test_me_requires_authentication(client: TestClient):
    response = client.get("/api/v1/admin/auth/me")
    assert response.status_code == 401


def test_me_returns_csrf_after_refresh(client: TestClient, password: str):
    """Browser refresh: session cookie survives; CSRF rehydrated from /me."""
    login = _login(client, username="admin", password=password)
    assert login.status_code == 200
    login_csrf = login.json()["csrf_token"]

    me = client.get("/api/v1/admin/auth/me")
    assert me.status_code == 200
    body = me.json()
    assert body["username"] == "admin"
    assert body["csrf_token"] == login_csrf
    assert body["session_id"]
    assert body["expires_at"]


def test_me_get_does_not_mutate_last_seen(
    client: TestClient, password: str, session_factory
):
    login = _login(client, username="admin", password=password)
    assert login.status_code == 200
    with session_factory() as db:
        row = db.scalars(select(AdminSession)).first()
        assert row is not None
        frozen = row.last_seen_at
        row.last_seen_at = datetime(2020, 1, 1, tzinfo=timezone.utc)
        db.add(row)
        db.commit()
        frozen = row.last_seen_at

    me = client.get("/api/v1/admin/auth/me")
    assert me.status_code == 200

    with session_factory() as db:
        row = db.scalars(select(AdminSession)).first()
        assert row is not None
        assert row.last_seen_at == frozen


def test_logout_requires_csrf(client: TestClient, password: str):
    login = _login(client, username="admin", password=password)
    assert login.status_code == 200
    response = client.post(
        "/api/v1/admin/auth/logout",
        headers=MUTATION_HEADERS,
    )
    assert response.status_code == 403
    assert "CSRF" in response.json()["detail"]


def test_logout_requires_allowed_origin(client: TestClient, password: str):
    login = _login(client, username="admin", password=password)
    csrf = login.json()["csrf_token"]
    response = client.post(
        "/api/v1/admin/auth/logout",
        headers={"X-CSRF-Token": csrf},
    )
    assert response.status_code == 403
    assert "Origin" in response.json()["detail"]

    evil = client.post(
        "/api/v1/admin/auth/logout",
        headers={
            "Origin": "https://evil.example",
            "X-CSRF-Token": csrf,
        },
    )
    assert evil.status_code == 403
    assert "Origin" in evil.json()["detail"]


def test_logout_with_csrf_revokes_session(
    client: TestClient, password: str, session_factory
):
    login = _login(client, username="admin", password=password)
    csrf = login.json()["csrf_token"]
    response = client.post(
        "/api/v1/admin/auth/logout",
        headers={**MUTATION_HEADERS, "X-CSRF-Token": csrf},
    )
    assert response.status_code == 200
    assert response.json()["ok"] is True

    me = client.get("/api/v1/admin/auth/me")
    assert me.status_code == 401

    with session_factory() as db:
        row = db.scalars(select(AdminSession)).first()
        assert row is not None
        assert row.revoked_at is not None


def test_me_csrf_usable_for_logout_after_refresh(client: TestClient, password: str):
    login = _login(client, username="admin", password=password)
    assert login.status_code == 200
    me = client.get("/api/v1/admin/auth/me")
    csrf = me.json()["csrf_token"]
    response = client.post(
        "/api/v1/admin/auth/logout",
        headers={**MUTATION_HEADERS, "X-CSRF-Token": csrf},
    )
    assert response.status_code == 200


def test_expired_session_rejected(client: TestClient, password: str, session_factory):
    login = _login(client, username="admin", password=password)
    assert login.status_code == 200
    with session_factory() as db:
        row = db.scalars(select(AdminSession)).first()
        assert row is not None
        row.expires_at = datetime.now(timezone.utc) - timedelta(seconds=5)
        db.add(row)
        db.commit()
    me = client.get("/api/v1/admin/auth/me")
    assert me.status_code == 401


def test_missing_auth_configuration_fails_closed(session_factory):
    empty = Settings(
        _env_file=None,
        assistant_enabled=False,
        admin_username=None,
        admin_password_hash=None,
        database_password=None,
    )

    def override_get_db():
        db = session_factory()
        try:
            yield db
        finally:
            db.close()

    from app.core import admin_deps

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[admin_deps.get_settings] = lambda: empty

    try:
        with TestClient(app) as test_client:
            response = test_client.post(
                "/api/v1/admin/auth/login",
                json={"username": "admin", "password": "x"},
            )
            assert response.status_code == 503
            assert "not configured" in response.json()["detail"].lower()
    finally:
        app.dependency_overrides.clear()


def test_require_admin_ping_without_csrf(client: TestClient, password: str):
    login = _login(client, username="admin", password=password)
    assert login.status_code == 200
    ping = client.get("/api/v1/admin/auth/ping")
    assert ping.status_code == 200
    assert ping.json() == {"ok": True, "username": "admin"}


def test_login_throttle_blocks(client: TestClient, throttle: LoginThrottle):
    for _ in range(5):
        response = _login(client, username="admin", password="bad")
        assert response.status_code == 401
    blocked = _login(client, username="admin", password="bad")
    assert blocked.status_code == 429


def test_cors_does_not_expose_credentials_to_foreign_origin(client: TestClient):
    """Permissive CORS must not allow arbitrary origins to read credentialed responses."""
    response = client.options(
        "/api/v1/admin/auth/me",
        headers={
            "Origin": "https://evil.example",
            "Access-Control-Request-Method": "GET",
        },
    )
    # Starlette CORSMiddleware omits ACAO for disallowed origins.
    assert response.headers.get("access-control-allow-origin") != "https://evil.example"
