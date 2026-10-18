# Phase 6B.1 — Secure Admin Foundation and Document Persistence

Implements schema + session auth foundations only. **No PDF upload, no React,
no embedding execution.**

## Architecture decisions applied

| Decision | Choice |
|----------|--------|
| Admin auth | Backend-managed **sessions** (not permanent bearer token in Vite) |
| Password | Argon2id hash from env (`ADMIN_PASSWORD_HASH`); username from env |
| Session storage | SHA-256 of opaque session id in `admin_sessions` |
| CSRF | Session-scoped synchronizer; returned at login **and** from authenticated `GET /auth/me` |
| Cookie | HttpOnly, `SameSite=Lax` (configurable), `Secure` for HTTPS staging, path `/api/v1/admin` |
| Mutations | Unsafe methods require `X-CSRF-Token` **and** `Origin` in `ADMIN_ALLOWED_ORIGINS` |
| Vectors | Immutable versioned **filesystem artifacts** (refs only in DB this phase) |
| Active retrieval | **Global** `rag_index_snapshots` + singleton `rag_active_index_pointer` |
| Eligibility | Explicit `eligibility_policy_id` + `eligibility_approved`; Greenheck page 31 policy-scoped |
| Runtime | **Single API worker** assumed for login throttle / session semantics |
| Greenheck demo | Unchanged filesystem bootstrap via existing assistant settings |

## Final auth and CSRF contract

1. **Login** (`POST /api/v1/admin/auth/login`): validates credentials; sets
   HttpOnly session cookie; returns `{ username, csrf_token, expires_at }` in
   JSON. The CSRF value is **not** a password and is not placed in localStorage
   by design — React must keep it in memory.
2. **Refresh / session rehydrate** (`GET /api/v1/admin/auth/me`): cookie-only
   authentication. Returns `{ username, session_id, expires_at, csrf_token }` so
   a browser refresh can continue mutating without re-login and without storing
   auth secrets in localStorage/sessionStorage.
3. **Safe GETs** (`/me`, `/ping`) do **not** update `last_seen_at` and do not
   require CSRF.
4. **Mutations** (e.g. logout; future document writes): require session cookie +
   matching `X-CSRF-Token` + `Origin` in the allowlist
   (`http://localhost:5173` / `http://127.0.0.1:5173` by default). Missing or
   foreign Origin → 403.
5. **CORS**: credentials allowed only for the Vite localhost origins — foreign
   origins cannot read credentialed responses (including CSRF from `/me`).
6. **Logout**: CSRF + Origin; revokes session row; clears cookie.
7. **Fail closed**: missing `ADMIN_USERNAME` / `ADMIN_PASSWORD_HASH` → **503**.
8. **Throttle key**: peer address from the ASGI connection (`request.client.host`),
   **not** `X-Forwarded-For`.

## Auth routes

| Method | Path | Auth |
|--------|------|------|
| POST | `/api/v1/admin/auth/login` | Credentials; sets session cookie; returns CSRF |
| GET | `/api/v1/admin/auth/me` | Session cookie; returns CSRF for refresh |
| GET | `/api/v1/admin/auth/ping` | Session cookie (`require_admin`); no CSRF |
| POST | `/api/v1/admin/auth/logout` | Session cookie + CSRF + allowed Origin; revokes |

## Tables

`rag_documents`, `rag_index_runs`, `rag_chunks`, `rag_index_snapshots`,
`rag_index_snapshot_members`, `rag_active_index_pointer`, `admin_sessions`.

Alembic revision: `b7e4c2a91f05` (revises `c8625a040993`).

Existing HVAC tables (`organizations`, `buildings`, `equipment`) are unchanged
by this revision.

## Lifecycle invariants (helpers in `app/rag/lifecycle.py`)

- `uploaded → validating → awaiting_review → processing → ready`
- Failures → `failed`; `ready → archived`
- No auto-trust without eligibility approval
- Failed runs must not replace active snapshot pointer
- Archived docs not eligible for new snapshots
- Greenheck exclusions apply only under `greenheck_erv_v1`

## PostgreSQL migration validation status

**PENDING** for this commit: no disposable PostgreSQL database was targeted
with `RUN_ALEMBIC_PG_TEST=1`. Offline suite skips the opt-in Alembic PG test.

Migration integrity was **reviewed** (revision chain, seven new tables,
singleton pointer check, snapshot member FKs). Applying `alembic upgrade head`
against a working/production database is **not** authorized by this phase.

**Blocker before enabling real document persistence:** run the opt-in test (or
equivalent) against an explicitly disposable database and confirm upgrade from
`c8625a040993` → `b7e4c2a91f05`.

## Staging security blockers

- Equipment / Buildings / Assistant ask remain **unauthenticated**
- CORS still localhost Vite origins only
- Login throttle is **in-process** (single worker)
- CSRF secret is stored server-side for `/me` re-supply (session-scoped; not a
  password) — acceptable for this phase; revisit hashing if threat model changes
- PostgreSQL migration not proven on a live disposable instance yet

## Phase 6B.2 restrictions (exact)

Do **not** start 6B.2 until:

1. Disposable Postgres migration validation is green (`RUN_ALEMBIC_PG_TEST=1`).
2. Upload/index code stays behind admin session + CSRF + Origin checks.
3. No public anonymous upload; no Vite-stored admin secrets.
4. Greenheck filesystem bootstrap remains the fallback when no ready DB docs.
5. Single-worker assumption documented (or shared throttle introduced).
6. Private storage root stays gitignored; vendor manuals not committed.
7. Offline tests continue without Ollama / real vendor PDFs.

## Operator setup (local)

```bash
python -c "from app.core.admin_passwords import hash_admin_password; print(hash_admin_password('YOUR_PASSWORD'))"
```

Set in backend `.env` (never Vite):

```
ADMIN_USERNAME=admin
ADMIN_PASSWORD_HASH=...argon2id...
ADMIN_COOKIE_SECURE=false
ADMIN_ALLOWED_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
```

For HTTPS staging: `ADMIN_COOKIE_SECURE=true`.

Apply migration only when authorized on a disposable or staging DB:

```
alembic upgrade head
```

Opt-in migration test: `RUN_ALEMBIC_PG_TEST=1` against a **disposable** database.
