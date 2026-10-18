# Phase 6 — Document Management Implementation Plan

Companion to `phase6_document_management_architecture.md`.

**Phase 6A = design only (this repo state).** No migrations/APIs/UI in 6A.

Baseline backend: `5aa3cb3`. Frontend observed: `366ab94`.

---

## Principles for every subphase

1. Keep `POST /api/v1/assistant/ask` working with the existing Greenheck demo path.
2. Do not weaken operational safety or eligibility gates (page 31 stays excluded
   under Greenheck policy).
3. Do not ship public upload endpoints.
4. Prefer small PRs with offline tests; opt-in live checks only when needed.
5. Roll back by feature flag / settings without deleting HVAC tables.

---

## Phase 6B.1 — Persistence and authorization foundations

### Scope (implemented)

- Alembic migration `b7e4c2a91f05`: documents, index runs, chunks, global
  snapshots + active pointer, admin sessions
- SQLAlchemy models in `app/models_rag.py`
- Session auth: login / me / logout / ping; Argon2id; HttpOnly cookie; CSRF
  with `/me` refresh rehydrate + Origin checks on mutations
- `require_admin` dependency for future document routes
- Lifecycle invariant helpers (`app/rag/lifecycle.py`) — not the full engine
- Settings: `admin_username`, `admin_password_hash`, cookie/secure flags,
  `admin_allowed_origins`, `rag_storage_root`, `rag_documents_enabled`
- Offline SQLite tests; opt-in Postgres Alembic test
- **No** multipart upload / React / embedding execution

### Preserve assistant

- Lifespan still uses `try_create_assistant_from_settings` + local PDF when
  `assistant_enabled`
- No change to ask contract

### Rollback

- Downgrade Alembic revision; unset admin env; flag off

### Exit criteria

- Migration reviewed; PostgreSQL opt-in validation may remain **PENDING** until
  a disposable DB is available (blocker before real persistence)
- Unauthenticated call to protected admin routes fails closed
- Full offline pytest green
- See `phase6b1_admin_foundation.md` for final CSRF contract

---

## Phase 6B.2 — Secure ingestion and indexing

### Preconditions (hard)

- Disposable Postgres: `RUN_ALEMBIC_PG_TEST=1` green (or equivalent proof)
- Admin mutations remain cookie + CSRF + Origin gated
- Single-worker throttle limitation acknowledged (or replaced)

### Scope

- Private storage under gitignored root (e.g. `backend/data/rag_private/`)
- Upload endpoint: validate PDF magic, size, page limits, SHA-256, sanitize
  display filename, server `storage_key`
- Pipeline job (sync-in-process v1 acceptable): extract → eligibility policy →
  chunk → embed → write artifact → create `rag_index_runs` + `rag_chunks`
- Atomic activate when run ready
- Greenheck policy module: pages `{15,29,33}`, hard-exclude 31
- Default narrative policy for unknown manuals: conservative eligible-only
  narrative chunks; **do not** auto-include all pages
- Duplicate SHA handling
- Failed status + safe errors
- Offline tests with tiny synthetic PDFs / fixtures (no vendor manual in git)

### Assistant evolution (still compatible)

- If DB has zero `ready` docs → keep filesystem bootstrap
- If DB has ready docs → load active runs into memory index (multi-doc concat)
- Reload after activate under lock / idle

### Rollback

- Disable `rag_documents_enabled`; assistant falls back to manual path
- Leave uploaded bytes quarantined; do not delete HVAC data

### Exit criteria

- Upload → ready path works in local dev
- Failed PDF does not activate
- Ask still returns server-side citations only
- Page 31 never indexed under Greenheck policy tests

---

## Phase 6B.3 — Document CRUD and lifecycle

### Scope

- List / detail / index-status / retry-index / archive (and optional hard purge
  job)
- Idempotent retry (new run)
- Prevent citing archived docs
- Audit log rows for admin actions
- Rate limit / single-flight indexer
- API contract tests (status codes 401/409/413/202)

### Preserve

- Eval datasets unchanged; optional seed script to register existing Greenheck
  PDF as managed document **without** changing chunk_id scheme

### Rollback

- Archive all managed docs; bootstrap path remains

### Exit criteria

- Full admin HTTP surface covered by offline tests with FakeLLM / fake embedder
- Assistant regression suite still passes

---

## Phase 6C — React admin integration

### Scope (`thermalops-front`)

- `/documents` page + Sidebar link
- List with status badges (map to existing UI language)
- Upload control (`FormData` helper)
- Retry / archive with confirmation
- Loading / empty / error states via existing components
- Admin token handling for prototype (documented threat model: local-only)
- Playwright/unit tests for rendering + error mapping where practical

### Preserve

- Dashboard, Equipment, AI Insights unchanged except nav
- Assistant ask UX unchanged

### Rollback

- Hide nav flag / route; APIs remain for CLI use

### Exit criteria

- Operator can upload → see processing → ready → ask cites new doc title/page
- Delete/archive removes doc from answers after reload

---

## Adjusted boundary rationale

| Subphase | Why split this way |
|----------|--------------------|
| 6B.1 before upload | Auth + tables without storing untrusted bytes yet |
| 6B.2 before full CRUD | Hardest risk surface (PDF/index) isolated |
| 6B.3 | Lifecycle polish once pipeline trusted |
| 6C last | Frontend against stable contracts |

If implementation pressure is high, **6B.2+6B.3** may merge, but **do not**
merge 6C before admin APIs are real (no mock-only UI).

---

## Test plan (cross-cutting)

| Layer | Tests |
|-------|-------|
| Offline unit | Models, authz, validators, eligibility policy, activate transaction |
| Offline API | Admin routes with TestClient + token; assistant still FakeLLM |
| RAG eval | Existing `pytest` suite must stay green every subphase |
| Live opt-in | Manual: upload sample PDF, ask one question; not CI-default |
| Security | Attempt upload without token; path traversal filename; oversized body |

---

## Rollback strategy (summary)

1. Feature flag `rag_documents_enabled=false`
2. Keep `assistant_manual_path` Greenheck bootstrap
3. Alembic downgrade only if tables unused
4. Retain audit of storage keys before purge

---

## Suggested commit themes (future — not 6A)

- `feat(rag): add document metadata tables and admin auth foundation`
- `feat(rag): add secure PDF ingest and versioned index activation`
- `feat(rag): expose document lifecycle admin APIs`
- `feat(front): add documents admin page`

---

## Open items to resolve at start of 6B.1

1. Confirm staging will use Bearer admin token vs accelerating full user auth.
2. Confirm max upload size / page cap numbers for the deployment target.
3. Confirm whether optional `equipment_id` ships in first migration.
4. Confirm ask endpoint remains unauthenticated for local demo.
