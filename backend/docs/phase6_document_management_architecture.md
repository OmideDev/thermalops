# Phase 6A — Multi-Document RAG Architecture (Design Only)

**Status:** Phase 6A design + Phase 6B.1 foundations (session auth + schema).
Still **no** PDF upload or React admin UI.

**Backend HEAD (6A baseline):** `5aa3cb3dc22967afe2a0561c3a7ecb9a61d9039a`

**Frontend HEAD (observed):** `366ab94e6f7c370af388d22cd3f492019e2a639d`
(`thermalops-front`)

### Decision updates (post-6A review → 6B.1)

| Topic | Decision |
|-------|----------|
| Admin auth | **Session cookies** + CSRF (not a permanent bearer token in React/Vite) |
| Password | Argon2id hash via backend env only |
| Active retrieval | **Global immutable snapshot** + singleton active pointer |
| Vectors | Versioned filesystem artifacts (DB holds refs/checksums) |
| Workers | Single backend worker for initial staging |
| Eligibility | Explicit per-document policy + approval before activation |
| Greenheck p31 | Policy-scoped (`greenheck_erv_v1`), not a global page ban |

---

## 1. Current architecture findings

### 1.1 Backend data layer

| Area | Reality |
|------|---------|
| Models | `backend/app/models.py`: `Organization`, `Building`, `Equipment` only |
| Conventions | UUID PKs, `DateTime(timezone=True)` + `server_default=func.now()`, string `SAEnum` with `native_enum=False`, FKs `ondelete="RESTRICT"` |
| Session | `app/db/session.py` → `get_db` |
| Base | `app/db/base.py` → `DeclarativeBase` |
| Alembic | `backend/migrations/`; one revision `c8625a040993_create_initial_hvac_data_models.py` |
| Auth | **Absent.** No JWT/session deps, no roles, no `get_current_user`. Equipment/Buildings/Assistant routes are open. |

### 1.2 HTTP surface

| Router | Module | Notes |
|--------|--------|-------|
| Health | `app/api/v1/health.py` | Unauthenticated |
| Buildings | `app/api/v1/buildings.py` | List only; open |
| Equipment | `app/api/v1/equipment.py` | CRUD-ish; open |
| Assistant | `app/api/v1/assistant.py` | `POST /api/v1/assistant/ask`; open when loaded |

App entry: `backend/main.py` — CORS for Vite `5173`, lifespan loads assistant.

### 1.3 Assistant / RAG pipeline (exact modules)

| Concern | Module / function |
|---------|-------------------|
| Lifespan init | `main.lifespan` → `try_create_assistant_from_settings(settings)` → `app.state.assistant` |
| Settings | `app/core/config.py`: `assistant_enabled` (default **false**), `assistant_manual_path`, `assistant_allowed_pages="15,29,33"`, Ollama timeouts, `assistant_top_k` |
| Build | `AssistantService.build_from_manual` in `app/services/assistant.py` |
| Extract | `scripts/extract_pdf_text.py` loaded dynamically via `_load_extract_module` |
| Eligible chunks | `load_eligible_chunks_from_manual` → `collect_eligible_chunks_from_pages` (`app/rag/search.py`) |
| Page 31 | Explicitly stripped: `chunk["page_number"] != 31` **and** must be in `allowed_pages` |
| Chunk IDs | `app/rag/chunk.py` `_stable_chunk_id`: `{document_id}:p{page}:c{index}:{sha12}:{digest}` |
| Document ID / SHA | From PDF extraction metadata (`document_id`, `content_sha256`) |
| Embeddings | `try_create_minilm_embedder` (`app/rag/embeddings.py`), model `sentence-transformers/all-MiniLM-L6-v2` |
| Index | `build_index` → `InMemorySemanticIndex` (process RAM only) |
| Ask | `ask_with_evidence` (`app/rag/ask.py`); citations **server-mapped** from evidence IDs; titles from `document_titles` |
| Safety | `app/rag/safety.py` pre/post operational gates |
| Concurrency | Per-process non-blocking `threading.Lock` in `AssistantService.ask` |
| Public contract | `{answer, sources[{document_id, title, page}]}` — `app/schemas/assistant.py` |

**Critical behavior today:** the index is built **once at process start**. It is
not rebuilt per request. There is no DB-backed document registry, no upload
path, and no hot-swap of indexes.

### 1.4 Frontend findings (`thermalops-front`)

| Area | Reality |
|------|---------|
| Routing | `src/App.tsx`: `/`, `/ai`, `/equipment`, `/equipment/:id` under `AppShell` — **no auth guards** |
| Auth | **Absent** (no login, tokens, or protected routes) |
| Nav | `Sidebar.tsx`: Settings / Help marked **Soon**; no Documents item |
| API client | `src/api/client.ts`: `apiGet` / `apiPost`, `ApiError` from FastAPI `detail` |
| Assistant | `src/api/assistant.ts` → `POST /api/v1/assistant/ask`; hook `useAssistantAsk` |
| UI primitives | `EmptyState`, `LoadingState`, `ErrorState`, `StatusBadge`, shadcn-style `Card`/`Button` |
| Upload patterns | **None** yet |

### 1.5 Evaluation assets to preserve

- Dataset `app/rag_eval/datasets/greenheck_erv_eval_v1.json` (page 31 excluded)
- Answer smoke selection + claim audits (`rag_eval/`, docs Phase 5B/5C)
- Offline FakeLLM tests; opt-in live Ollama scripts

---

## 2. Goals and non-goals

### Goals

- Incremental path from one configured local PDF → admin-managed multi-document RAG
- Preserve Greenheck demo + citation contract + safety gates + eligibility quality bars
- Secure admin-only ingestion (never public upload)
- Durable metadata and recoverable index activation

### Non-goals (Phase 6A)

- Implementing APIs, migrations, React pages, or auth code
- Introducing cloud vector DBs or paid embedding APIs
- Treating every PDF page/table as indexable
- Changing public `AskResponse` unless a later phase separately justifies it

---

## 3. Recommended data model (PostgreSQL)

Minimal schema aligned with existing UUID / timestamp / string-enum conventions.

### 3.1 `rag_documents`

| Column | Type | Purpose |
|--------|------|---------|
| `id` | UUID PK | Stable document identity for APIs/citations |
| `display_title` | String(200) | Citation title (maps to today’s `document_titles`) |
| `original_filename` | String(255) | User-facing original name (display only) |
| `storage_key` | String(512) unique | Server-generated relative key under private storage |
| `content_sha256` | String(64) unique indexed | Duplicate detection / provenance |
| `mime_validated` | Bool | Passed PDF sniff/validation |
| `byte_size` | BigInteger | Upload size |
| `page_count` | Int nullable | After parse |
| `indexed_page_count` | Int nullable | Pages that entered eligibility |
| `status` | String enum | See lifecycle §5 |
| `error_code` / `error_message` | String nullable | Safe operator-facing errors (no stack dumps to clients) |
| `eligibility_policy_id` | String | e.g. `greenheck_erv_v1` or `default_narrative_v1` |
| `pipeline_version` | String | Extract/chunk/embed config stamp |
| `active_index_run_id` | UUID FK nullable | Pointer to activated run |
| `equipment_id` | UUID FK nullable → `equipment.id` | Optional association |
| `uploaded_at` / `indexed_at` / `archived_at` | timestamptz | Audit |
| `created_at` / `updated_at` | timestamptz | Standard |

### 3.2 `rag_index_runs`

Separate run table is **justified**: build is expensive and must not mutate the
live pointer until complete.

| Column | Purpose |
|--------|---------|
| `id` | UUID PK |
| `document_id` | FK → `rag_documents` |
| `status` | `pending` / `building` / `ready` / `failed` / `superseded` |
| `embedding_model` | e.g. MiniLM id |
| `chunk_count` / `vector_dim` | Stats |
| `artifact_path` | Relative path to durable index artifact |
| `content_sha256` | Echo of bytes indexed |
| `eligibility_policy_id` / `pipeline_version` | Reproducibility |
| `started_at` / `finished_at` | Timing |
| `error_message` | Operator-facing |

### 3.3 `rag_chunks` (metadata + text; v1)

For prototype scale (tens–low thousands of chunks), store chunks in PostgreSQL:

| Column | Purpose |
|--------|---------|
| `id` | UUID PK (internal) |
| `index_run_id` | FK |
| `chunk_id` | Stable string id (same format as today) unique per run |
| `document_id` | Denormalized for citations |
| `page_number` | Citation page |
| `section_heading` | Optional |
| `text` | Eligible narrative text |
| `char_count` | Gate stats |
| `retrieval_eligibility` / `safe_for_retrieval` | Preserve quality flags |

### 3.4 Embeddings storage (v1 recommendation)

**Do not introduce a new vector DB for the first multi-doc cut.**

| Option | Fit for this project | Recommendation |
|--------|----------------------|----------------|
| Pure in-memory only | Current behavior; lost on restart | Keep as **runtime cache**, not sole source of truth |
| PostgreSQL `BYTEA` / float array column | Fine at small N; simple backups | Acceptable for vectors if preferred |
| **Filesystem artifact** (numpy/npz sidecar keyed by `index_run_id`) | Simple, matches “build once”; easy atomic swap | **Preferred v1** alongside PG chunk rows |
| pgvector / Qdrant / Weaviate | Extra ops surface | **Defer** until N or latency demands it |

**v1 design:** PostgreSQL owns document + run + chunk text/metadata; each ready
`rag_index_runs` has an on-disk embedding artifact under private storage.
Process loads **active** runs into `InMemorySemanticIndex` (or a multi-doc
wrapper) at startup / after activate.

### 3.5 Relationships (conceptual)

```
Organization 1—* Building 1—* Equipment
Equipment 0..1 ←—* rag_documents (optional)
rag_documents 1—* rag_index_runs
rag_index_runs 1—* rag_chunks
rag_documents.active_index_run_id → rag_index_runs.id  (nullable until ready)
```

---

## 4. Secure file ingestion

### 4.1 Essential controls (local prototype)

| Control | Specification |
|---------|---------------|
| Authz | Admin-only routes (see §6); **no public upload** |
| MIME | Require `application/pdf`; magic-byte sniff, not extension alone |
| Size | Hard max (e.g. 25–50 MiB) at reverse-proxy + app |
| Pages | Hard max page count after open (e.g. 200) |
| Filename | Sanitize display name; **never** use as storage path |
| Storage key | Server UUID / hashed key under `backend/data/rag_private/` (gitignored) |
| SHA-256 | Compute on ingest; reject/reuse duplicates per policy |
| Encrypted/malformed PDF | Fail `validating` → `failed` with safe message |
| Processing limits | Time budget + memory; one indexing job at a time initially |
| Content trust | PDF text is **untrusted evidence** (existing ask prompt already treats evidence as untrusted) |

### 4.2 Essential for internet-accessible / staging

Everything above, plus:

- TLS termination
- Rotatable admin secret / real identity
- Rate limits + upload quotas per admin
- Virus/malware scanning where policy requires
- Audit log of upload/delete/reindex/activate
- No directory listing; storage outside static/web roots
- CSRF strategy if cookie sessions are used later

### 4.3 Prompt-injection / embedded content

- Do not execute PDF JS or follow “SYSTEM OVERRIDE” text as instructions
  (already framed in `ask.py` / safety)
- Eligibility gates remain; exclude flattened troubleshooting tables by policy
  (Greenheck page 31 must stay excluded under `greenheck_erv_v1` policy)
- Never auto-mark all pages eligible

---

## 5. Indexing lifecycle

### 5.1 Document statuses

```
uploaded → validating → processing → ready
                              ↘ failed
ready → archived
failed → processing (retry)
```

Terminal purge after archive/delete policy

### 5.2 Transitions (allowed)

| From | To | Trigger |
|------|----|---------|
| (new) | uploaded | Bytes accepted + row created |
| uploaded | validating | Worker starts PDF checks |
| validating | processing | PDF OK; extract/chunk/embed begins |
| validating | failed | Invalid PDF / limits |
| processing | ready | Index run `ready` **and** atomic activate |
| processing | failed | Pipeline error |
| ready | archived | Admin archive |
| failed | processing | Admin retry (new `rag_index_runs` row) |
| archived | (purged) | Background delete of bytes + artifacts |

### 5.3 Atomic activation

1. Build **new** `rag_index_runs` + chunk rows + embedding artifact to a temp path.
2. Fsync / finalize artifact.
3. In a DB transaction: set run `ready`, set `rag_documents.active_index_run_id`,
   mark previous run `superseded`.
4. Signal assistant runtime to **reload** active documents (or reload on next
   safe window).
5. Only then delete old artifacts (after reload success).

**Never** replace the live in-memory index with a half-built structure.
**Never** rebuild embeddings on every ask request.

### 5.4 Evolving today’s startup index

Today: `try_create_assistant_from_settings` builds from one filesystem PDF.

Target evolution:

1. **Compatibility mode:** if no `ready` documents in DB, keep current
   `assistant_manual_path` bootstrap (Greenheck demo remains).
2. **DB mode:** load all `ready` documents’ active runs into a multi-doc
   `InMemorySemanticIndex` (concatenate eligible chunks) once per process /
   after activate.
3. Hot reload under the same generation lock (or “reload when idle”) so ask
   never searches a torn index.

### 5.5 Failure / delete / stale prevention

- Failed runs never become `active_index_run_id`.
- Delete/archive: deactivate first → wait for reload → delete storage.
- Citations must only resolve documents that are `ready` and active.
- Chunk IDs include content hash prefix; content change ⇒ new run, not silent
  overwrite.

### 5.6 Concurrency / workers

- v1: **single indexer** (DB advisory lock or Redis-less file lock).
- Ask path keeps per-process generation lock.
- Multi-worker: each worker holds its own RAM index copy; activation must
  bump a `index_generation` counter (DB) that workers poll, or publish reload
  event. Document this as an open ops decision before horizontal scale.

### 5.7 Idempotent retries

Retry creates a **new** `rag_index_runs` with same `content_sha256`; success
activates the new run. Duplicate uploads of identical SHA return existing
document (409 or 200 idempotent — choose in API §7).

---

## 6. Authorization design

### 6.1 Current state

Admin **session auth** is implemented for `/api/v1/admin/auth/*` (Phase 6B.1).
Equipment / Buildings / Assistant ask remain unauthenticated (staging blocker).
Document upload APIs are not shipped yet.

### 6.2 Secure approach (implemented in 6B.1)

**Backend-managed admin sessions** (preferred over permanent bearer tokens in React):

| Aspect | Decision |
|--------|----------|
| Login | `POST /api/v1/admin/auth/login` verifies username + Argon2id hash from env |
| Session | Opaque cookie (`HttpOnly`, `SameSite=Lax`, `Secure` on HTTPS); **hash** stored in `admin_sessions` |
| CSRF | Synchronizer returned at login **and** from authenticated `GET /auth/me` after refresh |
| Mutations | Require `X-CSRF-Token` + allowlisted `Origin` (aligned with CORS Vite origins) |
| Secrets | Never via Vite env / localStorage / sessionStorage |
| Who | Single admin (no multi-user RBAC yet) |
| Fail closed | Missing admin env config → 503 on admin auth routes |
| Throttle | In-process login attempt limiter (single-worker assumption); no `X-Forwarded-For` |
| Assistant ask | Remains open for local demo — **staging blocker** until locked down |
| Audit | Upload/delete audit table deferred to 6B.2/6B.3 |

### 6.3 Explicitly forbidden

- Public anonymous upload/delete/reindex
- Hardcoded admin passwords in source
- Serving `rag_private/` via StaticFiles

### 6.4 Rate limits / quotas (v1)

- Per-token: N uploads/hour, max concurrent index jobs = 1
- Reject oversized bodies early (413)

---

## 7. API contracts (proposed)

Base: `/api/v1/admin/documents` — **all require admin auth** unless noted.

### First-version (MVP)

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/v1/admin/documents` | List (status, title, sha, updated) |
| POST | `/api/v1/admin/documents` | Multipart PDF upload |
| GET | `/api/v1/admin/documents/{id}` | Detail + active run summary |
| GET | `/api/v1/admin/documents/{id}/index-status` | Run status / errors / counts |
| POST | `/api/v1/admin/documents/{id}/retry-index` | New run if failed/stale policy |
| POST | `/api/v1/admin/documents/{id}/archive` | Soft archive + deactivate |

#### Upload `POST /api/v1/admin/documents`

- Request: `multipart/form-data` file=PDF; optional `display_title`, `equipment_id`
- 201: document id + status `uploaded`/`validating`
- 400: validation failure
- 401/403: auth
- 409: duplicate `content_sha256` (return existing id)
- 413: too large
- 429: quota

#### Detail / status

- Include: status, page_count, indexed_page_count, pipeline_version,
  active_index_run_id, error_message (safe), timestamps
- **Do not** return absolute filesystem paths or raw storage roots to clients

#### Retry

- 202 accepted if new run enqueued
- 409 if already processing

#### Archive

- 204/200; document no longer cited; ask index reloads without it

### Future enhancements (not MVP)

- Re-eligibility policy editor per document
- Partial page allow-lists UI beyond presets
- Bulk upload
- Public read APIs for document catalog
- User accounts / RBAC

### Assistant contract

Keep `POST /api/v1/assistant/ask` response shape unchanged in Phase 6B/C unless
a separate ADR changes it. Multi-doc only expands which chunks can be retrieved;
citations still `{document_id, title, page}`.

---

## 8. React integration design (no implementation yet)

Fit existing shell:

1. Add primary nav item **Documents** (or under System → Settings later).
2. Route `/documents` → list page using `EmptyState` / `LoadingState` / `ErrorState`.
3. Upload control: native file input + `FormData` (extend `client.ts` with
   `apiUpload` later — not JSON `apiPost`).
4. Row status: reuse badge patterns (map `ready`/`processing`/`failed` like
   equipment status colors).
5. Actions: Retry (failed), Archive (confirm dialog), Detail drawer/page.
6. Errors: map `ApiError.status` (401 → “admin token required”, 413 size, etc.).
7. Auth UX (prototype): settings field to store admin token locally for admin
   calls only — never send token to assistant unless product requires it.
8. No mock-only final UI: pages call real admin APIs once 6B ships.

---

## 9. Preserving existing RAG behavior

| Requirement | Design rule |
|-------------|-------------|
| Citation provenance | Always from active run chunk metadata; never LLM-supplied |
| No deleted/inactive cites | Search only active ready runs |
| Not every page trusted | `eligibility_policy_id`; Greenheck policy keeps pages **15/29/33** only |
| Page 31 | Hard exclude in policy + pipeline (mirror `load_eligible_chunks_from_manual`) |
| Demo continuity | Bootstrap path from `assistant_manual_path` until DB has ready docs; seed Greenheck as first managed doc |
| Safety gates | Unchanged modules `app/rag/safety.py` / ask pre-post gates |
| Eval datasets | Keep using stable `document_id` / chunk_id scheme |

---

## 10. Open decisions / tradeoffs

1. **Ask endpoint auth:** leave open for local demo vs require admin/user token in staging.
2. **Embedding artifact format:** numpy `.npz` vs JSONL+bin — pick in 6B.2.
3. **Duplicate SHA policy:** 409 vs return existing 200.
4. **Multi-worker reload:** poll `index_generation` vs process restart only for v1.
5. **Equipment association:** optional FK now vs defer until product needs it.
6. **Whether chunk text lives in PG or only in artifact:** recommend PG for
   auditability; artifact for vectors.

---

## 11. Expected files in next implementation phases

*(Listed for planning — do not create in 6A.)*

**Backend add:** `app/models` document tables or extend `models.py`; schemas;
`app/api/v1/admin_documents.py`; `app/services/rag_ingestion.py`;
`app/core/admin_auth.py`; storage helpers; Alembic revision; gitignore for
`data/rag_private/`; tests.

**Backend modify:** `main.py` lifespan reload hook; `AssistantService` multi-doc
load; optionally `config.py` settings for storage root + admin token.

**Frontend add:** `pages/DocumentsPage.tsx`, `api/documents.ts`, types, hooks;
nav link; upload helper.

**Do not modify in early 6B:** public AskResponse schema; safety rules; eval gold
labels; Greenheck eligibility constants without an explicit policy object.
