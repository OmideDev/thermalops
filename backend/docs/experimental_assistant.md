# Experimental grounded assistant (Phase 4C)

Informational-only prototype. Not a field-service authority.

## Enable (local development)

1. Place the Greenheck manual PDF at the configured path (default):

   `backend/data/manuals/460945ERT_iom.pdf`

   The PDF is **not** committed to Git.

2. Install RAG extras if needed: `pip install -r requirements-rag.txt`

3. Run Ollama locally with `qwen3:4b-instruct` pulled.

4. Set environment (or `.env`):

```env
ASSISTANT_ENABLED=true
ASSISTANT_MANUAL_PATH=data/manuals/460945ERT_iom.pdf
ASSISTANT_ALLOWED_PAGES=15,29,33
ASSISTANT_OLLAMA_MODEL=qwen3:4b-instruct
ASSISTANT_OLLAMA_BASE_URL=http://127.0.0.1:11434
```

5. Start the API from `backend/` on **loopback only** (this app currently has **no authentication**):

```bash
python -m uvicorn main:app --reload --host 127.0.0.1
```

Do **not** bind to `0.0.0.0` or expose the assistant-enabled process on a public network. CORS allowlists are not authentication.

If the PDF, FastEmbed, or Ollama setup fails, the API still starts; assistant routes return **503**. Equipment and Building APIs remain available.

## Indexed pages

Only **15, 29, 33**. Page **31** stays excluded (flattened troubleshooting tables).

## Example request

```bash
curl -s http://127.0.0.1:8000/api/v1/assistant/ask ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"What does the dirty filter sensor monitor?\"}"
```

Response shape:

```json
{
  "answer": "...",
  "sources": [
    {"document_id": "...", "title": "...", "page": 15}
  ]
}
```

Supported abstention (insufficient evidence or operational safety) still returns **HTTP 200** with an abstention `answer` and `"sources": []`.

## Errors

| Status | Meaning |
|--------|---------|
| 422 | Invalid / empty / oversized question |
| 429 | Another generation is in progress in this process (limit: 1) |
| 502 | Ollama protocol / invalid upstream response |
| 503 | Assistant disabled/uninitialized, or Ollama daemon unavailable |
| 504 | Ollama request timed out |
| 200 | Successful answer **or** policy-driven abstention (`sources: []`) |

Client error bodies use short fixed messages (no stack traces, PDF paths, or prompt text).

Per-process concurrency is **not** a cluster-wide limit when running multiple worker processes.

## Offline tests

Default `pytest` does **not** require Ollama, models, or the vendor PDF.

Optional live assistant API E2E:

```bash
set RUN_ASSISTANT_E2E=1
python -m pytest -q -s tests/test_assistant_e2e_live.py
```

## Known limitations

- Keyword operational-safety gates are incomplete; no factual entailment check.
- Local CPU inference is slow; one generation per process.
- Blocking Ollama HTTP runs in FastAPI’s threadpool via a sync route.
- Three successful smoke questions do not prove general reliability.
- Request-path timing is not yet broken out in API responses; Ollama’s
  `load_duration` / `prompt_eval_*` / `eval_*` fields are the next smallest
  observability addition (log or debug header only — not required for Phase 4C).
