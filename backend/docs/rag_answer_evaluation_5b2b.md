# Phase 5B.2B — Grounded Answer Evaluation

Evaluates factual-grounding **structure**, abstention, and application-level
safety independently of retrieval Hit@1/Recall@3/MRR. Reuses
`greenheck_erv_eval_v1` (no second conflicting framework).

## Selection (`answer_eval_smoke_v1`)

Eight curated scenarios (IDs and gold labels preserved):

| ID | Role |
|----|------|
| S01 | Supported informational (direct) |
| S05 | Supported informational (multi-evidence) |
| S13 | Supported informational (paraphrase) |
| U02 | Unsupported / absent |
| U08 | Unsupported / absent |
| E02 | Excluded page 31 |
| P01 | Operational policy (application gate ≠ grounding success) |
| A01 | Adversarial (synthetic fixture only) |

Rationale is recorded in `app/rag_eval/answer_selection_v1.json`.

## What is measured

**A. Retrieval (report-only alongside answers):** gold coverage in top-3,
retrieved vs gold chunk IDs.

**B. Model output (when LLM is called):** JSON schema validity, evidence ID
validity, abstention flag, injection-marker follow on raw text.

**C. Application behavior:** pre-generation operational abstention,
post-generation operational abstention, insufficient-evidence abstention,
answer+citations, unsafe escape.

Supported answers that pass structural checks are labeled
`needs_human_review=true`. A valid citation is **not** proof of factual support.
Pre-generation policy blocking is **not** counted as LLM grounding success.

## Offline tests

```bash
cd backend
python -m pytest -q tests/test_rag_answer_eval.py
python -m pytest -q
```

Uses `FakeLLMBackend` and synthetic indexes only (no PDF / Ollama).

## Opt-in live Qwen (≤8 scenarios)

```bash
cd backend
set RUN_RAG_ANSWER_EVAL=1
python scripts/run_answer_eval.py
```

Requires local PDF, FastEmbed MiniLM, and Ollama `qwen3:4b-instruct`.
No automatic retries. Timeouts are recorded as failures.

A01 is reported three ways:

1. Production-index ask (often pre-generation operational gate)
2. Synthetic adversarial index ask (isolated; never merges into production PDF index)
3. Raw model on poisoned prompt (gates bypassed) — separate from application outcomes

Raw results land under `app/rag_eval/reports/` (gitignored; see
`reports/README.md`). Commit-safe audit:
`docs/rag_answer_eval_5b2b_audit.md`.

Optional separate live post-generation probe (one call; synthetic index only):

```bash
set RUN_RAG_ANSWER_EVAL_POSTGATE=1
python scripts/run_answer_eval.py --postgate-only
```

## Limitations

- Smoke set of eight scenarios — do not overgeneralize.
- Does not claim comprehensive prompt-injection resistance.
- Does not change retrieval, safety rules, model, or public API schema.
