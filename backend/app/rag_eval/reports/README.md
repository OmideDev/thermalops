# Answer-eval local reports

Raw smoke-run JSON/Markdown under this directory are **gitignored**.

They may contain full model answers and timing metrics from a local Ollama
run. They are not required for CI.

## Reproduce

From `backend/`:

```bat
set RUN_RAG_ANSWER_EVAL=1
python scripts/run_answer_eval.py
```

Optional separate post-generation probe (one Ollama call; synthetic index only):

```bat
set RUN_RAG_ANSWER_EVAL_POSTGATE=1
python scripts/run_answer_eval.py --postgate-only
```

Commit-safe methodology and human-review notes:
`backend/docs/rag_answer_eval_5b2b_audit.md`.
