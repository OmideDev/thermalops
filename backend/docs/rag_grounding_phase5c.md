# Phase 5C — Evidence-scoped answer generation

Baseline commit: `35fb2a90faac5dc489cbb7647e0118beba70e3e6`.

## Change

Minimal system/user prompt guidance in `app/rag/ask.py`:

- Prefer the shortest faithful answer to the **question**
- Do not add mechanisms, timers, setpoints, part numbers, or procedures
  unless asked and present in evidence
- Do not invent specifications
- Do not put evidence IDs in the answer text

No retrieval, safety, model, or public `AskResponse` schema changes.
No claim verifier and no second LLM.

Prompt changes do **not** guarantee factual grounding.

## Claim audit

See `app/rag_eval/claim_audit_phase5c.json` (paraphrased claims + chunk IDs only;
no long vendor excerpts).

## S13 completeness judgment

**Question:** “What does the manual say about internal filter maintenance?”

Omitting “2-inch thick pleated paper” after the Phase 5C prompt is classified
as **helpful but optional (B)**, not a necessary completeness failure:

- The question asks about **maintenance** practice, not filter identity/SKU.
- Supported maintenance claims (routine check/replace, clean if dirty, same
  performance/quality replacements, upstream-of-coil checks) remain valid
  answers.
- Gold reference facts list useful section content; not every gold fact must
  appear in every valid answer.

Therefore no second prompt retune was applied for S13.

## Limited live comparison (same model / index)

Reproduce (local PDF + FastEmbed + Ollama; results gitignored):

```bat
set RUN_RAG_GROUNDING_COMPARE=1
python scripts/compare_grounding_prompt.py
```

| ID | Observation | Tokens (prev → new) |
|----|-------------|---------------------|
| S05 | Dropped VFD/timer over-elaboration; listed three options | 150 → 47 |
| S13 | Still supported maintenance claims; optional filter-type omitted | 105 → 90 |
| U02 | Abstention unchanged | 21 → 21 |

**Shorter is not automatically better.** Judge factual support and question
completeness separately from token count.

## Grounding limitations (remaining)

- Prompt steering is heuristic; models may still omit useful optional detail
  or over-include neighbor facts.
- Citation validity ≠ claim-level support.
- Incomplete timing details (e.g. S05 “5 minutes” without “every 30 minutes”)
  can still occur when elaborating.
- Eight-scenario smoke / three-scenario compare — do not overgeneralize.
- Keep human claim review when adding manuals (Phase 6+).
