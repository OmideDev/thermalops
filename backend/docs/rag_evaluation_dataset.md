# RAG evaluation dataset (Phase 5B.2A)

Academically oriented, retrieval-first evaluation assets for the experimental
ThermalOps grounded assistant.

## Scope

- Document: local Greenheck ERV IOM PDF (`460945ERT_iom.pdf`, not committed)
- Eligible indexed pages: **15, 29, 33**
- Excluded: **page 31** (flattened troubleshooting tables)
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (FastEmbed ONNX)
- This phase evaluates **retrieval only** — no Qwen / Ollama generation runs

## Files

| Path | Purpose |
|------|---------|
| `app/rag_eval/datasets/greenheck_erv_eval_v1.json` | Curated scenarios + provenance |
| `app/rag_eval/fixtures/adversarial_synthetic_passages.json` | Labeled synthetic hostile passages |
| `app/rag_eval/dataset.py` | Schema load/validation |
| `app/rag_eval/metrics.py` | Hit@1, Recall@3, MRR + negative summaries |
| `app/rag_eval/retrieval.py` | Retrieval harness over existing search index |
| `scripts/run_retrieval_eval.py` | Opt-in local MiniLM evaluation CLI |
| `tests/test_rag_eval_*.py` | Offline schema/metric/harness unit tests |

## Scenario categories

See `scenario_counts` inside the dataset JSON. Categories:

- `supported_informational` — gold chunk IDs from eligible pages
- `unsupported_absent` — fact not present on 15/29/33
- `unsupported_excluded_page` — depends on excluded page 31
- `operational_policy` — application safety abstention (not grounding success)
- `adversarial_injection` — separate synthetic fixtures; not production gold

## Creation criteria

1. Supported facts were verified by reading eligible chunks from the local PDF.
2. LLM answers were never used as ground truth.
3. Policy abstention is labeled separately from insufficient-evidence abstention.
4. Adversarial passages are synthetic and must not be inserted into the production PDF index.
5. Dataset stores paraphrased reference facts only (no long copyrighted excerpts).

## Provenance

- `document_id` / `content_sha256` are recorded in the dataset root and per scenario.
- Chunk IDs include a content-hash prefix; if the PDF bytes change, re-verify gold IDs.
- Page 31 remains excluded because column-aligned troubleshooting tables flatten poorly.

## Metrics

For **retrieval_positive** only:

- Hit@1
- Recall@3
- MRR

For unsupported / policy queries:

- Report whether Top-K still returns neighbors and mean top-1 similarity
- **Do not** compute recall/MRR as if gold chunks existed
- Nearest neighbors ≠ supported answers; no universal similarity threshold is claimed

Adversarial items are not scored against the production index gold set.

## How to rerun

Offline (default CI / developer):

```bash
cd backend
python -m pytest -q
```

Local retrieval-only evaluation (PDF + FastEmbed required):

```bash
cd backend
set RUN_RETRIEVAL_EVAL=1
python scripts/run_retrieval_eval.py
```

## Version history

- **v1.0.0** — initial 40-scenario benchmark.
- **v1.0.1** — gold audit: S08 reduced to the single chunk containing the
  timed-exhaust 5/30-minute fact; difficulty labels added; companion
  `greenheck_erv_eval_v1_holdout_paraphrase.json` added (report separately;
  does not replace the core S01–S14 set).

## Recall@3 interpretation

For a supported query, Recall@3 = `|gold_chunk_ids ∩ top-3| / |gold_chunk_ids|`.

A dataset-level mean of **1.00** means every gold ID for every supported query
appeared somewhere in that query’s top-3 neighbors. It does **not** mean the
model answered correctly, and it is easier when most queries have a single gold
chunk.

## Known limitations

- Single product manual, three eligible pages — not a general HVAC QA benchmark.
- Several core supported questions are direct-terminology / heading-adjacent and
  are easy for lexical/semantic retrieval; use the holdout paraphrases for a
  harder secondary check.
- Semantic search always returns nearest neighbors; topical hits on unsupported questions are expected and must be interpreted carefully.
- Operational-policy cases may overlap thematically with excluded page 31 content; scoring them as “grounding success” would be incorrect.
- Overlap between positive and negative top-1 similarities on this small set is
  expected; do **not** derive a production similarity threshold from it.
- Generation / injection resistance evaluation is intentionally deferred.
