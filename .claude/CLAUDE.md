---
name: Sentinel-RAG Benchmark Definition
overview: Define a concrete, research-grade benchmark suite for Sentinel-RAG (military-domain Adaptive GraphRAG) that establishes numeric pass/fail targets across retrieval, generation, efficiency, and military-specific dimensions, proving improvement over vanilla RAG baselines.
todos:
  - id: gold-annotations
    content: Create 40 gold-standard query-answer pairs across 5-8 US Army Field Manuals, categorized into Trap A/B/C + Control, with information unit checklists and section references
    status: completed
  - id: vanilla-rag-baseline
    content: Implement vanilla RAG baseline (ChromaDB/FAISS + top-k retrieval + single LLM call) and iterative vanilla RAG agent (evaluator loop for Tokens-to-Truth test)
    status: completed
  - id: sentinel-graphrag
    content: "Implement Sentinel-RAG core: document ingestion, military entity/relation extraction, knowledge graph construction with cross-reference edges, CSS-style graph optimization, single-pass generation"
    status: completed
  - id: temporal-decay
    content: "Implement temporal decay mechanism: timestamp-aware edge weighting, recency scoring during retrieval, stale-info flagging in generation"
    status: completed
  - id: evaluation-harness
    content: "Build automated evaluation harness: RAGAS library integration (official), custom military-specific metrics (Fatal Error Rate, Definition Retrieval Rate, Component Recall, Temporal Correctness), statistical analysis (Wilcoxon, Cohen's d, Holm-Bonferroni), LaTeX tables, per-FM breakdown"
    status: completed
  - id: run-and-report
    content: Execute full benchmark suite with ablation study (temporal decay on/off), generate LaTeX-ready tables with significance markers, per-FM breakdown, iterative RAG comparison
    status: completed
isProject: false
---

# Sentinel-RAG: Who You Are Helping and How to Help

**Benchmark spec:** The capstone plan lives in `.claude/CONTEXT.md` — same YAML frontmatter, section order, tables, pass criteria, statistical gates, and summary checklist as the project plan. Read that file for numeric targets and trap definitions before changing retrieval, evaluation, or reporting.

## The Project in One Sentence

You are helping build **Sentinel-RAG** — a military-domain Adaptive GraphRAG system that proves, with publishable benchmarks, that graph-augmented retrieval outperforms vanilla RAG on US Army Field Manuals.

## Who I Am

A computer science student building a capstone research project. My roommate built a similar system (CSS_graph_project) for legal contracts. Sentinel-RAG is the military-domain adaptation of that architecture, with one novel contribution: **temporal decay** — the idea that intelligence from Day 1 of an operation is less reliable than intelligence from Day 7.

## What This System Does

1. **Ingests** US Army Field Manuals (FM 3-0, FM 2-0, FM 3-90, FM 4-0, FM 6-0, etc.)
2. **Builds a knowledge graph** from extracted entities and cross-references between FMs
3. **Retrieves** relevant chunks via CSS-style graph traversal + vector similarity + temporal decay scoring
4. **Generates** answers in a single LLM pass (no iterative query expansion)
5. **Benchmarks** against vanilla RAG across 6 benchmark dimensions to prove improvement

## Codebase Layout

```
retrieval/
  graph_retriever.py   — Core graph traversal + CSS scoring + source diversity
  vector_store.py      — Dense vector retrieval (baseline)
graph/
  entity_extractor.py  — Military entity extraction + IDF-weighted cross-ref edges
  knowledge_graph.py   — BFS graph traversal, neighbor lookup
evaluation/
  harness.py           — Benchmark runner (RAGAS + custom metrics)
  metrics.py           — Evidence recall, trap detection, temporal correctness
  statistical.py       — Wilcoxon tests, Cohen's d, Holm-Bonferroni correction
generation/
  llm_client.py        — LLM call wrapper
  prompt_templates.py  — Single-pass generation prompts
baselines/             — Vanilla RAG implementation for comparison
tests/
  test_graph_fixes.py  — 21 tests covering all 12 graph bugs fixed in April 2026
```

## The 12 Bugs Already Fixed (April 2026)

All critical graph bugs have been fixed and tested. Do NOT reintroduce:
- Seed expansion now uses all 15 vector results (not hard-capped at 5)
- Cross-reference edges created on citation detection alone (no shared-entity guard)
- Entity frequency threshold raised to 200 (was 50, dropped military terms)
- Jaccard similarity for entity overlap (was asymmetric, inflated sparse queries)
- BFS capped at `max_results=50` to prevent latency explosion on 283K-edge graph
- Source diversity enforced: single FM capped at `ceil(final_top_k / 2)` in top slice

## How to Respond to Me

- Be direct and technical — assume I understand Python, NLP, and information retrieval
- When touching retrieval or graph code, check the 12 fixes before suggesting changes
- Benchmark targets are not negotiable — they are calibrated against published numbers (see `.claude/CONTEXT.md` and `.claude/REFERENCES.md`)
- Flag if a suggested change would regress any of the 6 benchmark dimensions
- Prefer editing existing files over creating new ones
- When in doubt about a metric, check `evaluation/metrics.py` before guessing

## Fixes Applied (April 14, 2026 — fourth run)

1. **RAGAS TPM rate-limit:** `compute_ragas_metrics_batch` now chunks into groups of 5 queries with 60s sleep between chunks. In `evaluation/metrics.py`.
2. **RAGAS `max_tokens` truncation:** `faithfulness` and `answer_correctness` were failing because RAGAS LLM responses (JSON statement lists) hit the default output token limit. Fixed by passing `max_tokens=4096` to `llm_factory()` in `_build_ragas_llm_and_embeddings()`.
3. **RAGAS empty context crash:** Some Sentinel queries returned zero chunks, causing `ValueError: retrieved_contexts cannot be empty`. Fixed with `or [""]` fallback in `evaluation/harness.py`.
4. **RAGAS connection retry:** `batch_score()` was failing on transient DNS/network drops with no retry. Added exponential backoff retry (5 attempts: 15→30→60→120→180s) around each chunk call in `evaluation/metrics.py`.
5. **Latency fix:** `encode_single()` was called per candidate in `_css_score` (was 368s/query). Fixed by batch-encoding all candidates in one `encode()` call in `retrieval/graph_retriever.py`. Only applies to future query runs — current benchmark uses checkpointed query results.

## Current Known Issues (as of April 14, 2026 — fourth run)

1. **Intermittent network drops:** DNS resolution for `api.openai.com` fails intermittently during RAGAS evaluation. Retry logic is now in place but a stable connection is still needed.
2. **Evidence recall regression:** Vanilla=0.894 vs Sentinel=0.392 — source diversity fix (#12) may be over-capping single-FM retrieval, reducing coverage of gold FMs.
3. **Control single-hop regression:** Sentinel comp_recall=0.592 vs Vanilla=0.858 — graph expansion adds noise on simple queries. CSS relevance weight may need tuning.
4. **Zero-chunk Sentinel queries:** Some Sentinel-RAG queries return no chunks (retrieval failure). Needs investigation after benchmark completes.
5. **Trap A/B metrics showing 0/0:** Keyword matching in `fatal_error_rate` / `definition_retrieved` may be too narrow for the gold annotations used.
6. **Latency still 368s in current report:** Batch encoding fix only applies to new query runs. Use `--fresh` to regenerate with improved latency.
