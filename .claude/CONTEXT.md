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

# Sentinel-RAG: Benchmark Definition for Capstone

## Context and Lessons from Roommate's Project

Your roommate's project (CSS_graph_project) builds a **GraphRAG system for legal contracts** using CUAD data. Their critical evaluation revealed several pitfalls you must avoid from the start:

- **No ground truth = no valid results.** Their RAGAS context_recall used the generated answer as a proxy for ground truth (circular reasoning). You must create gold-standard annotations first.
- **Stubs invalidate claims.** Their contradiction_flagger was a stub returning 0.0. Every feature you claim as a benchmark advantage must actually be implemented.
- **Statistical rigor is non-negotiable.** They lacked significance tests, confidence intervals, and cross-document variance analysis.

Your project adapts their general architecture (knowledge graph + optimized retrieval + single-pass generation) to the **military domain**, adding **temporal decay** as a novel contribution. The benchmarks below are designed so that passing them constitutes publishable evidence of improvement.

---

## Evaluation Corpus

**Primary corpus:** US Army Field Manuals (publicly available from Army Publishing Directorate and the Hugging Face `Heralax/us-army-fm-instruct` dataset). Use 5-8 field manuals spanning different operational domains (e.g., FM 3-0 Operations, FM 2-0 Intelligence, FM 3-90 Tactics, FM 4-0 Sustainment, FM 6-0 Commander and Staff Organization).

**Why this works for multi-hop:** Field manuals heavily cross-reference each other (e.g., FM 3-0 references FM 2-0 for intelligence procedures, which references FM 6-0 for staff processes). This creates natural multi-hop reasoning requirements analogous to the legal cross-reference problem in your roommate's project.

**Gold-standard annotation requirement:** 40 query-answer pairs (minimum), manually created. Each pair includes:

- The natural language query
- The correct answer (extracted verbatim from the FM text)
- Section references that contain the answer components
- An "information unit checklist" (the atomic facts needed for a complete answer)
- Query category label (see Adversarial Categories below)
- Hop count (1-hop, 2-hop, 3-hop, 4-hop)

---

## Benchmark 1: Retrieval Quality (RAGAS-based)

These metrics measure how well the retriever fetches the right context before the LLM ever sees it. All targets are derived from published GraphRAG vs vanilla RAG comparisons.


| Metric                | Definition                                                                  | Vanilla RAG Baseline (expected) | Sentinel-RAG Target | Source / Justification                                                                                                                                       |
| --------------------- | --------------------------------------------------------------------------- | ------------------------------- | ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Context Recall**    | Fraction of ground-truth information units present in retrieved context     | 0.45 - 0.55                     | >= 0.75             | GraphRAG-Bench (arXiv:2502.11371) shows ~20-30 point improvement on multi-hop; roommate's V3 achieved similar gaps on legal Trap A/B                         |
| **Context Precision** | Fraction of retrieved chunks that are actually relevant (not noise)         | 0.50 - 0.60                     | >= 0.72             | Vanilla RAG retrieves fixed top-k with no pruning; graph-based CSS pruning (as in roommate's project) consistently improves precision                        |
| **Evidence Recall**   | Percentage of gold-standard section references that appear in retrieved set | 0.40 - 0.50                     | >= 0.80             | This is the critical military metric -- missing a referenced section (like missing "Notwithstanding Section 4.1" in legal) means missing operational context |
| **MRR@10**            | Mean Reciprocal Rank of first relevant chunk in top-10 results              | 0.55 - 0.65                     | >= 0.78             | Standard IR metric; graph-based re-ranking consistently improves by 10-15 points over dense retrieval alone                                                  |


**Pass criterion:** Sentinel-RAG must beat the vanilla RAG baseline on ALL FOUR metrics with p < 0.05 (paired Wilcoxon signed-rank test across the 40 queries).

---

## Benchmark 2: Generation Quality (RAGAS-based)

These metrics measure the final answer quality. Use the official `ragas` Python library (not custom implementations -- a lesson from your roommate's project).


| Metric                 | Definition                                                               | Vanilla RAG Baseline (expected) | Sentinel-RAG Target | Justification                                                                                                                                                      |
| ---------------------- | ------------------------------------------------------------------------ | ------------------------------- | ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Faithfulness**       | Fraction of claims in the answer that are supported by retrieved context | 0.60 - 0.70                     | >= 0.85             | Vanilla RAG hallucinates when it fails to retrieve definitions/cross-refs (roommate's Trap B). GraphRAG retrieves the definition, so the LLM doesn't need to guess |
| **Answer Correctness** | Semantic + factual overlap with ground truth answer                      | 0.45 - 0.55                     | >= 0.70             | Published GraphRAG results show ~4.5% improvement on HotpotQA (simple 2-hop); military multi-hop (3-4 hop) should show larger gaps                                 |
| **Answer Relevancy**   | How directly the answer addresses the query                              | 0.65 - 0.75                     | >= 0.82             | Better context = more focused answer; less hallucination filler                                                                                                    |
| **ROUGE-L**            | Lexical overlap with ground truth                                        | 0.30 - 0.40                     | >= 0.50             | Conservative target; ROUGE-L is noisy for long-form answers                                                                                                        |


**Pass criterion:** Sentinel-RAG must beat vanilla RAG on Faithfulness AND Answer Correctness with p < 0.05. Answer Relevancy and ROUGE-L are secondary (report, but don't gate on them).

---

## Benchmark 3: Efficiency (Tokens-to-Truth)

This is adapted directly from your roommate's "Tokens-to-Truth Iteration Test." It measures how many queries and tokens are needed to achieve a complete answer.


| Metric                                      | Definition                                                                          | Vanilla RAG Baseline (expected)  | Sentinel-RAG Target         | Justification                                                                                             |
| ------------------------------------------- | ----------------------------------------------------------------------------------- | -------------------------------- | --------------------------- | --------------------------------------------------------------------------------------------------------- |
| **Information Unit Coverage (single pass)** | Fraction of checklist items satisfied in one query                                  | 0.40 - 0.55                      | >= 0.80                     | Core hypothesis: graph-based cross-reference expansion retrieves scattered info in one pass               |
| **Queries to 80% Coverage**                 | Number of iterative queries TRAG needs to match Sentinel-RAG's single-pass coverage | 2.5 - 4.0                        | 1.0 (by definition)         | Roommate's protocol estimated 3+ iterations for TRAG; military FMs with deep cross-refs should be similar |
| **Total Prompt Tokens**                     | Tokens sent to LLM across all iterations to reach 80% coverage                      | 8,000 - 15,000 (TRAG cumulative) | 2,000 - 4,000 (single pass) | Sentinel-RAG's CSS-style pruning keeps context compact; TRAG accumulates conversation history             |
| **End-to-End Latency**                      | Wall-clock time to final answer                                                     | 8 - 20s (TRAG iterative)         | 3 - 8s (single pass)        | Single LLM call vs multiple iterative calls                                                               |


**Pass criterion:** Sentinel-RAG must achieve >= 0.80 Information Unit Coverage in a single pass, while vanilla RAG requires >= 2 additional iterative queries to match that coverage level.

---

## Benchmark 4: Adversarial Categories (Military-Adapted "Traps")

Adapted from your roommate's three trap types to the military domain. These are the queries where vanilla RAG is structurally guaranteed to fail.

### Trap A: "The Overriding Directive" (10 queries)

**Military version of "Invisible Exception."** A tactical procedure in FM 3-90 appears to be the correct answer, but a commander's authority override in FM 6-0 supersedes it under specific conditions.

- **Key metric:** Fatal Error Rate (answers that omit the override)
- **Vanilla RAG expected:** 60-80% fatal error rate (the override section has low vector similarity to the tactical query)
- **Sentinel-RAG target:** <= 15% fatal error rate

### Trap B: "The Distant Definition" (10 queries)

**Military version of "Bridge Node."** A query about a specific military procedure uses a doctrinal term (e.g., "Main Battle Area," "Decisive Point") that is defined in a different FM or a different chapter.

- **Key metric:** Definition Retrieval Rate (does the system retrieve the term's definition?)
- **Vanilla RAG expected:** 20-35% retrieval rate (generic doctrinal definitions have low similarity to specific tactical queries)
- **Sentinel-RAG target:** >= 80% retrieval rate

### Trap C: "The Scattered Components" (10 queries)

**Military version of "Subquery Synthesis."** A query asks for all conditions or steps for a process (e.g., "List all conditions for transition from defense to offense"), but the components are spread across multiple sections and annexes.

- **Key metric:** Component Recall (fraction of scattered components retrieved)
- **Vanilla RAG expected:** 0.35 - 0.50 (top-k clusters around the most similar section, missing distant components)
- **Sentinel-RAG target:** >= 0.75

### Control: Single-Hop Queries (10 queries)

Simple factual questions where the answer is in a single contiguous passage. Both systems should perform well. This ensures Sentinel-RAG doesn't regress on easy queries.

- **Key metric:** Answer Correctness
- **Vanilla RAG expected:** 0.70 - 0.80
- **Sentinel-RAG target:** >= 0.75 (must not be significantly worse than vanilla RAG)

---

## Benchmark 5: Temporal Decay (Novel Contribution)

This is your **unique research contribution** -- not present in your roommate's project or in standard GraphRAG literature. It addresses the fact that military intelligence has a shelf life.

**Setup:** Create a sub-corpus where some documents have explicit timestamps (e.g., SITREPs from Day 1, Day 3, Day 7 of an operation). Include deliberately contradictory information across time periods (e.g., "Enemy strength at OBJ Alpha: 2 platoons" on Day 1 vs "Enemy strength at OBJ Alpha: 1 company" on Day 5).


| Metric                     | Definition                                                                                                                                            | Standard GraphRAG Baseline (no decay)  | Sentinel-RAG Target (with temporal decay) |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------- | ----------------------------------------- |
| **Temporal Correctness**   | When conflicting info exists across time, does the system prefer the most recent?                                                                     | 0.50 (random -- no temporal awareness) | >= 0.85                                   |
| **Recency Bias Detection** | When asked about historical trends ("How has enemy strength changed?"), does the system synthesize across time rather than just returning the latest? | 0.40 (returns latest only)             | >= 0.70                                   |
| **Stale Info Rate**        | Percentage of answers that cite outdated information without flagging it as potentially stale                                                         | 40-60%                                 | <= 15%                                    |


**Pass criterion:** Sentinel-RAG with temporal decay must outperform Sentinel-RAG WITHOUT temporal decay (ablation study) on Temporal Correctness with p < 0.05. This isolates the contribution of your novel feature.

---

## Benchmark 6: Graph Quality (Structural Metrics)

These measure the quality of the knowledge graph itself, independent of the QA task. Important for demonstrating that the graph construction is sound.


| Metric                            | Definition                                                                                                                  | Target                                                        |
| --------------------------------- | --------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| **Entity Extraction F1**          | Precision/recall of extracted military entities (units, locations, equipment, operations) against manually annotated sample | >= 0.75 F1                                                    |
| **Relation Extraction F1**        | Precision/recall of extracted relations (cross-references, hierarchy, temporal)                                             | >= 0.65 F1                                                    |
| **Graph Connectivity**            | Percentage of nodes reachable from any other node (should be high for useful traversal)                                     | >= 0.60 (single connected component covers majority of nodes) |
| **Cross-Reference Edge Accuracy** | Precision of automatically detected cross-reference edges (manually verify a sample of 100)                                 | >= 0.80 precision                                             |


---

## Statistical Requirements

For every comparison between Sentinel-RAG and vanilla RAG:

- **Significance test:** Paired Wilcoxon signed-rank test (non-parametric, does not assume normality)
- **Threshold:** p < 0.05
- **Effect size:** Report Cohen's d for all primary metrics
- **Confidence intervals:** 95% CI for all mean scores
- **Multiple comparison correction:** Holm-Bonferroni across the 4 primary RAGAS metrics
- **Cross-document variance:** Report per-FM performance breakdown to show results are not driven by a single document
- **Ablation study:** Test Sentinel-RAG with temporal decay disabled to isolate the contribution of your novel feature

---

## Summary: What "Passing" Looks Like

To claim "Sentinel-RAG improves on existing technology," you must demonstrate ALL of the following:

1. **Retrieval:** Context Recall >= 0.75 AND Evidence Recall >= 0.80 (both significantly better than vanilla RAG baseline)
2. **Generation:** Faithfulness >= 0.85 AND Answer Correctness >= 0.70 (both significant)
3. **Efficiency:** Single-pass Information Unit Coverage >= 0.80 (vanilla RAG needs 2+ iterations to match)
4. **Adversarial:** Fatal Error Rate <= 15% on Trap A (vs ~70% for vanilla RAG)
5. **Novel contribution:** Temporal Correctness >= 0.85 with decay enabled vs ~0.50 without (ablation significant at p < 0.05)
6. **No regression:** Single-hop Answer Correctness is not significantly worse than vanilla RAG

These targets are calibrated against published numbers from "RAG vs GraphRAG: A Systematic Evaluation" (arXiv:2502.11371), GraphRAG-Bench, and HotpotQA multi-hop baselines, adjusted for the domain-specific nature of military text (which typically makes retrieval harder, widening the gap in Sentinel-RAG's favor).

---

# Live project context (repository)

## Current State (April 11, 2026)

- Graph rebuilt: 283,278 edges, 73,551 cross-reference edges, 100% node connectivity
- **21 tests** in `tests/test_graph_fixes.py` all passing (7 new tests added covering fixes #4 Jaccard, #5 cross-doc base_score, #6 temporal decay, #11 cross-doc cap, #12 source diversity)
- RAGAS API fixed: migrated from deprecated `ragas.metrics` + `Dataset` (0.3.x) to `ragas.metrics.collections` + `batch_score()` + `AsyncOpenAI` (0.4.x)
- Harness silent exception fixed: bare `except: pass` replaced with `except Exception as exc: print + traceback`
- Third benchmark run complete (April 11, 2026) — see results below

---

## Third Benchmark Run Results (April 11, 2026)

### RAGAS metrics — still all 0.0 (NEW root cause identified)

RAGAS is now correctly wired to the API but **rate-limited to 0** during the benchmark run.
Root cause: `compute_ragas_metrics_batch` fires all 40 queries × 5 metrics = 200 async LLM calls
simultaneously per system. With 4 systems evaluated back-to-back, that is ~800 calls in
rapid succession, blowing the org's 200k TPM limit. Tenacity retries exhaust and the
per-metric handler logs `[RAGAS] metric 'X' failed` → score stays 0.0.

**Fix needed:** Chunk the batch into groups of ~5 queries with a `asyncio.sleep(60)` between
chunks inside `compute_ragas_metrics_batch`. This keeps calls under ~1k TPM/min.

### Non-RAGAS metrics (real, from third run)

| Metric | Vanilla RAG | Sentinel-RAG | Target | Status |
|---|---|---|---|---|
| Evidence Recall | 0.894 | 0.392 | ≥0.80 | FAIL — regression persists |
| ROUGE-L | 0.164 | 0.096 | ≥0.50 | FAIL |
| IU Coverage (single pass) | 0.298 | 0.231 | ≥0.80 | FAIL |
| Trap C Component Recall | 0.457 | 0.673 | ≥0.75 | Close — +0.216 improvement |
| Overall Component Recall | 0.670 | 0.706 | — | Sentinel leads |
| Fatal Errors (Trap A) | 0/10 | 0/10 | ≤15% | Both 0% (likely keyword mismatch in metric) |
| Defn Retrieved (Trap B) | 0/10 | 0/10 | ≥80% | Likely keyword mismatch in metric |
| Avg Latency | 18.45s | 368.63s | 3–8s | FAIL — still severe |
| Avg Tokens | 6,523 | 8,234 | 2–4k | FAIL |

### Per-category breakdown (component_recall)

| Category | Vanilla RAG | Sentinel-RAG |
|---|---|---|
| Trap A (Overriding Directive) | 0.590 | 0.767 |
| Trap B (Distant Definition) | 0.773 | 0.793 |
| Trap C (Scattered Components) | 0.457 | 0.673 |
| Control (Single-Hop) | 0.858 | 0.592 |

### Open problems after third run

1. **RAGAS rate-limit:** Add chunked batching with delays in `compute_ragas_metrics_batch` — chunk size ~5, sleep 60s between chunks
2. **Evidence recall regression (0.894 → 0.392):** Source diversity fix (#12) may be over-capping; Sentinel pulls from fewer gold FMs. Investigate source cap vs gold FM coverage.
3. **Control single-hop regression:** Vanilla=0.858 comp_recall vs Sentinel=0.592 — graph expansion is pulling in noise on simple queries. CSS scoring may need higher relevance weight for single-hop.
4. **Latency (368s/query):** BFS max_results=50 cap helped vs pre-fix but is still too slow for production. Profile the CSS scoring loop — `encode_single()` called per candidate is the likely bottleneck.
5. **Trap A/B showing 0/0:** `fatal_error_rate` and `definition_retrieved` keyword matching may be too narrow. Review `evaluation/harness.py:_extract_override_keywords` and gold annotation keywords.

## What Good Output Looks Like

- A fix that targets exactly one bug without touching unrelated code
- A benchmark run where Sentinel-RAG beats vanilla RAG on ALL four RAGAS retrieval metrics
- A graph retriever that finds cross-FM evidence for multi-hop queries in a single pass
- Statistical analysis that includes p-values, Cohen's d, 95% CI, and Holm-Bonferroni correction

## What to Avoid

- Do NOT re-introduce hard caps removed in the 12 bug fixes (e.g., `seed_ids[:5]`)
- Do NOT mock RAGAS or evaluate metrics — use the real `ragas` library
- Do NOT skip the Wilcoxon test — results without significance tests are not publishable
- Do NOT conflate "answer generated" with "ground truth" (circular reasoning, roommate's mistake)
- Do NOT add hallucinated features — every benchmark claim must be implemented and tested

**Temporal sub-corpus (Benchmark 5):** SITREPs from Day 1, Day 3, Day 7 with deliberate contradictions (e.g., enemy strength estimates that change over time) to test temporal decay.
