# References

## Primary Inspiration: CSS_graph_project

**GitHub:** [SuyashPandey9/CSS_graph_project](https://github.com/SuyashPandey9/CSS_graph_project)

Roommate's GraphRAG implementation for legal contracts (CUAD dataset). Sentinel-RAG directly adapts its architecture:
- CSS (Contextual Semantic Scoring) for graph-based retrieval ranking
- Three trap query types (Invisible Exception, Bridge Node, Subquery Synthesis) → adapted to military Traps A/B/C
- Tokens-to-Truth efficiency benchmark → adapted as Benchmark 3
- Key failure mode to avoid: using the generated answer as a proxy for ground truth in RAGAS context_recall (circular reasoning)

---

## Core Research Papers

### GraphRAG vs Vanilla RAG

- [RAG vs. GraphRAG: A Systematic Evaluation and Key Insights (arXiv:2502.11371)](https://arxiv.org/abs/2502.11371)
  — Primary source for baseline numbers. Shows ~20-30 point improvement on multi-hop tasks. All Sentinel-RAG targets are calibrated against these numbers.

- [GraphRAG-Bench: When to Use Graphs in RAG (ICLR'26)](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark)
  — Official benchmark repo. GraphRAG's advantage scales with query complexity; diminishes on single-hop factual retrieval. Informs the Control (single-hop) benchmark design.

- [When to Use Graphs in RAG: A Comprehensive Analysis (arXiv:2506.05690)](https://arxiv.org/abs/2506.05690)
  — Establishes conditions under which GraphRAG outperforms vanilla RAG. Key insight: cross-document reasoning tasks are where the gap is largest.

### Multi-hop QA (calibration for generation targets)

- [HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering (EMNLP 2018)](https://arxiv.org/abs/1809.09600)
  — Multi-hop QA baseline cited in the capstone plan for Answer Correctness / GraphRAG comparisons (simple 2-hop); military 3–4 hop is expected to show larger gaps.

### RAGAS Evaluation Framework

- [RAGAS: Automated Evaluation of Retrieval Augmented Generation (arXiv:2309.15217)](https://arxiv.org/abs/2309.15217)
  — Original paper defining the four core metrics: Context Precision, Context Recall, Faithfulness, Answer Relevancy.

- [RAGAS Official Documentation](https://docs.ragas.io/en/stable/)
  — Implementation reference. Use the official `ragas` Python library — do NOT re-implement metrics (critical lesson from roommate's project).

- [RAGAS on ACL Anthology (EACL 2024)](https://aclanthology.org/2024.eacl-demo.16/)
  — Peer-reviewed version. Citable in the capstone paper.

---

## GraphRAG for Structured Domains (Legal / Contract Analogues)

- [GraphRAG in Action: Commercial Contracts to Q&A Agent (TDS)](https://towardsdatascience.com/graphrag-in-action-from-commercial-contracts-to-a-dynamic-q-a-agent-7d4a6caa6eb5/)
  — Shows 4-stage GraphRAG pipeline (extraction → knowledge graph → retrieval → agent) on CUAD contracts. Closest published analogue to what CSS_graph_project and Sentinel-RAG do.

- [Agentic GraphRAG for Commercial Contracts (Neo4j)](https://neo4j.com/blog/developer/agentic-graphrag-for-commercial-contracts/)
  — Graph construction patterns for structured legal text. Relevant for cross-reference edge design.

- [GraphRAG for Legal AI: Why Knowledge Graphs Beat Vector Search (Medium)](https://medium.com/@thomasrehmer/graphrag-for-legal-ai-why-knowledge-graphs-beat-vector-search-01436abfe095)
  — Explains why vector search fails on structured cross-referencing text — same failure mode applies to Army FMs.

---

## Military Corpus

- [Heralax/us-army-fm-instruct (Hugging Face)](https://huggingface.co/datasets/Heralax/us-army-fm-instruct)
  — Primary dataset source for Army Field Manuals in instruction-tuning format.

- [Army Publishing Directorate](https://armypubs.army.mil/)
  — Official source for FM 3-0, FM 2-0, FM 3-90, FM 4-0, FM 6-0. Always prefer the official version over cached copies.

- [GlobalSecurity.org Army Intelligence Field Manuals](https://www.globalsecurity.org/intell/library/policy/army/fm/index.html)
  — Useful archive of older FM versions. Cross-reference with official APD for current doctrine.

---

## Comparison Architectures

- [GraphRAG vs HippoRAG vs PathRAG vs OG-RAG (Medium)](https://medium.com/graph-praxis/graphrag-vs-hipporag-vs-pathrag-vs-og-rag-choosing-the-right-architecture-for-your-knowledge-graph-a4745e8b125f)
  — Survey of graph retrieval architectures. Useful for the related-work section of the capstone.

- [Graph RAG in 2026: A Practitioner's Guide (Medium)](https://medium.com/graph-praxis/graph-rag-in-2026-a-practitioners-guide-to-what-actually-works-dca4962e7517)
  — Current state of the field. Confirms Sentinel-RAG's approach (CSS scoring + BFS traversal) is still competitive.

- [RAG vs GraphRAG in 2025: A Builder's Field Guide (Medium)](https://medium.com/@Quaxel/rag-vs-graphrag-in-2025-a-builders-field-guide-82bb33efed81)
  — Practitioner perspective on when to use which architecture. Supports the multi-hop framing of the benchmark.

---

## Statistical Methods

- Wilcoxon signed-rank test: Non-parametric paired comparison. Use `scipy.stats.wilcoxon`. Required for all primary metric comparisons (p < 0.05).
- Holm-Bonferroni correction: Apply across the 4 RAGAS metrics to control familywise error rate.
- Cohen's d: Effect size for all primary comparisons.
- 95% CI: Bootstrap or normal approximation for all mean scores.

Python: `scipy`, `numpy`, `pingouin` for statistical tests. Implementation in `evaluation/statistical.py`.

---

## Notes

- The roommate's project (CSS_graph_project) is the architectural baseline. Always check what they did before designing a new component — either adapt it or consciously diverge.
- RAGAS metrics must come from the real `ragas` library. Any 0.0 result is a bug in the harness, not a valid score.
- Benchmark targets are calibrated against arXiv:2502.11371 (GraphRAG-Bench) and HotpotQA multi-hop baselines. Do not lower targets without a documented reason.
- Temporal decay is the **novel contribution** — it must appear in the ablation study (Sentinel with decay vs Sentinel without decay), not just in the comparison against vanilla RAG.
