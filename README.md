# Sentinel-RAG

Sentinel-RAG is a research codebase for **graph-augmented retrieval-augmented generation (GraphRAG)** aimed at **military doctrine and field-manual corpora**. It combines dense vector search over chunked PDFs with a **NetworkX knowledge graph** (entities, same-document adjacency, and cross-reference edges), optional **temporal decay** over chunk timestamps, and a **Contrastive Subgraph Scoring (CSS)**-style reranking stage. The repository also ships **vanilla** and **iterative** RAG baselines and an evaluation harness driven by **gold annotations** and **RAGAS** metrics.

## Features

- **Corpus ingestion**: PDF text extraction (PyMuPDF-first), chunking, and metadata such as source FM and section IDs.
- **FAISS vector index**: Embedding model configurable in `config/config.yaml` (default: `all-MiniLM-L6-v2`).
- **Knowledge graph**: Entity extraction, weighted edges, cross-reference detection via regex patterns tuned for military citations (FM, ATP, chapters, and similar).
- **Graph retriever**: Seeds from the vector store, expands along the graph within a hop budget, merges scores with CSS weights, applies source-diversity and redundancy handling, and can **skip graph expansion** when initial retrieval scores are already high.
- **Temporal decay**: Optional exponential decay and stale-chunk warnings for generation prompts.
- **Baselines and benchmarks**: Scripts for vanilla RAG, iterative RAG, full multi-system benchmarks with checkpointing, and standalone graph-quality metrics.

## Requirements

- **Python**: 3.10 or newer recommended.
- **OS**: Developed on Windows; paths in scripts use `pathlib` and should run on Linux and macOS as well.

Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

- **`config/config.yaml`**: LLM provider and models, embedding model, chunking, graph heuristics, CSS weights, temporal decay, and evaluation settings.
- **`.env`** (create at the repository root; do not commit secrets):

  | Variable | Purpose |
  |----------|---------|
  | `OPENAI_API_KEY` | Required when `llm_provider` is `openai`. |
  | `GEMINI_API_KEY` | Required when `llm_provider` is `gemini`. |

Scripts load `.env` via `python-dotenv`.

## Data layout

| Path | Contents |
|------|----------|
| `data/corpus/` | Input PDFs (field manuals and similar). Filenames are parsed to infer FM labels where possible. |
| `data/index/` | Built FAISS index and chunk store (created by `build_index.py`; gitignored when present). |
| `data/graph/` | Pickled graph and node payloads (created by `build_graph.py`; large artifacts may be gitignored). |
| `data/results/` | Benchmark outputs, checkpoints, and reports (gitignored). |

Place your corpus under `data/corpus/` before building artifacts.

## Building indexes and graph

From the repository root (ensure `PYTHONPATH` includes the project root, or run scripts as shown so they adjust `sys.path`):

```bash
python scripts/build_index.py
python scripts/build_graph.py
```

This reads `config/config.yaml`, chunks PDFs from `data/corpus/`, writes the vector store to `data/index/`, and writes `graph.gpickle` and `nodes.pkl` under `data/graph/`.

## Running benchmarks and evaluation

- **Baselines only** (vector store must exist):

  ```bash
  python scripts/run_baselines.py
  ```

- **Full benchmark suite** (vector store + graph artifacts; uses LLM APIs and may take a long time):

  ```bash
  python scripts/run_full_benchmark.py
  ```

  Use `--fresh` to ignore `data/results/benchmark_checkpoint.pkl` and rerun all query phases from scratch. The suite can resume from checkpoints when the file exists.

- **Graph quality metrics** (loads built graph from `data/graph/`):

  ```bash
  python scripts/evaluate_graph.py
  ```

Outputs such as `benchmark_report.md`, JSON summaries, and per-system result files are written under `data/results/` by the benchmark scripts.

## Tests

```bash
python -m pytest tests/
```

## Project structure (high level)

| Area | Role |
|------|------|
| `core/` | Config loading, embeddings, document processing, shared data models. |
| `retrieval/` | FAISS vector store and graph-aware retriever. |
| `graph/` | Knowledge graph construction, entity extraction, neighbor queries. |
| `sentinel/` | `SentinelRAG` orchestration and temporal decay. |
| `generation/` | LLM client (OpenAI / Gemini) and prompts. |
| `baselines/` | Vanilla and iterative RAG for comparison. |
| `evaluation/` | Harness, RAGAS-oriented metrics, statistical helpers. |
| `data/` | Gold annotations and corpus/index/graph directories. |
| `scripts/` | CLI entry points for indexing, graph build, and benchmarks. |

## Citation

If you use this repository in academic or product work, cite or link to the project as appropriate for your venue.

## Acknowledgments

Retrieval scoring and graph heuristics draw on ideas from contrastive subgraph scoring and graph-pruning literature; see inline comments in `retrieval/graph_retriever.py` and related modules for specific references to prior work where noted in code.
