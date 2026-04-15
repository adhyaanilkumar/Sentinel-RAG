"""Evaluation metrics for Sentinel-RAG benchmarks.

Combines official RAGAS library metrics (Benchmarks 1-2) with custom
military-specific metrics (Benchmarks 3-5) and structural metrics (Benchmark 6).
"""

from __future__ import annotations

import re
from typing import Optional


# ---------------------------------------------------------------------------
# Custom military-specific metrics (Benchmarks 3, 4, 5)
# ---------------------------------------------------------------------------

def information_unit_coverage(answer: str, information_units: list[str]) -> float:
    """Fraction of information units from the gold checklist present in the answer."""
    if not information_units:
        return 1.0
    covered = 0
    answer_lower = answer.lower()
    for unit in information_units:
        keywords = [w.strip().lower() for w in re.split(r"[,;:\-()]", unit) if len(w.strip()) > 3]
        if not keywords:
            covered += 1
            continue
        matched = sum(1 for kw in keywords if kw in answer_lower)
        if matched / len(keywords) >= 0.5:
            covered += 1
    return covered / len(information_units)


def fatal_error_rate(answer: str, ground_truth: str, override_keywords: list[str]) -> bool:
    """Check if the answer omits a critical override/exception (Trap A metric).
    Returns True if the answer has a fatal error (missing the override)."""
    answer_lower = answer.lower()
    for kw in override_keywords:
        if kw.lower() in answer_lower:
            return False
    return True


def definition_retrieved(
    retrieved_sources: list[str],
    retrieved_texts: list[str],
    definition_fm: str,
    definition_keywords: list[str],
) -> bool:
    """Check if the definition from the correct FM was retrieved (Trap B metric)."""
    for src, text in zip(retrieved_sources, retrieved_texts):
        if definition_fm.lower() in src.lower():
            text_lower = text.lower()
            if any(kw.lower() in text_lower for kw in definition_keywords):
                return True
    return False


def component_recall(answer: str, components: list[str]) -> float:
    """Fraction of scattered components mentioned in the answer (Trap C metric)."""
    if not components:
        return 1.0
    found = 0
    answer_lower = answer.lower()
    for comp in components:
        keywords = [w.strip().lower() for w in comp.split() if len(w.strip()) > 3]
        if not keywords:
            found += 1
            continue
        matched = sum(1 for kw in keywords if kw in answer_lower)
        if matched / len(keywords) >= 0.4:
            found += 1
    return found / len(components)


def evidence_recall(
    retrieved_sources: list[str],
    gold_section_references: list[str],
) -> float:
    """Fraction of gold-standard section references that appear in retrieved set."""
    if not gold_section_references:
        return 1.0
    found = 0
    sources_lower = " ".join(retrieved_sources).lower()
    for ref in gold_section_references:
        fm_match = re.search(r"(FM\s+[\d\-]+)", ref, re.IGNORECASE)
        if fm_match and fm_match.group(1).lower() in sources_lower:
            found += 1
    return found / len(gold_section_references)


def temporal_correctness(
    answer: str,
    recent_info: str,
    old_info: str,
) -> float:
    """Score for whether the answer prefers recent info over old info.
    Returns 1.0 if recent is preferred, 0.5 if both, 0.0 if old preferred."""
    answer_lower = answer.lower()
    has_recent = any(
        kw in answer_lower
        for kw in [w.lower() for w in recent_info.split() if len(w) > 4][:5]
    )
    has_old = any(
        kw in answer_lower
        for kw in [w.lower() for w in old_info.split() if len(w) > 4][:5]
    )
    if has_recent and not has_old:
        return 1.0
    if has_recent and has_old:
        return 0.5
    if has_old and not has_recent:
        return 0.0
    return 0.5


def mrr_at_k(
    retrieved_chunk_ids: list[str],
    relevant_chunk_ids: set[str],
    k: int = 10,
) -> float:
    """Mean Reciprocal Rank at K."""
    for i, cid in enumerate(retrieved_chunk_ids[:k]):
        if cid in relevant_chunk_ids:
            return 1.0 / (i + 1)
    return 0.0


# ---------------------------------------------------------------------------
# ROUGE-L (Benchmark 2)
# ---------------------------------------------------------------------------

def compute_rouge_l(answer: str, reference: str) -> float:
    """Compute ROUGE-L F1 score using the official rouge-score library."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = scorer.score(reference, answer)
        return scores["rougeL"].fmeasure
    except ImportError:
        return _rouge_l_fallback(answer, reference)


def _rouge_l_fallback(answer: str, reference: str) -> float:
    """LCS-based ROUGE-L when rouge-score is unavailable."""
    ans_tokens = answer.lower().split()
    ref_tokens = reference.lower().split()
    if not ans_tokens or not ref_tokens:
        return 0.0
    m, n = len(ref_tokens), len(ans_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == ans_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs_len = dp[m][n]
    precision = lcs_len / n if n > 0 else 0.0
    recall = lcs_len / m if m > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Official RAGAS metrics (Benchmarks 1-2)
# ---------------------------------------------------------------------------

def _build_ragas_llm_and_embeddings():
    """Construct RAGAS 0.4.x LLM and embeddings from the project's OpenAI client.

    Returns (llm, embeddings) or raises if dependencies are missing.
    """
    import os
    from pathlib import Path
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from ragas.embeddings import OpenAIEmbeddings

    # Auto-load .env from the project root if OPENAI_API_KEY is not already set
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line.startswith("OPENAI_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
                    os.environ["OPENAI_API_KEY"] = api_key
                    break

    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set — RAGAS metrics unavailable")

    # RAGAS 0.4.x batch_score() runs async internally — all clients must be async.
    # max_tokens=4096: faithfulness and answer_correctness emit large JSON statement
    # lists that truncate at the default limit, causing all retries to fail.
    async_client = AsyncOpenAI(api_key=api_key)
    llm = llm_factory("gpt-4o-mini", client=async_client, max_tokens=4096)
    embeddings = OpenAIEmbeddings(client=async_client)
    return llm, embeddings


def compute_ragas_metrics(
    query: str,
    answer: str,
    ground_truth: str,
    retrieved_contexts: list[str],
    metric_names: list[str] | None = None,
) -> dict[str, float]:
    """Compute official RAGAS 0.4.x metrics for a single query.

    Returns dict with keys: context_recall, context_precision, faithfulness,
    answer_correctness, answer_relevancy. Missing metrics default to 0.0.
    """
    if metric_names is None:
        metric_names = [
            "context_recall", "context_precision", "faithfulness",
            "answer_correctness", "answer_relevancy",
        ]
    batch = compute_ragas_metrics_batch(
        [query], [answer], [ground_truth], [retrieved_contexts], metric_names
    )
    return batch[0]


def compute_ragas_metrics_batch(
    queries: list[str],
    answers: list[str],
    ground_truths: list[str],
    retrieved_contexts_list: list[list[str]],
    metric_names: list[str] | None = None,
) -> list[dict[str, float]]:
    """Batch RAGAS 0.4.x evaluation using direct per-metric batch_score() calls.

    RAGAS 0.4.x uses a new BaseMetric API (ragas.metrics.collections) that is
    incompatible with the old ragas.evaluate() path — each metric must be called
    individually via its batch_score() method with metric-specific input dicts.
    """
    if metric_names is None:
        metric_names = [
            "context_recall", "context_precision", "faithfulness",
            "answer_correctness", "answer_relevancy",
        ]

    n = len(queries)
    defaults = [{m: 0.0 for m in metric_names} for _ in range(n)]

    try:
        from ragas.metrics.collections import (
            ContextRecall,
            ContextPrecision,
            Faithfulness,
            AnswerCorrectness,
            AnswerRelevancy,
        )

        llm, embeddings = _build_ragas_llm_and_embeddings()

        # Each metric needs a different subset of inputs per RAGAS 0.4.x API.
        # Input key mapping: (metric_class, [required_input_keys], input_builder_fn)
        def _inputs_faithfulness(i):
            return {"user_input": queries[i], "response": answers[i],
                    "retrieved_contexts": retrieved_contexts_list[i]}

        def _inputs_context_recall(i):
            return {"user_input": queries[i], "reference": ground_truths[i],
                    "retrieved_contexts": retrieved_contexts_list[i]}

        def _inputs_context_precision(i):
            return {"user_input": queries[i], "reference": ground_truths[i],
                    "retrieved_contexts": retrieved_contexts_list[i]}

        def _inputs_answer_correctness(i):
            return {"user_input": queries[i], "response": answers[i],
                    "reference": ground_truths[i]}

        def _inputs_answer_relevancy(i):
            return {"user_input": queries[i], "response": answers[i]}

        metric_configs = {
            "faithfulness": (lambda: Faithfulness(llm=llm), _inputs_faithfulness),
            "context_recall": (lambda: ContextRecall(llm=llm), _inputs_context_recall),
            "context_precision": (lambda: ContextPrecision(llm=llm), _inputs_context_precision),
            "answer_correctness": (lambda: AnswerCorrectness(llm=llm, embeddings=embeddings),
                                   _inputs_answer_correctness),
            "answer_relevancy": (lambda: AnswerRelevancy(llm=llm, embeddings=embeddings),
                                 _inputs_answer_relevancy),
        }

        # Accumulate per-query scores; start with defaults
        scores: list[dict[str, float]] = [{m: 0.0 for m in metric_names} for _ in range(n)]

        CHUNK_SIZE = 5      # queries per batch_score() call
        SLEEP_BETWEEN_CHUNKS = 60  # seconds — stay under 200k TPM

        for m_name in metric_names:
            if m_name not in metric_configs:
                continue
            metric_factory, input_builder = metric_configs[m_name]
            try:
                metric = metric_factory()
                all_inputs = [input_builder(i) for i in range(n)]

                # Fire in chunks to avoid blowing the TPM rate limit.
                # 4 systems × 40 queries × 5 metrics ≈ 800 LLM calls;
                # chunking to CHUNK_SIZE keeps bursts manageable.
                import time
                for chunk_start in range(0, n, CHUNK_SIZE):
                    chunk = all_inputs[chunk_start:chunk_start + CHUNK_SIZE]

                    # Retry each chunk up to 5 times with exponential backoff
                    # to survive transient DNS/connection drops.
                    chunk_results = None
                    for attempt, backoff in enumerate([15, 30, 60, 120, 180], start=1):
                        try:
                            chunk_results = metric.batch_score(chunk)
                            break
                        except Exception as conn_exc:
                            print(f"[RAGAS] {m_name} chunk {chunk_start//CHUNK_SIZE+1} attempt {attempt} failed: {conn_exc}")
                            if attempt < 5:
                                print(f"[RAGAS] retrying in {backoff}s...")
                                time.sleep(backoff)

                    if chunk_results is None:
                        print(f"[RAGAS] {m_name} chunk {chunk_start//CHUNK_SIZE+1} gave up after 5 attempts — using 0.0")
                    else:
                        for j, res in enumerate(chunk_results):
                            idx = chunk_start + j
                            scores[idx][m_name] = float(res.value) if res.value is not None else 0.0

                    # Sleep between chunks (skip after the last one)
                    if chunk_start + CHUNK_SIZE < n:
                        print(f"[RAGAS] {m_name}: scored {chunk_start + len(chunk)}/{n} — sleeping {SLEEP_BETWEEN_CHUNKS}s")
                        time.sleep(SLEEP_BETWEEN_CHUNKS)

            except Exception as exc:
                import traceback
                print(f"[RAGAS] metric '{m_name}' failed: {exc}")
                traceback.print_exc()

        return scores

    except Exception as exc:
        import traceback
        print(f"[RAGAS] batch evaluation setup failed: {exc}")
        traceback.print_exc()
        return defaults
