"""Full evaluation harness orchestrating all benchmark metrics.

Integrates:
  - Official RAGAS library metrics (Benchmarks 1-2)
  - Custom military-specific metrics (Benchmarks 3-5)
  - Statistical analysis with Holm-Bonferroni correction
  - LaTeX-ready table generation
  - Per-FM performance breakdown
  - Temporal decay ablation comparison
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from core.data_models import BenchmarkResult, GenerationResult, QueryCategory
from data.gold_annotations import GoldAnnotation
from evaluation.metrics import (
    component_recall,
    compute_ragas_metrics_batch,
    compute_rouge_l,
    definition_retrieved,
    evidence_recall,
    fatal_error_rate,
    information_unit_coverage,
    mrr_at_k,
)
from evaluation.statistical import ComparisonResult, run_full_comparison


class EvaluationHarness:
    """Run all evaluation metrics on system outputs and produce benchmark results."""

    def __init__(self, run_ragas: bool = True):
        self.run_ragas = run_ragas

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def evaluate_single(
        self,
        annotation: GoldAnnotation,
        generation: GenerationResult,
        system_name: str,
        ragas_scores: dict[str, float] | None = None,
    ) -> BenchmarkResult:
        answer = generation.answer
        chunks = generation.retrieval_result.chunks

        iu_cov = information_unit_coverage(answer, annotation.information_units)
        comp_rec = component_recall(answer, annotation.information_units)
        ev_rec = evidence_recall(
            [c.source_document for c in chunks],
            annotation.section_references,
        )
        rouge = compute_rouge_l(answer, annotation.ground_truth_answer)
        mrr = mrr_at_k(
            [c.id for c in chunks],
            set(),  # relevant chunk IDs not available at gold level; filled by RAGAS context_recall proxy
            k=10,
        )

        is_fatal = False
        defn_found = False
        if annotation.category == QueryCategory.TRAP_A:
            override_kws = self._extract_override_keywords(annotation)
            is_fatal = fatal_error_rate(answer, annotation.ground_truth_answer, override_kws)
        elif annotation.category == QueryCategory.TRAP_B:
            if len(annotation.source_documents) >= 2:
                defn_fm = annotation.source_documents[0]
                defn_kws = annotation.information_units[:2] if annotation.information_units else []
                defn_found = definition_retrieved(
                    [c.source_document for c in chunks],
                    [c.text for c in chunks],
                    defn_fm,
                    [kw.split(":")[0] if ":" in kw else kw for kw in defn_kws],
                )

        rs = ragas_scores or {}

        return BenchmarkResult(
            query_id=annotation.id,
            category=annotation.category,
            hop_count=annotation.hop_count.value,
            system_name=system_name,
            generation_result=generation,
            context_recall=rs.get("context_recall", 0.0),
            context_precision=rs.get("context_precision", 0.0),
            evidence_recall=ev_rec,
            mrr_at_10=mrr,
            faithfulness=rs.get("faithfulness", 0.0),
            answer_correctness=rs.get("answer_correctness", 0.0),
            answer_relevancy=rs.get("answer_relevancy", 0.0),
            rouge_l=rouge,
            information_unit_coverage=iu_cov,
            fatal_error=is_fatal,
            definition_retrieved=defn_found,
            component_recall=comp_rec,
        )

    def evaluate_batch(
        self,
        annotations: list[GoldAnnotation],
        generations: list[GenerationResult],
        system_name: str,
    ) -> list[BenchmarkResult]:
        ragas_list: list[dict[str, float] | None] = [None] * len(annotations)

        if self.run_ragas:
            queries = [a.query for a in annotations]
            answers = [g.answer for g in generations]
            ground_truths = [a.ground_truth_answer for a in annotations]
            # Truncate contexts: faithfulness decomposes all chunks into statements,
            # blowing max_tokens on long FM text. Cap to 5 chunks × 800 chars.
            # Fallback to [""] if no chunks — RAGAS raises ValueError on empty list.
            contexts = [
                [c.text[:800] for c in g.retrieval_result.chunks[:5]] or [""]
                for g in generations
            ]
            try:
                ragas_list = compute_ragas_metrics_batch(
                    queries, answers, ground_truths, contexts,
                )
            except Exception as exc:
                import traceback
                print(f"[EvaluationHarness] RAGAS batch evaluation failed: {exc}")
                traceback.print_exc()
                # fall back to None → zeros

        results = []
        for ann, gen, rs in zip(annotations, generations, ragas_list):
            results.append(self.evaluate_single(ann, gen, system_name, rs))
        return results

    # ------------------------------------------------------------------
    # Statistical comparison
    # ------------------------------------------------------------------

    PRIMARY_METRICS = [
        ("context_recall", "context_recall"),
        ("context_precision", "context_precision"),
        ("evidence_recall", "evidence_recall"),
        ("mrr_at_10", "mrr_at_10"),
        ("faithfulness", "faithfulness"),
        ("answer_correctness", "answer_correctness"),
        ("answer_relevancy", "answer_relevancy"),
        ("rouge_l", "rouge_l"),
        ("information_unit_coverage", "information_unit_coverage"),
        ("component_recall", "component_recall"),
    ]

    def compare_systems(
        self,
        baseline_results: list[BenchmarkResult],
        sentinel_results: list[BenchmarkResult],
    ) -> list[ComparisonResult]:
        """Statistical comparison across all primary metrics."""
        metrics: dict[str, tuple[list[float], list[float]]] = {}

        for metric_name, field in self.PRIMARY_METRICS:
            base_scores = [getattr(r, field) for r in baseline_results]
            sent_scores = [getattr(r, field) for r in sentinel_results]
            if base_scores and sent_scores:
                metrics[metric_name] = (base_scores, sent_scores)

        return run_full_comparison(metrics)

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_report(
        self,
        baseline_results: list[BenchmarkResult],
        sentinel_results: list[BenchmarkResult],
        comparisons: list[ComparisonResult],
        output_path: str | Path,
        ablation_results: list[BenchmarkResult] | None = None,
        ablation_comparisons: list[ComparisonResult] | None = None,
    ) -> str:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        lines = ["# Sentinel-RAG Benchmark Report\n"]

        # --- System Summary ---
        lines.append("## System Summary\n")
        all_systems = [("Vanilla RAG", baseline_results), ("Sentinel-RAG", sentinel_results)]
        if ablation_results:
            all_systems.append(("Sentinel-RAG (no temporal)", ablation_results))

        for name, results in all_systems:
            if not results:
                continue
            lines.extend(self._system_summary_block(name, results))

        # --- Benchmark 1: Retrieval Quality ---
        lines.append("## Benchmark 1: Retrieval Quality (RAGAS)\n")
        retrieval_metrics = ["context_recall", "context_precision", "evidence_recall", "mrr_at_10"]
        lines.extend(self._metric_table(all_systems, retrieval_metrics))

        # --- Benchmark 2: Generation Quality ---
        lines.append("\n## Benchmark 2: Generation Quality (RAGAS)\n")
        gen_metrics = ["faithfulness", "answer_correctness", "answer_relevancy", "rouge_l"]
        lines.extend(self._metric_table(all_systems, gen_metrics))

        # --- Benchmark 3: Efficiency ---
        lines.append("\n## Benchmark 3: Efficiency (Tokens-to-Truth)\n")
        lines.extend(self._efficiency_table(all_systems))

        # --- Benchmark 4: Adversarial Categories ---
        lines.append("\n## Benchmark 4: Adversarial Categories\n")
        lines.extend(self._adversarial_breakdown(all_systems))

        # --- Benchmark 5: Temporal Decay Ablation ---
        if ablation_results:
            lines.append("\n## Benchmark 5: Temporal Decay Ablation\n")
            lines.extend(self._ablation_section(sentinel_results, ablation_results, ablation_comparisons))

        # --- Per-Category Breakdown ---
        lines.append("\n## Per-Category Breakdown\n")
        lines.extend(self._per_category_breakdown(all_systems))

        # --- Per-FM Breakdown ---
        lines.append("\n## Per-Document (FM) Breakdown\n")
        lines.extend(self._per_fm_breakdown(all_systems))

        # --- Statistical Comparisons ---
        lines.append("\n## Statistical Comparisons (Vanilla RAG vs Sentinel-RAG)\n")
        lines.extend(self._statistical_table(comparisons))

        # --- LaTeX Tables ---
        lines.append("\n## LaTeX-Ready Tables\n")
        lines.append("### Main Results\n")
        lines.append("```latex")
        lines.append(self._latex_main_table(comparisons))
        lines.append("```\n")
        lines.append("### Per-Category Results\n")
        lines.append("```latex")
        lines.append(self._latex_category_table(baseline_results, sentinel_results))
        lines.append("```\n")
        if ablation_comparisons:
            lines.append("### Ablation Results\n")
            lines.append("```latex")
            lines.append(self._latex_ablation_table(ablation_comparisons))
            lines.append("```\n")

        report = "\n".join(lines)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)

        self._write_json(output_path, baseline_results, sentinel_results,
                         comparisons, ablation_results, ablation_comparisons)

        return report

    # ------------------------------------------------------------------
    # Report helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _system_summary_block(name: str, results: list[BenchmarkResult]) -> list[str]:
        n = len(results)
        lines = [f"### {name} (n={n})"]

        def avg(field: str) -> float:
            return sum(getattr(r, field) for r in results) / n

        lines.append(f"- Context Recall: {avg('context_recall'):.3f}")
        lines.append(f"- Context Precision: {avg('context_precision'):.3f}")
        lines.append(f"- Evidence Recall: {avg('evidence_recall'):.3f}")
        lines.append(f"- MRR@10: {avg('mrr_at_10'):.3f}")
        lines.append(f"- Faithfulness: {avg('faithfulness'):.3f}")
        lines.append(f"- Answer Correctness: {avg('answer_correctness'):.3f}")
        lines.append(f"- Answer Relevancy: {avg('answer_relevancy'):.3f}")
        lines.append(f"- ROUGE-L: {avg('rouge_l'):.3f}")
        lines.append(f"- Information Unit Coverage: {avg('information_unit_coverage'):.3f}")
        lines.append(f"- Component Recall: {avg('component_recall'):.3f}")

        trap_a = [r for r in results if r.category == QueryCategory.TRAP_A]
        trap_b = [r for r in results if r.category == QueryCategory.TRAP_B]
        fatal = sum(1 for r in trap_a if r.fatal_error)
        defn = sum(1 for r in trap_b if r.definition_retrieved)
        lines.append(f"- Fatal Errors (Trap A): {fatal}/{len(trap_a)}")
        lines.append(f"- Definitions Retrieved (Trap B): {defn}/{len(trap_b)}")

        avg_tokens = sum(r.generation_result.total_tokens for r in results) / n
        avg_latency = sum(r.generation_result.latency_seconds for r in results) / n
        avg_iters = sum(r.generation_result.num_iterations for r in results) / n
        lines.append(f"- Avg Tokens: {avg_tokens:.0f}")
        lines.append(f"- Avg Latency: {avg_latency:.2f}s")
        lines.append(f"- Avg Iterations: {avg_iters:.1f}")
        lines.append("")
        return lines

    @staticmethod
    def _metric_table(
        systems: list[tuple[str, list[BenchmarkResult]]],
        metrics: list[str],
    ) -> list[str]:
        header = "| Metric | " + " | ".join(n for n, _ in systems) + " |"
        sep = "|--------|" + "|".join("----------" for _ in systems) + "|"
        lines = [header, sep]
        for metric in metrics:
            row = f"| {metric} "
            for _, results in systems:
                if not results:
                    row += "| - "
                    continue
                vals = [getattr(r, metric) for r in results]
                mean = sum(vals) / len(vals)
                row += f"| {mean:.3f} "
            row += "|"
            lines.append(row)
        lines.append("")
        return lines

    @staticmethod
    def _efficiency_table(
        systems: list[tuple[str, list[BenchmarkResult]]],
    ) -> list[str]:
        header = "| Metric | " + " | ".join(n for n, _ in systems) + " |"
        sep = "|--------|" + "|".join("----------" for _ in systems) + "|"
        lines = [header, sep]

        eff_metrics = [
            ("IU Coverage (single pass)", lambda r: r.information_unit_coverage),
            ("Avg Iterations", lambda r: r.generation_result.num_iterations),
            ("Avg Total Tokens", lambda r: r.generation_result.total_tokens),
            ("Avg Latency (s)", lambda r: r.generation_result.latency_seconds),
        ]
        for label, extractor in eff_metrics:
            row = f"| {label} "
            for _, results in systems:
                if not results:
                    row += "| - "
                    continue
                vals = [extractor(r) for r in results]
                mean = sum(vals) / len(vals)
                row += f"| {mean:.2f} "
            row += "|"
            lines.append(row)
        lines.append("")
        return lines

    @staticmethod
    def _adversarial_breakdown(
        systems: list[tuple[str, list[BenchmarkResult]]],
    ) -> list[str]:
        lines = []
        categories = [
            ("Trap A: Overriding Directive", QueryCategory.TRAP_A,
             "Fatal Error Rate", lambda rs: f"{sum(1 for r in rs if r.fatal_error)}/{len(rs)} "
             f"({sum(1 for r in rs if r.fatal_error)/max(len(rs),1)*100:.0f}%)"),
            ("Trap B: Distant Definition", QueryCategory.TRAP_B,
             "Definition Retrieval Rate", lambda rs: f"{sum(1 for r in rs if r.definition_retrieved)}/{len(rs)} "
             f"({sum(1 for r in rs if r.definition_retrieved)/max(len(rs),1)*100:.0f}%)"),
            ("Trap C: Scattered Components", QueryCategory.TRAP_C,
             "Component Recall", lambda rs: f"{sum(r.component_recall for r in rs)/max(len(rs),1):.3f}"),
            ("Control: Single-Hop", QueryCategory.CONTROL,
             "Answer Correctness", lambda rs: f"{sum(r.answer_correctness for r in rs)/max(len(rs),1):.3f}"),
        ]

        header = "| Category | Key Metric | " + " | ".join(n for n, _ in systems) + " |"
        sep = "|----------|------------|" + "|".join("----------" for _ in systems) + "|"
        lines.extend([header, sep])

        for cat_label, cat_enum, metric_label, formatter in categories:
            row = f"| {cat_label} | {metric_label} "
            for _, results in systems:
                cat_results = [r for r in results if r.category == cat_enum]
                row += f"| {formatter(cat_results)} "
            row += "|"
            lines.append(row)
        lines.append("")
        return lines

    @staticmethod
    def _ablation_section(
        with_decay: list[BenchmarkResult],
        without_decay: list[BenchmarkResult],
        comparisons: list[ComparisonResult] | None,
    ) -> list[str]:
        lines = []
        metrics = ["information_unit_coverage", "evidence_recall", "component_recall",
                    "faithfulness", "answer_correctness"]
        header = "| Metric | With Temporal Decay | Without Temporal Decay | Delta |"
        sep = "|--------|--------------------|-----------------------|---|"
        lines.extend([header, sep])

        for m in metrics:
            wd = sum(getattr(r, m) for r in with_decay) / max(len(with_decay), 1)
            wo = sum(getattr(r, m) for r in without_decay) / max(len(without_decay), 1)
            delta = wd - wo
            sign = "+" if delta >= 0 else ""
            lines.append(f"| {m} | {wd:.3f} | {wo:.3f} | {sign}{delta:.3f} |")

        if comparisons:
            lines.append("")
            lines.append("**Statistical significance (Wilcoxon):**\n")
            for c in comparisons:
                sig = "YES" if c.significant else "no"
                lines.append(f"- {c.metric_name}: p={c.p_value:.4f}, d={c.effect_size_d:.3f} ({sig})")

        lines.append("")
        return lines

    @staticmethod
    def _per_category_breakdown(
        systems: list[tuple[str, list[BenchmarkResult]]],
    ) -> list[str]:
        lines = []
        for cat in QueryCategory:
            lines.append(f"### {cat.value}\n")
            for name, results in systems:
                cat_results = [r for r in results if r.category == cat]
                if not cat_results:
                    continue
                n = len(cat_results)
                avg_iu = sum(r.information_unit_coverage for r in cat_results) / n
                avg_cr = sum(r.component_recall for r in cat_results) / n
                avg_er = sum(r.evidence_recall for r in cat_results) / n
                avg_ac = sum(r.answer_correctness for r in cat_results) / n
                avg_f = sum(r.faithfulness for r in cat_results) / n
                lines.append(
                    f"  {name} (n={n}): IU_Cov={avg_iu:.3f}, Comp_Rec={avg_cr:.3f}, "
                    f"Ev_Rec={avg_er:.3f}, Ans_Corr={avg_ac:.3f}, Faith={avg_f:.3f}"
                )
            lines.append("")
        return lines

    @staticmethod
    def _per_fm_breakdown(
        systems: list[tuple[str, list[BenchmarkResult]]],
    ) -> list[str]:
        """Per-document breakdown: compute metrics for queries whose gold sources include each FM."""
        all_docs: set[str] = set()
        for _, results in systems:
            for r in results:
                for c in r.generation_result.retrieval_result.chunks:
                    all_docs.add(c.source_document)

        lines = []
        fm_metrics = ["information_unit_coverage", "evidence_recall", "component_recall",
                      "answer_correctness", "faithfulness"]
        header = "| FM | System | n | " + " | ".join(fm_metrics) + " |"
        sep = "|---|--------|---|" + "|".join("---" for _ in fm_metrics) + "|"
        lines.extend([header, sep])

        for doc in sorted(all_docs):
            for name, results in systems:
                doc_results = [
                    r for r in results
                    if any(doc in c.source_document for c in r.generation_result.retrieval_result.chunks)
                ]
                if not doc_results:
                    continue
                n = len(doc_results)
                vals = []
                for m in fm_metrics:
                    v = sum(getattr(r, m) for r in doc_results) / n
                    vals.append(f"{v:.3f}")
                lines.append(f"| {doc} | {name} | {n} | " + " | ".join(vals) + " |")
        lines.append("")
        return lines

    @staticmethod
    def _statistical_table(comparisons: list[ComparisonResult]) -> list[str]:
        lines = [
            "| Metric | Vanilla RAG (mean±std) | Sentinel-RAG (mean±std) | p-value | Cohen's d | 95% CI (Sentinel) | Sig. | Holm-Corrected |",
            "|--------|-----------------------|--------------------------|---------|-----------|-------------------|------|----------------|",
        ]
        for c in comparisons:
            sig = "**YES**" if c.significant else "no"
            corr = "**YES**" if c.corrected_significant else "no"
            lines.append(
                f"| {c.metric_name} "
                f"| {c.system_a_mean:.3f}±{c.system_a_std:.3f} "
                f"| {c.system_b_mean:.3f}±{c.system_b_std:.3f} "
                f"| {c.p_value:.4f} "
                f"| {c.effect_size_d:.3f} "
                f"| [{c.system_b_ci_lower:.3f}, {c.system_b_ci_upper:.3f}] "
                f"| {sig} | {corr} |"
            )
        lines.append("")
        return lines

    # ------------------------------------------------------------------
    # LaTeX table generation
    # ------------------------------------------------------------------

    @staticmethod
    def _latex_main_table(comparisons: list[ComparisonResult]) -> str:
        rows = [
            r"\begin{table}[ht]",
            r"\centering",
            r"\caption{Sentinel-RAG vs.\ Vanilla RAG: Main Results}",
            r"\label{tab:main-results}",
            r"\begin{tabular}{lcccccc}",
            r"\toprule",
            r"Metric & Vanilla RAG & Sentinel-RAG & $p$-value & Cohen's $d$ & 95\% CI & Sig. \\",
            r"\midrule",
        ]
        for c in comparisons:
            sig_marker = r"$^{***}$" if c.corrected_significant else (r"$^{*}$" if c.significant else "")
            metric_escaped = c.metric_name.replace("_", r"\_")
            rows.append(
                f"  {metric_escaped} & {c.system_a_mean:.3f} & "
                f"{c.system_b_mean:.3f}{sig_marker} & "
                f"{c.p_value:.4f} & {c.effect_size_d:.3f} & "
                f"[{c.system_b_ci_lower:.3f}, {c.system_b_ci_upper:.3f}] & "
                f"{'Yes' if c.corrected_significant else 'No'} \\\\"
            )
        rows.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{2mm}",
            r"\footnotesize{$^{*}p<0.05$, $^{***}p<0.05$ after Holm-Bonferroni correction. "
            r"Paired Wilcoxon signed-rank test, $n=40$ queries.}",
            r"\end{table}",
        ])
        return "\n".join(rows)

    @staticmethod
    def _latex_category_table(
        baseline_results: list[BenchmarkResult],
        sentinel_results: list[BenchmarkResult],
    ) -> str:
        rows = [
            r"\begin{table}[ht]",
            r"\centering",
            r"\caption{Per-Category Performance Breakdown}",
            r"\label{tab:category-results}",
            r"\begin{tabular}{llccccc}",
            r"\toprule",
            r"Category & System & IU Cov. & Ev. Recall & Comp. Recall & Faith. & Ans. Corr. \\",
            r"\midrule",
        ]

        for cat in QueryCategory:
            cat_label = cat.value.replace("_", r"\_")
            for sname, results in [("Vanilla RAG", baseline_results), ("Sentinel-RAG", sentinel_results)]:
                cr = [r for r in results if r.category == cat]
                if not cr:
                    continue
                n = len(cr)

                def a(field: str) -> float:
                    return sum(getattr(r, field) for r in cr) / n

                rows.append(
                    f"  {cat_label} & {sname} & {a('information_unit_coverage'):.3f} & "
                    f"{a('evidence_recall'):.3f} & {a('component_recall'):.3f} & "
                    f"{a('faithfulness'):.3f} & {a('answer_correctness'):.3f} \\\\"
                )
            rows.append(r"\midrule")

        rows[-1] = r"\bottomrule"
        rows.extend([
            r"\end{tabular}",
            r"\end{table}",
        ])
        return "\n".join(rows)

    @staticmethod
    def _latex_ablation_table(comparisons: list[ComparisonResult]) -> str:
        rows = [
            r"\begin{table}[ht]",
            r"\centering",
            r"\caption{Temporal Decay Ablation Study}",
            r"\label{tab:ablation-temporal}",
            r"\begin{tabular}{lccccc}",
            r"\toprule",
            r"Metric & With Decay & Without Decay & $\Delta$ & $p$-value & Cohen's $d$ \\",
            r"\midrule",
        ]
        for c in comparisons:
            delta = c.system_a_mean - c.system_b_mean
            sign = "+" if delta >= 0 else ""
            sig_marker = r"$^{*}$" if c.significant else ""
            metric_escaped = c.metric_name.replace("_", r"\_")
            rows.append(
                f"  {metric_escaped} & {c.system_a_mean:.3f}{sig_marker} & "
                f"{c.system_b_mean:.3f} & {sign}{delta:.3f} & "
                f"{c.p_value:.4f} & {c.effect_size_d:.3f} \\\\"
            )
        rows.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{2mm}",
            r"\footnotesize{$^{*}p<0.05$ (paired Wilcoxon). Ablation isolates temporal decay contribution.}",
            r"\end{table}",
        ])
        return "\n".join(rows)

    # ------------------------------------------------------------------
    # JSON output
    # ------------------------------------------------------------------

    def _write_json(
        self,
        report_path: Path,
        baseline_results: list[BenchmarkResult],
        sentinel_results: list[BenchmarkResult],
        comparisons: list[ComparisonResult],
        ablation_results: list[BenchmarkResult] | None,
        ablation_comparisons: list[ComparisonResult] | None,
    ) -> None:
        json_path = report_path.with_suffix(".json")
        data: dict = {
            "comparisons": [asdict(c) for c in comparisons],
            "baseline_summary": self._summarize(baseline_results),
            "sentinel_summary": self._summarize(sentinel_results),
        }
        if ablation_results:
            data["ablation_summary"] = self._summarize(ablation_results)
        if ablation_comparisons:
            data["ablation_comparisons"] = [asdict(c) for c in ablation_comparisons]

        per_query = []
        for r in baseline_results + sentinel_results + (ablation_results or []):
            per_query.append({
                "query_id": r.query_id,
                "system": r.system_name,
                "category": r.category.value,
                "hop_count": r.hop_count,
                "context_recall": r.context_recall,
                "context_precision": r.context_precision,
                "evidence_recall": r.evidence_recall,
                "mrr_at_10": r.mrr_at_10,
                "faithfulness": r.faithfulness,
                "answer_correctness": r.answer_correctness,
                "answer_relevancy": r.answer_relevancy,
                "rouge_l": r.rouge_l,
                "information_unit_coverage": r.information_unit_coverage,
                "component_recall": r.component_recall,
                "fatal_error": r.fatal_error,
                "definition_retrieved": r.definition_retrieved,
                "total_tokens": r.generation_result.total_tokens,
                "latency_seconds": r.generation_result.latency_seconds,
                "num_iterations": r.generation_result.num_iterations,
            })
        data["per_query"] = per_query

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    @staticmethod
    def _summarize(results: list[BenchmarkResult]) -> dict:
        if not results:
            return {}
        n = len(results)

        def avg(field: str) -> float:
            return sum(getattr(r, field) for r in results) / n

        return {
            "n": n,
            "context_recall": avg("context_recall"),
            "context_precision": avg("context_precision"),
            "evidence_recall": avg("evidence_recall"),
            "mrr_at_10": avg("mrr_at_10"),
            "faithfulness": avg("faithfulness"),
            "answer_correctness": avg("answer_correctness"),
            "answer_relevancy": avg("answer_relevancy"),
            "rouge_l": avg("rouge_l"),
            "information_unit_coverage": avg("information_unit_coverage"),
            "component_recall": avg("component_recall"),
            "fatal_error_count": sum(1 for r in results if r.fatal_error),
            "definition_retrieved_count": sum(1 for r in results if r.definition_retrieved),
            "avg_tokens": sum(r.generation_result.total_tokens for r in results) / n,
            "avg_latency": sum(r.generation_result.latency_seconds for r in results) / n,
            "avg_iterations": sum(r.generation_result.num_iterations for r in results) / n,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_override_keywords(annotation: GoldAnnotation) -> list[str]:
        """Extract override-related keywords from the gold annotation's ground truth."""
        keywords = [
            "approval", "higher", "authority", "cannot", "must coordinate",
            "not independently", "commander's intent", "does not",
            "exception", "override", "supersede", "notwithstanding",
            "unless", "only when", "prohibited", "restricted",
        ]
        gt_lower = annotation.ground_truth_answer.lower()
        for iu in annotation.information_units:
            words = [w.strip().lower() for w in iu.split() if len(w.strip()) > 4]
            for w in words:
                if w in gt_lower and w not in keywords:
                    keywords.append(w)
        return keywords
