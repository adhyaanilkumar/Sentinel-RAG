"""Statistical analysis for benchmark comparisons."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ComparisonResult:
    metric_name: str
    system_a_name: str
    system_b_name: str
    system_a_mean: float
    system_b_mean: float
    system_a_std: float
    system_b_std: float
    system_a_ci_lower: float
    system_a_ci_upper: float
    system_b_ci_lower: float
    system_b_ci_upper: float
    p_value: float
    effect_size_d: float
    significant: bool
    corrected_significant: bool


def paired_wilcoxon(a: list[float], b: list[float]) -> float:
    """Paired Wilcoxon signed-rank test. Returns p-value."""
    from scipy.stats import wilcoxon
    diffs = [ai - bi for ai, bi in zip(a, b)]
    if all(d == 0 for d in diffs):
        return 1.0
    try:
        _, p = wilcoxon(diffs, alternative="greater")
        return float(p)
    except ValueError:
        return 1.0


def cohens_d(a: list[float], b: list[float]) -> float:
    """Cohen's d effect size for paired samples."""
    diffs = np.array([ai - bi for ai, bi in zip(a, b)])
    if diffs.std() == 0:
        return 0.0
    return float(diffs.mean() / diffs.std())


def confidence_interval_95(values: list[float]) -> tuple[float, float]:
    """95% confidence interval using t-distribution."""
    from scipy.stats import t as t_dist
    arr = np.array(values)
    n = len(arr)
    if n < 2:
        return (arr.mean(), arr.mean())
    mean = arr.mean()
    se = arr.std(ddof=1) / np.sqrt(n)
    t_crit = t_dist.ppf(0.975, df=n - 1)
    return (float(mean - t_crit * se), float(mean + t_crit * se))


def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Holm-Bonferroni correction for multiple comparisons.
    Returns list of booleans: True if significant after correction."""
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    significant = [False] * n
    for rank, (orig_idx, p) in enumerate(indexed):
        adjusted_alpha = alpha / (n - rank)
        if p <= adjusted_alpha:
            significant[orig_idx] = True
        else:
            break
    return significant


def compare_systems(
    metric_name: str,
    system_a_name: str,
    system_a_scores: list[float],
    system_b_name: str,
    system_b_scores: list[float],
    alpha: float = 0.05,
) -> ComparisonResult:
    """Full statistical comparison between two systems on one metric."""
    a_mean = np.mean(system_a_scores)
    b_mean = np.mean(system_b_scores)
    a_std = np.std(system_a_scores, ddof=1) if len(system_a_scores) > 1 else 0.0
    b_std = np.std(system_b_scores, ddof=1) if len(system_b_scores) > 1 else 0.0
    a_ci = confidence_interval_95(system_a_scores)
    b_ci = confidence_interval_95(system_b_scores)
    p = paired_wilcoxon(system_b_scores, system_a_scores)
    d = cohens_d(system_b_scores, system_a_scores)

    return ComparisonResult(
        metric_name=metric_name,
        system_a_name=system_a_name,
        system_b_name=system_b_name,
        system_a_mean=float(a_mean),
        system_b_mean=float(b_mean),
        system_a_std=float(a_std),
        system_b_std=float(b_std),
        system_a_ci_lower=a_ci[0],
        system_a_ci_upper=a_ci[1],
        system_b_ci_lower=b_ci[0],
        system_b_ci_upper=b_ci[1],
        p_value=p,
        effect_size_d=d,
        significant=p < alpha,
        corrected_significant=False,
    )


def run_full_comparison(
    metrics: dict[str, tuple[list[float], list[float]]],
    system_a_name: str = "Vanilla RAG",
    system_b_name: str = "Sentinel-RAG",
    alpha: float = 0.05,
) -> list[ComparisonResult]:
    """Run comparisons on multiple metrics with Holm-Bonferroni correction."""
    results = []
    for metric_name, (a_scores, b_scores) in metrics.items():
        result = compare_systems(metric_name, system_a_name, a_scores, system_b_name, b_scores, alpha)
        results.append(result)

    p_values = [r.p_value for r in results]
    corrected = holm_bonferroni(p_values, alpha)
    for r, is_sig in zip(results, corrected):
        r.corrected_significant = is_sig

    return results
