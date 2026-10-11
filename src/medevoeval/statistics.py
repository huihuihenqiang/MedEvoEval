from __future__ import annotations

import math
from statistics import mean


def _binom_two_sided(k: int, n: int) -> float:
    if n == 0:
        return 1.0
    try:
        from scipy.stats import binomtest

        return float(binomtest(k, n, 0.5, alternative="two-sided").pvalue)
    except Exception:
        observed = math.comb(n, k) * (0.5**n)
        return min(1.0, sum(math.comb(n, i) * (0.5**n) for i in range(n + 1) if math.comb(n, i) * (0.5**n) <= observed + 1e-15))


def wilcoxon_signed_rank(x: list[float], y: list[float]) -> dict:
    pairs = [(float(a), float(b)) for a, b in zip(x, y) if math.isfinite(float(a)) and math.isfinite(float(b))]
    if not pairs:
        return {"p_value": math.nan, "effect_size": math.nan, "n": 0, "statistic": math.nan}
    diffs = [b - a for a, b in pairs]
    nonzero = [d for d in diffs if abs(d) > 0]
    if not nonzero:
        return {"p_value": 1.0, "effect_size": 0.0, "n": len(pairs), "statistic": 0.0}
    try:
        from scipy.stats import wilcoxon

        result = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
        p_value = float(result.pvalue)
        statistic = float(result.statistic)
    except Exception:
        positives = sum(1 for d in nonzero if d > 0)
        p_value = _binom_two_sided(min(positives, len(nonzero) - positives), len(nonzero))
        statistic = float(positives)
    effect_size = (sum(1 for d in nonzero if d > 0) - sum(1 for d in nonzero if d < 0)) / float(len(nonzero))
    return {"p_value": p_value, "effect_size": effect_size, "n": len(pairs), "statistic": statistic}


def exact_mcnemar(x: list[int], y: list[int]) -> dict:
    pairs = [(int(a), int(b)) for a, b in zip(x, y)]
    b = sum(1 for a, c in pairs if a == 1 and c == 0)
    c = sum(1 for a, c2 in pairs if a == 0 and c2 == 1)
    p_value = _binom_two_sided(min(b, c), b + c)
    effect = (c - b) / float(b + c) if (b + c) else 0.0
    return {"p_value": p_value, "effect_size": effect, "n": len(pairs), "b": b, "c": c}


def paired_summary(rows: list[dict], metric: str, h0_col: str, h3_col: str, binary: bool = False) -> dict:
    h0 = [float(row[h0_col]) for row in rows if row.get(h0_col) not in {None, ""} and row.get(h3_col) not in {None, ""}]
    h3 = [float(row[h3_col]) for row in rows if row.get(h0_col) not in {None, ""} and row.get(h3_col) not in {None, ""}]
    if binary:
        test = exact_mcnemar([int(v) for v in h0], [int(v) for v in h3])
        test_name = "exact_mcnemar"
    else:
        test = wilcoxon_signed_rank(h0, h3)
        test_name = "wilcoxon_signed_rank"
    return {
        "metric": metric,
        "H0_mean": mean(h0) if h0 else math.nan,
        "H3_mean": mean(h3) if h3 else math.nan,
        "delta": (mean(h3) - mean(h0)) if h0 and h3 else math.nan,
        "test": test_name,
        "p_value": test["p_value"],
        "effect_size": test["effect_size"],
        "n": test["n"],
    }

