"""Diagnostic metrics with study-level bootstrap 95% CIs.

Bootstrap resamples *studies*, not images: images of one study are correlated (same patient,
same technologist), so resampling images would understate the uncertainty.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
from scipy.stats import rankdata

Metric = Callable[[np.ndarray, np.ndarray, np.ndarray], float]

# Metrics are plain numpy (not sklearn) because the bootstrap evaluates them ~10^5-10^6 times;
# equivalence with sklearn is covered by tests.


def sensitivity(y: np.ndarray, prob: np.ndarray, pred: np.ndarray) -> float:
    pos = y == 1
    return float(pred[pos].mean()) if pos.any() else math.nan


def specificity(y: np.ndarray, prob: np.ndarray, pred: np.ndarray) -> float:
    neg = y == 0
    return float(1 - pred[neg].mean()) if neg.any() else math.nan


def balanced_accuracy(y: np.ndarray, prob: np.ndarray, pred: np.ndarray) -> float:
    return float(np.nanmean([sensitivity(y, prob, pred), specificity(y, prob, pred)]))


def f1(y: np.ndarray, prob: np.ndarray, pred: np.ndarray) -> float:
    tp = float(np.sum((pred == 1) & (y == 1)))
    denom = float(np.sum(pred == 1) + np.sum(y == 1))
    return 2 * tp / denom if denom else 0.0


def roc_auc(y: np.ndarray, prob: np.ndarray, pred: np.ndarray) -> float:
    """Mann-Whitney U / (n_pos * n_neg), ties counted as 1/2 (identical to sklearn)."""
    n_pos = int(np.sum(y == 1))
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return math.nan
    ranks = rankdata(prob)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def pr_auc(y: np.ndarray, prob: np.ndarray, pred: np.ndarray) -> float:
    """Average precision, step-wise over distinct thresholds (sklearn's definition)."""
    n_pos = int(np.sum(y == 1))
    if n_pos == 0 or n_pos == len(y):
        return math.nan
    order = np.argsort(-prob, kind="mergesort")
    p_sorted, y_sorted = prob[order], y[order]
    last_of_run = np.r_[np.flatnonzero(np.diff(p_sorted)), len(p_sorted) - 1]
    tp = np.cumsum(y_sorted)[last_of_run]
    fp = (last_of_run + 1) - tp
    precision = tp / (tp + fp)
    recall = tp / n_pos
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


BINARY_METRICS: dict[str, Metric] = {
    "sensitivity": sensitivity,
    "specificity": specificity,
    "balanced_accuracy": balanced_accuracy,
    "f1": f1,
    "roc_auc": roc_auc,
    "pr_auc": pr_auc,
}


def best_score_threshold(y: np.ndarray, score: np.ndarray) -> float:
    """Cut on any real-valued score maximizing F1 (ties -> better balanced accuracy -> higher cut).

    Candidates are midpoints between consecutive distinct scores plus one below the minimum
    (flag everything) and one above the maximum (flag nothing), so the cut never sits exactly on
    a training example. Without positives the "flag nothing" cut is returned.
    """
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=np.float64)
    if not np.isfinite(score).all():
        raise ValueError("scores must be finite")
    values = np.unique(score)
    above = float(values[-1] + 1.0)
    if not y.any():
        return above
    candidates = np.concatenate([[values[0] - 1.0], (values[:-1] + values[1:]) / 2, [above]])
    best = (-1.0, -1.0, above)
    for t in candidates:
        pred = (score >= t).astype(int)
        key = (f1(y, score, pred), balanced_accuracy(y, score, pred), float(t))
        if key > best:
            best = key
    return best[2]


def best_f1_threshold(y: np.ndarray, prob: np.ndarray) -> float:
    """:func:`best_score_threshold` for probabilities, clipped to ``[0, 1]``."""
    return float(np.clip(best_score_threshold(y, prob), 0.0, 1.0))


def macro_f1(y: np.ndarray, pred: np.ndarray) -> float:
    """Macro-F1 over violation types (columns) for the multi-label output."""
    y, pred = np.atleast_2d(y), np.atleast_2d(pred)
    return float(np.mean([f1(y[:, j], y[:, j], pred[:, j]) for j in range(y.shape[1])]))


def bootstrap_ci(
    metric: Callable[[np.ndarray], float],
    groups: Sequence[str],
    *,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile CI of ``metric(index_array)`` under resampling of whole groups.

    ``metric`` receives row indices (with repeats) and must return a float (NaN is skipped,
    e.g. a resample without positives has no AUC).
    """
    if n_boot <= 0:
        return math.nan, math.nan
    groups = np.asarray(groups)
    unique, inverse = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inverse == g) for g in range(len(unique))]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        picked = rng.integers(0, len(unique), len(unique))
        idx = np.concatenate([members[g] for g in picked])
        v = metric(idx)
        if np.isfinite(v):
            values.append(v)
    if len(values) < n_boot // 2:
        return math.nan, math.nan
    lo, hi = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def binary_report(
    y: np.ndarray,
    probs: Sequence[np.ndarray] | np.ndarray,
    preds: Sequence[np.ndarray] | np.ndarray,
    groups: Sequence[str],
    *,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, object]:
    """All binary metrics with 95% CIs, JSON-serializable.

    ``probs``/``preds`` may hold several CV repetitions (list of arrays): the point estimate is
    the mean over repetitions and every bootstrap resample recomputes that same mean, so the CI
    and the point estimate describe the same quantity.
    """
    y = np.asarray(y).astype(int)
    if isinstance(probs, np.ndarray) and probs.ndim == 1:
        probs, preds = [probs], [preds]
    reps = [(np.asarray(p, float), np.asarray(c).astype(int)) for p, c in zip(probs, preds, strict=True)]
    report: dict[str, object] = {"n": len(y), "n_positive": int(y.sum()), "cv_repetitions": len(reps)}
    for name, fn in BINARY_METRICS.items():

        def over_reps(idx: np.ndarray, fn: Metric = fn) -> float:
            values = [fn(y[idx], p[idx], c[idx]) for p, c in reps]
            finite = [v for v in values if np.isfinite(v)]
            return float(np.mean(finite)) if finite else math.nan

        point = over_reps(np.arange(len(y)))
        per_rep = [fn(y, p, c) for p, c in reps]
        lo, hi = bootstrap_ci(over_reps, groups, n_boot=n_boot, seed=seed)
        report[name] = {
            "value": _round(point),
            "std_over_reps": _round(float(np.nanstd(per_rep))) if len(reps) > 1 else None,
            "ci95": [_round(lo), _round(hi)],
        }
    return report


def _round(v: float) -> float | None:
    return None if not np.isfinite(v) else round(float(v), 4)
