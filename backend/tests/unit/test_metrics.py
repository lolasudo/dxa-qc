from __future__ import annotations

import math

import numpy as np
import pytest

from dxa_qc.training.metrics import (
    balanced_accuracy,
    best_f1_threshold,
    binary_report,
    bootstrap_ci,
    f1,
    macro_f1,
    roc_auc,
    sensitivity,
    specificity,
)

Y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
PRED = np.array([1, 1, 0, 1, 0, 0, 0, 0])
PROB = np.array([0.9, 0.8, 0.3, 0.7, 0.2, 0.1, 0.1, 0.05])


def test_confusion_metrics() -> None:
    assert sensitivity(Y, PROB, PRED) == pytest.approx(2 / 3)
    assert specificity(Y, PROB, PRED) == pytest.approx(4 / 5)
    assert balanced_accuracy(Y, PROB, PRED) == pytest.approx((2 / 3 + 4 / 5) / 2)
    assert f1(Y, PROB, PRED) == pytest.approx(2 / 3)


def test_auc_undefined_for_single_class() -> None:
    assert math.isnan(roc_auc(np.zeros(4, int), np.arange(4.0), np.zeros(4, int)))
    assert math.isnan(sensitivity(np.zeros(4, int), np.arange(4.0), np.zeros(4, int)))


def test_best_f1_threshold_separates_perfectly() -> None:
    y = np.array([0, 0, 0, 1, 1])
    prob = np.array([0.1, 0.2, 0.4, 0.6, 0.9])
    t = best_f1_threshold(y, prob)
    assert 0.4 < t <= 0.6
    assert f1(y, prob, (prob >= t).astype(int)) == 1.0


def test_best_f1_threshold_without_positives_predicts_nothing() -> None:
    assert best_f1_threshold(np.zeros(3, int), np.array([0.1, 0.5, 0.9])) == 1.0


def test_macro_f1_averages_columns() -> None:
    y = np.array([[1, 0], [0, 1], [1, 0]])
    pred = np.array([[1, 0], [0, 0], [1, 0]])
    assert macro_f1(y, pred) == pytest.approx((1.0 + 0.0) / 2)


def test_bootstrap_ci_resamples_whole_groups() -> None:
    groups = ["a", "a", "b", "b", "c", "c"]
    seen_sizes = set()

    def metric(idx: np.ndarray) -> float:
        # every group contributes both of its rows or none
        counts = np.bincount(idx, minlength=6)
        assert np.all(counts[0::2] == counts[1::2])
        seen_sizes.add(len(idx))
        return float(len(idx))

    lo, hi = bootstrap_ci(metric, groups, n_boot=200, seed=1)
    assert lo == hi == 6.0
    assert seen_sizes == {6}


def test_bootstrap_ci_contains_point_estimate() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    prob = np.clip(y * 0.3 + rng.random(300) * 0.7, 0, 1)
    groups = [str(i // 2) for i in range(300)]
    lo, hi = bootstrap_ci(lambda idx: roc_auc(y[idx], prob[idx], y[idx]), groups, n_boot=300)
    assert lo < roc_auc(y, prob, y) < hi


def test_bootstrap_ci_nan_when_metric_mostly_undefined() -> None:
    assert all(math.isnan(v) for v in bootstrap_ci(lambda idx: math.nan, ["a", "b"], n_boot=10))


def test_binary_report_is_json_ready() -> None:
    import json

    report = binary_report(Y, PROB, PRED, [str(i) for i in range(len(Y))], n_boot=100)
    assert report["n"] == 8 and report["n_positive"] == 3
    assert set(report) >= {"sensitivity", "specificity", "balanced_accuracy", "f1", "roc_auc", "pr_auc"}
    json.dumps(report)


def test_numpy_metrics_match_sklearn() -> None:
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

    from dxa_qc.training.metrics import pr_auc

    rng = np.random.default_rng(3)
    for _ in range(20):
        y = rng.integers(0, 2, 60)
        prob = np.round(rng.random(60), 1)  # coarse values -> many ties
        pred = (prob >= 0.5).astype(int)
        assert roc_auc(y, prob, pred) == pytest.approx(roc_auc_score(y, prob))
        assert pr_auc(y, prob, pred) == pytest.approx(average_precision_score(y, prob))
        assert f1(y, prob, pred) == pytest.approx(f1_score(y, pred, zero_division=0))


def test_multi_rep_report_point_inside_ci() -> None:
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 120)
    probs = [np.clip(y * 0.3 + rng.random(120) * 0.7, 0, 1) for _ in range(4)]
    preds = [(p >= 0.5).astype(int) for p in probs]
    report = binary_report(y, probs, preds, [str(i // 2) for i in range(120)], n_boot=300)
    assert report["cv_repetitions"] == 4
    for name in ("roc_auc", "f1", "sensitivity"):
        block = report[name]
        assert block["ci95"][0] <= block["value"] <= block["ci95"][1]
        assert block["std_over_reps"] is not None


def test_best_score_threshold_on_unbounded_scores() -> None:
    from dxa_qc.training.metrics import best_score_threshold

    y = np.array([0, 0, 1, 1])
    score = np.array([-10.0, -3.0, 4.0, 12.0])
    t = best_score_threshold(y, score)
    assert -3.0 < t < 4.0
    assert best_score_threshold(np.zeros(4, int), score) > 12.0  # no positives: flag nothing
    with pytest.raises(ValueError):
        best_score_threshold(y, np.array([0.0, np.inf, 1.0, 2.0]))
