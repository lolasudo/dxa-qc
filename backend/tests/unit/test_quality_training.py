from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dxa_qc.domain.taxonomy import AnatomicalRegion, Violation
from dxa_qc.models.quality.embeddings import BackboneSpec
from dxa_qc.models.quality.model import OVERALL, ModelInputs, RegionQualityModel
from dxa_qc.training.quality import (
    Design,
    RegionData,
    _folds,
    apply_decision,
    crossfit_predictions,
    evaluate,
    fit_decision,
    fit_final,
    oof_scores,
)

FAKE = BackboneSpec("fake_emb", "fake", 32)
DESIGN = Design((FAKE,), {Violation.POSITIONING: ("g_pos", "g_noise"), Violation.HIP_ROI: ("g_roi",)})


def make_data(n_studies: int = 60, seed: int = 0) -> RegionData:
    """Two hips per study; ROI is detectable from geometry, positioning from the embedding."""
    rng = np.random.default_rng(seed)
    n = 2 * n_studies
    groups = np.repeat([f"s{i}" for i in range(n_studies)], 2)
    pos = (rng.random(n) < 0.3).astype(int)
    roi = (rng.random(n) < 0.2).astype(int)
    overall = pos | roi
    geometry = {
        "g_pos": rng.normal(size=n),  # uninformative geometry for positioning
        "g_noise": rng.normal(size=n),
        "g_roi": rng.normal(size=n) + 2.5 * roi,
    }
    emb = rng.normal(size=(n, 16))
    emb[:, 0] += 2.5 * pos
    emb += 5  # positive offset like CNN features, exercises L2 normalization
    return RegionData(
        AnatomicalRegion.HIP,
        groups,
        ModelInputs(geometry, {FAKE.key: emb}),
        {Violation.POSITIONING.name: pos, Violation.HIP_ROI.name: roi, OVERALL: overall},
    )


def test_folds_never_split_a_study() -> None:
    data = make_data()
    folds = _folds(data, 5, seed=1)
    assert sorted(np.concatenate(folds).tolist()) == list(range(data.n))
    for test in folds:
        train = np.setdiff1d(np.arange(data.n), test)
        assert not set(data.groups[test]) & set(data.groups[train])


def test_oof_scores_carry_signal() -> None:
    from sklearn.metrics import roc_auc_score

    data = make_data()
    scores = oof_scores(data, DESIGN, _folds(data, 5, seed=0))
    assert set(scores) == {"POSITIONING", "HIP_ROI", OVERALL}
    assert roc_auc_score(data.labels["HIP_ROI"], scores["HIP_ROI"]) > 0.85
    assert roc_auc_score(data.labels["POSITIONING"], scores["POSITIONING"]) > 0.8


def test_noise_labels_give_chance_auc() -> None:
    """Guards against leakage: labels independent of inputs must not be 'predicted'."""
    from sklearn.metrics import roc_auc_score

    data = make_data(seed=3)
    rng = np.random.default_rng(42)
    noise = {k: (rng.random(data.n) < 0.3).astype(int) for k in data.labels}
    shuffled = RegionData(data.region, data.groups, data.inputs, noise)
    aucs = []
    for rep in range(3):
        scores = oof_scores(shuffled, DESIGN, _folds(shuffled, 5, seed=rep))
        aucs.append(roc_auc_score(noise["POSITIONING"], scores["POSITIONING"]))
    assert abs(np.mean(aucs) - 0.5) < 0.12


def test_decision_is_consistent_with_report_contract() -> None:
    data = make_data()
    folds = _folds(data, 5, seed=0)
    scores = oof_scores(data, DESIGN, folds)
    crossfit = crossfit_predictions(data, scores, folds)
    flagged = crossfit["v_POSITIONING"].astype(bool) | crossfit["v_HIP_ROI"].astype(bool)
    assert np.all(crossfit["quality_class"][flagged] == 1), "a violation must imply quality_class=1"
    assert np.all((crossfit["quality_prob"] >= 0) & (crossfit["quality_prob"] <= 1))


def test_fit_decision_thresholds_in_range() -> None:
    data = make_data()
    scores = oof_scores(data, DESIGN, _folds(data, 5, seed=0))
    d = fit_decision(data.region, scores, data.labels)
    assert all(np.isfinite(t) for t in d.thresholds.values())
    assert np.isfinite(d.quality_threshold)
    out = apply_decision(data.region, scores, d)
    assert out["quality_class"].shape == (data.n,)


def test_evaluate_report_structure() -> None:
    data = make_data()
    report, mean_scores, crossfit = evaluate(data, DESIGN, reps=2, n_boot=50)
    assert len(crossfit) == 2
    assert set(report) == {
        "quality_class",
        "POSITIONING",
        "HIP_ROI",
        "violation_macro_f1",
        "quality_prob_brier",
    }
    brier = report["quality_prob_brier"]  # type: ignore[index]
    assert brier["value"] < brier["prevalence_baseline"], "informative model must beat the prior"
    auc = report["HIP_ROI"]["roc_auc"]  # type: ignore[index]
    assert auc["value"] > 0.85
    assert auc["ci95"][0] <= auc["value"] <= auc["ci95"][1]
    assert set(mean_scores) == {"POSITIONING", "HIP_ROI", OVERALL}


def test_fit_final_produces_loadable_model(tmp_path: Path) -> None:
    data = make_data()
    _, mean_scores, _ = evaluate(data, DESIGN, reps=1, n_boot=10)
    model = fit_final(data, DESIGN, mean_scores, {"source": "test"})
    path = tmp_path / "hip.npz"
    model.save(path)
    loaded = RegionQualityModel.load(path)
    preds = loaded.predict(data.inputs)
    assert len(preds) == data.n
    flagged = np.array([Violation.HIP_ROI in p.violations for p in preds])
    assert flagged[data.labels["HIP_ROI"] == 1].mean() > 0.6
    assert loaded.backbones == (FAKE.key,)


def test_design_overall_uses_union_of_groups() -> None:
    assert DESIGN.geometry_for(OVERALL) == ("g_pos", "g_noise", "g_roi")
    assert DESIGN.geometry_for("HIP_ROI") == ("g_roi",)
    with pytest.raises(KeyError):
        DESIGN.geometry_for("SPINE_AXIS")


# ---- C selection, embedding cache, region resolution ---------------------------------------------


def test_select_c_single_candidate_skips_fitting() -> None:
    from dxa_qc.training.quality import select_c

    empty = np.zeros((0, 1))  # would fail if anything were fitted
    assert select_c(empty, np.zeros(0), np.zeros(0), (0.3,), l2_normalize=False, n_splits=3, seed=0) == 0.3


def test_select_c_picks_best_inner_auc_and_breaks_ties_to_smaller_c(monkeypatch: pytest.MonkeyPatch) -> None:
    from dxa_qc.training import quality

    class Head:  # ranks perfectly for C=1, randomly for C=100, inverted for C=0.01
        def __init__(self, c: float) -> None:
            self.c = c

        def decision(self, x: np.ndarray) -> np.ndarray:
            return {1.0: x[:, 0], 100.0: x[:, 1], 0.01: -x[:, 0]}[self.c]

    fitted: list[float] = []

    def fake_fit(x, y, *, c, **_):
        fitted.append(c)
        return Head(c)

    monkeypatch.setattr(quality.LinearHead, "fit", staticmethod(fake_fit))
    data = make_data()
    y = data.labels["HIP_ROI"]
    x = np.column_stack(
        [y + 0.1 * np.random.default_rng(0).random(data.n), np.random.default_rng(1).random(data.n)]
    )
    kw = dict(l2_normalize=False, n_splits=3, seed=0)
    assert quality.select_c(x, y, data.groups, (100.0, 0.01, 1.0), **kw) == 1.0
    assert set(fitted) == {0.01, 1.0, 100.0} and len(fitted) == 9  # 3 candidates x 3 inner folds
    # Identical inner AUC for two candidates: the stronger regularization (smaller C) wins.
    Head.decision = lambda self, x: x[:, 0]  # type: ignore[method-assign]
    assert quality.select_c(x, y, data.groups, (1.0, 100.0), **kw) == 1.0


def test_select_c_ties_and_degenerate_labels_go_to_smallest_c() -> None:
    from dxa_qc.training.quality import select_c

    data = make_data()
    x = data.inputs.geometry["g_roi"][:, None]
    one_positive = np.zeros(data.n, int)
    one_positive[0] = 1  # fewer positives than folds: no inner CV possible
    assert select_c(x, one_positive, data.groups, (1.0, 0.1), l2_normalize=False, n_splits=3, seed=0) == 0.1


def test_select_c_uses_only_given_rows() -> None:
    """Deterministic and a function of its inputs only: the outer test fold never enters it."""
    from dxa_qc.training.quality import select_c

    data = make_data()
    train = np.arange(data.n) < 80
    x = data.inputs.embeddings[FAKE.key]
    args = dict(l2_normalize=True, n_splits=3, seed=1)
    a = select_c(x[train], data.labels["POSITIONING"][train], data.groups[train], (0.001, 0.01, 1.0), **args)
    b = select_c(x[train], data.labels["POSITIONING"][train], data.groups[train], (0.001, 0.01, 1.0), **args)
    assert a == b


def test_embedding_cache_round_trip_and_key_separation(tmp_path: Path) -> None:
    from dxa_qc.models.quality.embeddings import Crop
    from dxa_qc.training.quality import EmbeddingCache

    cache = EmbeddingCache(tmp_path)
    vec = np.arange(4, dtype=np.float32)
    assert cache.get(FAKE, "d1") is None
    cache.put(FAKE, "d1", vec)
    assert np.array_equal(cache.get(FAKE, "d1"), vec)
    cropped = BackboneSpec(FAKE.key, FAKE.timm_name, FAKE.input_size, Crop(0.1, 0.9))
    assert cache.get(cropped, "d1") is None, "any spec change must miss the cache"
    assert not list(tmp_path.rglob("*.tmp.npy")), "writes are atomic (temp file renamed)"


def test_embedding_cache_disabled_and_corrupted(tmp_path: Path) -> None:
    from dxa_qc.training.quality import EmbeddingCache

    off = EmbeddingCache(None)
    off.put(FAKE, "d", np.ones(2))
    assert off.get(FAKE, "d") is None
    cache = EmbeddingCache(tmp_path)
    cache.put(FAKE, "d", np.ones(2))
    (entry,) = tmp_path.rglob("d.npy")
    entry.write_bytes(b"not numpy")
    assert cache.get(FAKE, "d") is None  # recomputed, never crashes training


def test_embedding_cache_refuses_pickled_arrays(tmp_path: Path) -> None:
    from dxa_qc.training.quality import EmbeddingCache

    cache = EmbeddingCache(tmp_path)
    cache.put(FAKE, "d", np.ones(2))
    (entry,) = tmp_path.rglob("d.npy")
    np.save(entry, np.array([{"evil": 1}], dtype=object), allow_pickle=True)
    assert cache.get(FAKE, "d") is None


class _FakeRegionModel:
    def __init__(self, cls, probs=None) -> None:
        from dxa_qc.models.region.classifier import RegionPrediction

        self.calls = 0
        self.pred = RegionPrediction(cls, 0.9, probs or {cls: 0.9})

    def predict(self, pixels):
        self.calls += 1
        return self.pred


def _row(region=None, side=None):
    from dxa_qc.data.manifest import ManifestRow

    return ManifestRow(7, Path("x.dcm"), "s1", region, side, None, frozenset())


def test_resolve_region_trusts_explicit_manifest_values() -> None:
    from dxa_qc.domain.taxonomy import HipSide
    from dxa_qc.models.region.classifier import RegionClass
    from dxa_qc.training.quality import resolve_region

    model = _FakeRegionModel(RegionClass.HIP_LEFT)
    px = np.zeros((4, 4))
    assert resolve_region(_row(AnatomicalRegion.SPINE), px, model) == (AnatomicalRegion.SPINE, None)
    hip = resolve_region(_row(AnatomicalRegion.HIP, HipSide.RIGHT), px, model)
    assert hip == (AnatomicalRegion.HIP, HipSide.RIGHT)
    assert model.calls == 0
    assert resolve_region(_row(AnatomicalRegion.SPINE), px, None)[0] is AnatomicalRegion.SPINE


def test_resolve_region_asks_classifier_for_missing_values() -> None:
    from dxa_qc.domain.taxonomy import HipSide
    from dxa_qc.models.region.classifier import RegionClass
    from dxa_qc.training.quality import resolve_region

    px = np.zeros((4, 4))
    assert resolve_region(_row(), px, _FakeRegionModel(RegionClass.HIP_RIGHT)) == (
        AnatomicalRegion.HIP,
        HipSide.RIGHT,
    )
    # Region given as hip but the classifier's top class is spine: side still comes from hip classes.
    probs = {RegionClass.SPINE: 0.6, RegionClass.HIP_LEFT: 0.1, RegionClass.HIP_RIGHT: 0.3}
    model = _FakeRegionModel(RegionClass.SPINE, probs)
    assert resolve_region(_row(AnatomicalRegion.HIP), px, model) == (AnatomicalRegion.HIP, HipSide.RIGHT)
    with pytest.raises(ValueError, match="line 7"):
        resolve_region(_row(), px, None)
