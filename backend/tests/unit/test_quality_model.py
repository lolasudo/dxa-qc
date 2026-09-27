from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dxa_qc.domain.taxonomy import AnatomicalRegion, Violation
from dxa_qc.models.quality.heads import LinearHead, PlattCalibrator, logit, sigmoid
from dxa_qc.models.quality.model import (
    GEOMETRY,
    ModelInputs,
    QualityModelFormatError,
    RegionQualityModel,
    TargetEnsemble,
    quality_score,
)

RNG = np.random.default_rng(0)


def _separable(n: int = 80, d: int = 5) -> tuple[np.ndarray, np.ndarray]:
    y = np.arange(n) % 2
    x = RNG.normal(size=(n, d))
    x[:, 0] += 3 * y
    return x, y


# ---- heads --------------------------------------------------------------------------------


def test_sigmoid_is_stable_and_inverts_logit() -> None:
    assert sigmoid(np.array([-1000.0, 0.0, 1000.0])).tolist() == [0.0, 0.5, 1.0]
    p = np.array([0.01, 0.3, 0.99])
    assert np.allclose(sigmoid(logit(p)), p)


def test_linear_head_learns_signal_direction() -> None:
    x, y = _separable()
    head = LinearHead.fit(x, y, source="s", c=1.0, l2_normalize=False)
    assert head.coef[0] > 0
    z = head.decision(x)
    assert z[y == 1].mean() > z[y == 0].mean() + 2


def test_linear_head_l2_normalization_is_scale_invariant() -> None:
    x, y = _separable()
    head = LinearHead.fit(np.abs(x) + 1, y, source="emb", c=1.0, l2_normalize=True)
    assert np.allclose(head.decision(np.abs(x) + 1), head.decision((np.abs(x) + 1) * 7.5))


def test_linear_head_rejects_single_class_and_wrong_width() -> None:
    x, y = _separable()
    with pytest.raises(ValueError, match="both classes"):
        LinearHead.fit(x, np.zeros_like(y), source="s", c=1.0, l2_normalize=False)
    head = LinearHead.fit(x, y, source="s", c=1.0, l2_normalize=False)
    with pytest.raises(ValueError, match="expects 5"):
        head.decision(x[:, :3])


def test_linear_head_validates_parameters() -> None:
    with pytest.raises(ValueError):
        LinearHead("s", (), False, np.zeros(2), np.array([1.0, 0.0]), np.zeros(2), 0.0)
    with pytest.raises(ValueError):
        LinearHead("s", ("a",), False, np.zeros(2), np.ones(2), np.zeros(2), 0.0)


def test_platt_calibration_matches_frequencies() -> None:
    scores = RNG.normal(size=4000)
    y = (RNG.random(4000) < sigmoid(2 * scores - 1)).astype(int)
    cal = PlattCalibrator.fit(scores, y)
    assert cal.a == pytest.approx(2, rel=0.15)
    assert cal.b == pytest.approx(-1, abs=0.15)


def test_platt_falls_back_to_prior_without_signal() -> None:
    scores = -np.arange(40.0)
    y = (np.arange(40) >= 30).astype(int)  # positives have the LOWEST scores: anti-correlated
    cal = PlattCalibrator.fit(scores, y)
    assert np.allclose(cal(np.array([-5.0, 5.0])), 0.25, atol=1e-3)
    assert cal.a > 0
    with pytest.raises(ValueError):
        PlattCalibrator(-1.0, 0.0)


# ---- model --------------------------------------------------------------------------------


def _head(source: str, features: tuple[str, ...], coef: list[float], intercept: float = 0.0) -> LinearHead:
    d = len(coef)
    return LinearHead(source, features, False, np.zeros(d), np.ones(d), np.array(coef, float), intercept)


def _hip_model(threshold: float = 0.0, quality_threshold: float = 0.0) -> RegionQualityModel:
    pos = TargetEnsemble((_head(GEOMETRY, ("a",), [1.0]), _head("emb", (), [0.0, 1.0])), threshold=threshold)
    roi = TargetEnsemble((_head(GEOMETRY, ("b",), [1.0]),), threshold=threshold)
    overall = TargetEnsemble((_head(GEOMETRY, ("a", "b"), [0.5, 0.5]),))
    return RegionQualityModel(
        AnatomicalRegion.HIP,
        {Violation.POSITIONING: pos, Violation.HIP_ROI: roi},
        overall,
        quality_threshold=quality_threshold,
        metadata={"note": "test"},
    )


def _inputs(a: list[float], b: list[float]) -> ModelInputs:
    emb = np.column_stack([np.zeros(len(a)), np.array(a)])
    return ModelInputs({"a": np.array(a), "b": np.array(b)}, {"emb": emb})


def test_model_decision_rule() -> None:
    model = _hip_model()
    preds = model.predict(_inputs([-3.0, 3.0, -3.0], [-3.0, -3.0, 3.0]))
    assert preds[0].violations == frozenset() and preds[0].quality_class == 0
    assert preds[1].violations == {Violation.POSITIONING} and preds[1].quality_class == 1
    assert preds[2].violations == {Violation.HIP_ROI} and preds[2].quality_class == 1
    assert preds[1].criterion_probs[Violation.POSITIONING] == pytest.approx(sigmoid(3.0))


def test_quality_class_can_be_one_without_named_violation() -> None:
    """Experts mark scoliosis as bad with no criterion flag; the model may do the same."""
    model = _hip_model(threshold=5.0, quality_threshold=-1.0)
    pred = model.predict(_inputs([1.0], [1.0]))[0]
    assert pred.violations == frozenset()
    assert pred.quality_class == 1


def test_violation_always_implies_bad_quality() -> None:
    model = _hip_model(threshold=0.0, quality_threshold=100.0)
    pred = model.predict(_inputs([3.0], [-3.0]))[0]
    assert pred.violations and pred.quality_class == 1


def test_model_rejects_criteria_of_another_region() -> None:
    ens = TargetEnsemble((_head(GEOMETRY, ("a",), [1.0]),))
    with pytest.raises(ValueError, match="do not match"):
        RegionQualityModel(AnatomicalRegion.HIP, {Violation.SPINE_AXIS: ens}, ens)


def test_model_reports_missing_inputs() -> None:
    with pytest.raises(KeyError, match="emb"):
        _hip_model().predict(ModelInputs({"a": np.zeros(1), "b": np.zeros(1)}, {}))
    with pytest.raises(KeyError, match="geometry"):
        _hip_model().predict(ModelInputs({"a": np.zeros(1)}, {"emb": np.zeros((1, 2))}))


def test_quality_score_blends_overall_and_worst_criterion() -> None:
    s = quality_score(np.array([1.0]), np.array([[-2.0, 3.0]]))
    assert s[0] == pytest.approx(2.0)


def test_degenerate_calibrator_does_not_change_decisions() -> None:
    """A flat calibrator (no signal in few positives) must not flatten the violation decision."""
    from dataclasses import replace

    model = _hip_model()
    flat = PlattCalibrator(1e-6, 0.0)
    crit = {c: replace(e, calibrator=flat) for c, e in model.criteria.items()}
    flat_model = replace(model, criteria=crit, quality_calibrator=flat)
    x = _inputs([-3.0, 3.0], [-3.0, -3.0])
    assert [p.violations for p in flat_model.predict(x)] == [p.violations for p in model.predict(x)]
    assert [p.quality_class for p in flat_model.predict(x)] == [p.quality_class for p in model.predict(x)]


def test_model_save_load_roundtrip(tmp_path: Path) -> None:
    model = _hip_model(threshold=-0.3, quality_threshold=0.4)
    path = tmp_path / "hip.npz"
    model.save(path)
    loaded = RegionQualityModel.load(path)
    x = _inputs([0.2, -1.0], [0.5, 2.0])
    assert loaded.predict(x) == model.predict(x)
    assert loaded.metadata == {"note": "test"}
    assert loaded.backbones == ("emb",)
    assert loaded.geometry_features() == ("a", "b")


def test_model_load_errors(tmp_path: Path) -> None:
    with pytest.raises(QualityModelFormatError, match="not found"):
        RegionQualityModel.load(tmp_path / "missing.npz")
    bad = tmp_path / "bad.npz"
    bad.write_bytes(b"not a zip")
    with pytest.raises(QualityModelFormatError):
        RegionQualityModel.load(bad)
    wrong = tmp_path / "wrong.npz"
    np.savez(wrong, meta=np.array(json.dumps({"format_version": 999})))
    with pytest.raises(QualityModelFormatError, match="format"):
        RegionQualityModel.load(wrong)


def test_model_file_is_not_pickle(tmp_path: Path) -> None:
    path = tmp_path / "hip.npz"
    _hip_model().save(path)
    with np.load(path, allow_pickle=False) as data:  # would raise if any object array was stored
        assert "meta" in data.files
