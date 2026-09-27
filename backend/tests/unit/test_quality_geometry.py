from __future__ import annotations

import math

import numpy as np
import pytest

from dxa_qc.dicom_io.reader import PixelSpacing, SpacingSource
from dxa_qc.domain.taxonomy import HipSide
from dxa_qc.models.quality.geometry import (
    FeatureVector,
    canonical_hip,
    measure_hip,
    measure_spine,
    robust_line_fit,
    to_isotropic,
)
from dxa_qc.models.quality.geometry.hip import POSITIONING_FEATURES, PROSTHESIS_MIN_BLOB_MM2, ROI_FEATURES
from dxa_qc.models.quality.geometry.spine import AXIS_FEATURES, COVERAGE_FEATURES, FOREIGN_FEATURES


def synthetic_spine(tilt_deg: float = 0.0, h: int = 330, w: int = 180, crests: bool = True) -> np.ndarray:
    """Isotropic phantom: a 40 mm bright column tilted by ``tilt_deg``, optional iliac crests."""
    img = np.full((h, w), 0.2, np.float32)
    ys, xs = np.mgrid[0:h, 0:w]
    centre = w / 2 + math.tan(math.radians(tilt_deg)) * (ys - h / 2)
    img[np.abs(xs - centre) < 20] = 0.8
    if crests:
        img[h - 45 :, :35] = 0.7
        img[h - 45 :, w - 35 :] = 0.7
    return img


# ---- common -------------------------------------------------------------------------------


def test_to_isotropic_uses_physical_size() -> None:
    spacing = PixelSpacing(1.05, 0.6, SpacingSource.FALLBACK)
    out = to_isotropic(np.zeros((300, 300), np.float32), spacing)
    assert out.shape == (315, 180)


def test_to_isotropic_rejects_non_2d() -> None:
    with pytest.raises(ValueError):
        to_isotropic(np.zeros((3, 3, 3)), PixelSpacing(1, 1, SpacingSource.FALLBACK))


def test_canonical_hip_mirrors_only_right_side() -> None:
    img = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert np.array_equal(canonical_hip(img, HipSide.LEFT), img)
    assert np.array_equal(canonical_hip(img, HipSide.RIGHT), img[:, ::-1])


def test_robust_line_fit_ignores_outliers() -> None:
    y = np.arange(100, dtype=np.float64)
    x = 0.1 * y + 5
    x[::10] += 40  # 10% gross outliers
    slope, intercept, _ = robust_line_fit(x, y)
    assert slope == pytest.approx(0.1, abs=1e-3)
    assert intercept == pytest.approx(5, abs=0.1)


def test_robust_line_fit_needs_two_points() -> None:
    with pytest.raises(ValueError):
        robust_line_fit(np.array([1.0]), np.array([1.0]))


def test_feature_vector_rejects_nan_and_selects_by_name() -> None:
    fv = FeatureVector.from_dict({"a": 1.0, "b": 2.0})
    assert fv.select(("b", "a")).tolist() == [2.0, 1.0]
    with pytest.raises(KeyError):
        fv.select(("c",))
    with pytest.raises(ValueError, match="non-finite"):
        FeatureVector.from_dict({"a": float("nan")})


# ---- spine --------------------------------------------------------------------------------


@pytest.mark.parametrize("tilt", [0.0, 3.0, -6.9, 10.0])
def test_spine_axis_tilt_is_measured_in_degrees(tilt: float) -> None:
    geo = measure_spine(synthetic_spine(tilt))
    assert geo.axis_tilt_deg == pytest.approx(tilt, abs=0.5)
    assert geo.features.as_dict()["axis_tilt_abs_deg"] == pytest.approx(abs(tilt), abs=0.5)


def test_spine_tilt_is_physical_after_resampling() -> None:
    """A 5° physical tilt drawn on the anisotropic scanner grid must still read as ~5°."""
    spacing = PixelSpacing(1.05, 0.6, SpacingSource.FALLBACK)
    physical = synthetic_spine(5.0, h=336, w=180)
    import cv2

    raw = cv2.resize(physical, (300, 320), interpolation=cv2.INTER_AREA)  # 180/0.6 x 336/1.05
    geo = measure_spine(to_isotropic(raw, spacing))
    assert geo.axis_tilt_deg == pytest.approx(5.0, abs=0.6)


def test_spine_crests_raise_coverage_features() -> None:
    with_crests = measure_spine(synthetic_spine(crests=True)).features.as_dict()
    without = measure_spine(synthetic_spine(crests=False)).features.as_dict()
    assert with_crests["lat_bottom_max"] > without["lat_bottom_max"] + 0.3
    assert with_crests["crest_extent_mm"] > 30
    assert without["crest_extent_mm"] == 0


def test_spine_metal_dot_raises_foreign_object_features() -> None:
    clean = synthetic_spine()
    dirty = clean.copy()
    dirty[60:64, 20:24] = 1.0  # 4 mm metal clip in soft tissue
    f_clean = measure_spine(clean).features.as_dict()
    f_dirty = measure_spine(dirty).features.as_dict()
    assert f_dirty["tophat_max"] > f_clean["tophat_max"] + 0.4
    assert f_dirty["tophat_n025"] >= 1 > f_clean["tophat_n025"]


def test_spine_feature_groups_are_all_computed() -> None:
    fv = measure_spine(synthetic_spine()).features
    fv.select(AXIS_FEATURES + COVERAGE_FEATURES + FOREIGN_FEATURES)


def test_spine_rejects_tiny_image() -> None:
    with pytest.raises(ValueError):
        measure_spine(np.zeros((10, 10), np.float32))


def test_spine_survives_blank_image() -> None:
    fv = measure_spine(np.zeros((200, 150), np.float32)).features
    assert np.isfinite(fv.values).all()


# ---- hip ----------------------------------------------------------------------------------


def synthetic_hip(h: int = 330, w: int = 190) -> np.ndarray:
    """Canonical phantom: vertical shaft on the right half, pelvis blob top-left."""
    img = np.full((h, w), 0.15, np.float32)
    img[h // 3 :, 110:140] = 0.8  # shaft
    img[: h // 3, 90:160] = 0.7  # head/neck/greater trochanter
    img[: h // 4, :70] = 0.6  # pelvis
    return img


def test_hip_roi_features_follow_field_size() -> None:
    full = measure_hip(synthetic_hip()).features.as_dict()
    cropped = measure_hip(synthetic_hip()[:200]).features.as_dict()
    assert full["height_mm"] == 330
    assert cropped["height_mm"] == 200


def test_hip_vertical_shaft_has_no_tilt() -> None:
    geo = measure_hip(synthetic_hip())
    assert geo.shaft_tilt_deg == pytest.approx(0.0, abs=1.0)
    geo.features.select(ROI_FEATURES + POSITIONING_FEATURES)


def test_hip_prosthesis_detection_by_saturated_blob() -> None:
    normal = synthetic_hip()
    normal[60:70, 120:130] = 1.0  # small saturated cortex patch, 100 mm²
    assert not measure_hip(normal).prosthesis_suspected
    metal = synthetic_hip()
    side = math.ceil(math.sqrt(PROSTHESIS_MIN_BLOB_MM2)) + 2
    metal[20 : 20 + side, 100 : 100 + side] = 1.0
    geo = measure_hip(metal)
    assert geo.prosthesis_suspected
    assert geo.largest_metal_blob_mm2 >= PROSTHESIS_MIN_BLOB_MM2
