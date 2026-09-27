"""Classical measurements for the lumbar spine criteria.

Input is the isotropic 1 mm image (see :mod:`.common`), so every length is in mm and every
angle is physical. Feature groups map one-to-one to criteria; the groups are fixed a priori by
clinical meaning, not selected on the labels (selection on ~100 studies would overfit):

* axis: tilt of the robust centreline of the vertebral column vs. the image vertical;
* coverage: bone lateral to the column at the bottom (iliac crests) and top (ribs/Th12);
* foreign objects: small bright blobs and thin bright ridges in soft tissue.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d, label, white_tophat
from skimage.filters import sato

from dxa_qc.models.quality.geometry.common import FeatureVector, robust_line_fit

# Anatomical scales, mm.
_SMOOTH_MM = 3.0
_COLUMN_HALF_WIDTH_MM = 12.0  # profile smoothing ~ half a vertebral body
_TRACK_WINDOW_MM = 25.0  # max lateral jump of the centreline between adjacent rows
_LATERAL_BAND_MM = 30.0  # |x - centre| beyond this is "lateral" (pelvis, ribs)
_SOFT_BAND_MM = 22.0  # |x - centre| beyond this is soft tissue for the artefact search
_END_ZONE_MM = 60.0  # top/bottom zone where crests / Th12 must appear
_TOPHAT_MM = 7
_TOPHAT_LEVELS = (0.15, 0.25, 0.35)
ARTIFACT_LEVEL = 0.25  # top-hat level reported as a foreign-object candidate
_RIDGE_SIGMAS = (1.0, 1.5)
_RIDGE_LEVEL = 0.08

AXIS_FEATURES = (
    "axis_tilt_abs_deg",
    "axis_tilt_top_deg",
    "axis_tilt_bottom_deg",
    "column_offset_mm",
    "centreline_resid_mm",
)
COVERAGE_FEATURES = (
    "lat_bottom_max",
    "lat_bottom_rel",
    "lat_bottom_mean",
    "crest_extent_mm",
    "lat_top_max",
    "height_mm",
    "body_height_mm",
)
FOREIGN_FEATURES = (
    "tophat_max",
    "tophat_p999",
    "tophat_n025",
    "tophat_area025",
    "ridge_max",
    "ridge_p995",
    "ridge_area",
    "soft_max",
    "soft_p999",
    "ridge_top_max",
)


@dataclass(frozen=True, slots=True)
class SpineGeometry:
    features: FeatureVector
    axis_tilt_deg: float  # signed; + means the column leans to the image right going down
    centreline_x: np.ndarray  # per row, mm; for visualization
    valid_rows: np.ndarray  # bool per row
    axis_intercept: float  # fitted axis: x = tan(tilt) * y + axis_intercept (mm)
    artifact_mask: np.ndarray  # bool; bright compact structures in soft tissue (foreign-object candidates)


def _track_centreline(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-row x of the brightest column-sized structure, tracked outward from the middle row."""
    h, w = image.shape
    profile = gaussian_filter1d(gaussian_filter(image, _SMOOTH_MM), _COLUMN_HALF_WIDTH_MM, axis=1)
    start = int(np.argmax(gaussian_filter1d(profile.mean(axis=0), 5)))
    window = int(_TRACK_WINDOW_MM)
    xs = np.zeros(h)
    strength = np.zeros(h)

    def step(y: int, ref: int) -> int:
        lo, hi = max(0, ref - window), min(w, ref + window + 1)
        seg = profile[y, lo:hi]
        x = lo + int(np.argmax(seg))
        xs[y], strength[y] = x, seg.max()
        return x

    ref = start
    for y in range(h // 2, h):
        ref = step(y, ref)
    ref = int(xs[h // 2])
    for y in range(h // 2 - 1, -1, -1):
        ref = step(y, ref)
    return xs, strength


def _tilt_deg(slope: float) -> float:
    return math.degrees(math.atan(slope))


def _lateral_profile(image: np.ndarray, xs: np.ndarray) -> np.ndarray:
    h, w = image.shape
    cols = np.arange(w)
    out = np.zeros(h)
    band = _LATERAL_BAND_MM
    for y in range(h):
        vals = image[y, np.abs(cols - xs[y]) > band]
        out[y] = np.percentile(vals, 90) if vals.size else 0.0
    return gaussian_filter1d(out, 3)


def _percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q)) if values.size else 0.0


def measure_spine(image: np.ndarray) -> SpineGeometry:
    """``image``: isotropic 1 mm, float in [0, 1], bright = dense."""
    if image.ndim != 2 or min(image.shape) < 32:
        raise ValueError(f"spine image too small for measurement: {image.shape}")
    h, w = image.shape
    f: dict[str, float] = {}

    xs, strength = _track_centreline(image)
    ys = np.arange(h, dtype=np.float64)
    valid = strength > 0.25 * strength.max() if strength.max() > 0 else np.ones(h, bool)
    if valid.sum() < 8:
        valid = np.ones(h, bool)
    slope, intercept, resid = robust_line_fit(xs[valid], ys[valid])
    tilt = _tilt_deg(slope)
    f["axis_tilt_abs_deg"] = abs(tilt)
    for name, rows in (("top", ys < h / 2), ("bottom", ys >= h / 2)):
        part = valid & rows
        f[f"axis_tilt_{name}_deg"] = (
            abs(_tilt_deg(robust_line_fit(xs[part], ys[part])[0])) if part.sum() >= 8 else 0.0
        )
    f["column_offset_mm"] = abs(float(np.median(xs[valid])) - w / 2)
    f["centreline_resid_mm"] = float(np.std(resid))

    lateral = _lateral_profile(image, xs)
    zone = min(int(_END_ZONE_MM), h // 3)
    mid = float(lateral[h // 3 : 2 * h // 3].mean())
    f["lat_bottom_max"] = float(lateral[-zone:].max())
    f["lat_bottom_rel"] = f["lat_bottom_max"] - mid
    f["lat_bottom_mean"] = float(lateral[-zone:].mean())
    present = np.flatnonzero(lateral > mid + 0.15)
    f["crest_extent_mm"] = float(h - present.min()) if present.size and present.max() >= h - zone else 0.0
    f["lat_top_max"] = float(lateral[:zone].max())
    f["height_mm"] = float(h)
    f["body_height_mm"] = float((image > 0).any(axis=1).sum())

    cols = np.arange(w)
    soft_mask = np.abs(cols[None, :] - xs[:, None]) > _SOFT_BAND_MM
    soft_mask &= image > 0
    tophat = white_tophat(image, size=(_TOPHAT_MM, _TOPHAT_MM))
    tophat = np.where(soft_mask, tophat, 0.0)
    f["tophat_max"] = float(tophat.max())
    f["tophat_p999"] = _percentile(tophat[soft_mask], 99.9)
    for level in _TOPHAT_LEVELS:
        key = f"{level:.2f}".replace("0.", "0")
        _, n = label(tophat > level)
        f[f"tophat_n{key}"] = float(n)
        f[f"tophat_area{key}"] = float((tophat > level).sum())
    ridge = np.where(soft_mask, sato(image, sigmas=_RIDGE_SIGMAS, black_ridges=False), 0.0)
    f["ridge_max"] = float(ridge.max())
    f["ridge_p995"] = _percentile(ridge[soft_mask], 99.5)
    f["ridge_area"] = float((ridge > _RIDGE_LEVEL).sum())
    f["ridge_top_max"] = float(ridge[: h // 4].max())
    soft = image[soft_mask]
    f["soft_max"] = float(soft.max()) if soft.size else 0.0
    f["soft_p999"] = _percentile(soft, 99.9)

    return SpineGeometry(FeatureVector.from_dict(f), tilt, xs, valid, intercept, tophat > ARTIFACT_LEVEL)
