"""Classical measurements for the proximal femur criteria.

Input: isotropic 1 mm image in canonical orientation (pelvis on the left, see
:func:`.common.canonical_hip`).

* ROI coverage (>= 3 cm above/below, >= 2 cm lateral): field size and margins between the
  femur and the image edges. EDA: ROI violations are almost all short (cropped) fields.
* positioning / rotation: shaft axis and a coarse medial-contour profile. The
  lesser trochanter overlaps the ischium in this projection, so the contour measurement is only
  a weak cue - the learned embedding heads carry most of this criterion.
* endoprosthesis: metal saturates the detector; experts left such hips unlabeled,
  so it is reported as a finding rather than learned.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from dxa_qc.models.quality.geometry.common import FeatureVector

_SATURATION = 0.97
# Largest saturated blob that is still not metal: dense femoral heads of labeled hips saturate up
# to ~550 mm², the two metal prostheses in the dataset give 6600-7500 mm². Limitation: one more
# prosthesis in the dataset is rendered dark (metal masked by the scanner) and is not detected.
PROSTHESIS_MIN_BLOB_MM2 = 2000.0
_SHAFT_ROWS = 0.6  # rows below this fraction of the height contain only the femoral shaft
_SHAFT_SMOOTH_MM = 8.0

ROI_FEATURES = (
    "height_mm",
    "width_mm",
    "gt_top_mm",
    "pelvis_top_mm",
    "lateral_margin_mm",
    "medial_margin_mm",
)
POSITIONING_FEATURES = (
    "medial_bump_mm",
    "width_ratio_max",
    "shaft_tilt_abs_deg",
    "shaft_x_rel",
    "shaft_width_mm",
    "bone_tl",
    "bone_tr",
    "bone_bl",
    "bone_br",
)


@dataclass(frozen=True, slots=True)
class HipGeometry:
    features: FeatureVector
    shaft_tilt_deg: float
    prosthesis_suspected: bool
    largest_metal_blob_mm2: float
    shaft_line: tuple[float, float]  # canonical orientation: x = slope * y + intercept (mm)


def _largest_blob(mask: np.ndarray) -> float:
    if not mask.any():
        return 0.0
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return float(stats[1:, cv2.CC_STAT_AREA].max()) if n > 1 else 0.0


def _edges_at_half_max(row: np.ndarray, centre: int) -> tuple[int, int]:
    w = row.shape[0]
    lo, hi = max(0, centre - 9), min(w, centre + 10)
    threshold = 0.5 * row[lo:hi].max()
    left = right = centre
    while left > 0 and row[left] > threshold:
        left -= 1
    while right < w - 1 and row[right] > threshold:
        right += 1
    return left, right


def measure_hip(image: np.ndarray) -> HipGeometry:
    """``image``: isotropic 1 mm, canonical orientation, float in [0, 1]."""
    if image.ndim != 2 or min(image.shape) < 32:
        raise ValueError(f"hip image too small for measurement: {image.shape}")
    h, w = image.shape
    f: dict[str, float] = {}
    smooth = gaussian_filter(image, 1.0)
    body = image > 0

    f["height_mm"], f["width_mm"] = float(h), float(w)

    profile = gaussian_filter1d(smooth, _SHAFT_SMOOTH_MM, axis=1)
    rows = np.arange(int(h * _SHAFT_ROWS), h)
    xs = np.array([np.argmax(profile[y]) for y in rows], dtype=np.float64)
    slope, intercept = np.polyfit(rows.astype(np.float64), xs, 1)
    tilt = math.degrees(math.atan(slope))
    f["shaft_tilt_abs_deg"] = abs(tilt)
    f["shaft_x_rel"] = float(np.median(xs) / w)

    left = np.empty(h)
    right = np.empty(h)
    for y in range(h):
        centre = int(np.clip(slope * y + intercept, 0, w - 1))
        left[y], right[y] = _edges_at_half_max(smooth[y], centre)
    width = right - left
    low = int(h * 0.7)
    base = max(float(np.median(width[low:])), 1.0)
    f["shaft_width_mm"] = base
    mid = slice(int(h * 0.35), low)
    ref = np.polyval(np.polyfit(np.arange(low, h, dtype=np.float64), left[low:], 1), np.arange(h))
    f["medial_bump_mm"] = float(np.max(ref[mid] - left[mid]))
    f["width_ratio_max"] = float(np.max(width[mid]) / base)

    threshold = float(np.percentile(smooth[body], 60)) if body.any() else 0.3
    bone = smooth > threshold
    lateral_rows = np.flatnonzero(bone[:, int(w * 0.55) :].any(axis=1))
    f["gt_top_mm"] = float(lateral_rows.min()) if lateral_rows.size else float(h)
    medial_rows = np.flatnonzero(bone[:, : int(w * 0.45)].any(axis=1))
    f["pelvis_top_mm"] = float(medial_rows.min()) if medial_rows.size else float(h)
    cols = np.flatnonzero(bone.any(axis=0))
    f["lateral_margin_mm"] = float(w - 1 - cols.max()) if cols.size else 0.0
    f["medial_margin_mm"] = float(cols.min()) if cols.size else 0.0
    f["bone_tl"] = float(bone[: h // 2, : w // 2].mean())
    f["bone_tr"] = float(bone[: h // 2, w // 2 :].mean())
    f["bone_bl"] = float(bone[h // 2 :, : w // 2].mean())
    f["bone_br"] = float(bone[h // 2 :, w // 2 :].mean())

    blob = _largest_blob(image > _SATURATION)
    return HipGeometry(
        FeatureVector.from_dict(f),
        tilt,
        blob >= PROSTHESIS_MIN_BLOB_MM2,
        blob,
        (float(slope), float(intercept)),
    )
