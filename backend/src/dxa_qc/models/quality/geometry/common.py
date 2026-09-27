"""Shared image geometry helpers.

All quality features are computed on an isotropic 1 mm grid: the scanner's pixels are
anisotropic (1.05 mm rows x 0.6 mm columns), so angles and distances measured on the raw grid
would be skewed (a 5° tilt on the raw grid is ~8.7° physically).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from dxa_qc.dicom_io.reader import PixelSpacing
from dxa_qc.domain.taxonomy import HipSide

MM_PER_PX = 1.0  # resolution of the isotropic grid
# Defence in depth next to the reader's physical-size check: never allocate a huge grid.
MAX_ISO_SIDE = 4096


@dataclass(frozen=True, slots=True)
class FeatureVector:
    """Named scalar features; order is part of the model contract (stored with the weights)."""

    names: tuple[str, ...]
    values: np.ndarray

    def __post_init__(self) -> None:
        if self.values.shape != (len(self.names),):
            raise ValueError(f"{len(self.names)} names but values of shape {self.values.shape}")
        if not np.isfinite(self.values).all():
            bad = [n for n, v in zip(self.names, self.values, strict=True) if not np.isfinite(v)]
            raise ValueError(f"non-finite features: {bad}")

    @classmethod
    def from_dict(cls, features: dict[str, float]) -> FeatureVector:
        return cls(tuple(features), np.array([float(v) for v in features.values()], dtype=np.float64))

    def as_dict(self) -> dict[str, float]:
        return {n: float(v) for n, v in zip(self.names, self.values, strict=True)}

    def select(self, names: tuple[str, ...]) -> np.ndarray:
        index = {n: i for i, n in enumerate(self.names)}
        missing = [n for n in names if n not in index]
        if missing:
            raise KeyError(f"features not computed: {missing}")
        return self.values[[index[n] for n in names]]


def to_isotropic(pixels: np.ndarray, spacing: PixelSpacing) -> np.ndarray:
    """Resample a 2-D image in [0, 1] to a 1 mm x 1 mm grid (area interpolation, no aliasing)."""
    if pixels.ndim != 2:
        raise ValueError(f"expected a 2-D image, got shape {pixels.shape}")
    h, w = pixels.shape
    new_h = max(1, round(h * spacing.row_mm / MM_PER_PX))
    new_w = max(1, round(w * spacing.column_mm / MM_PER_PX))
    if new_h > MAX_ISO_SIDE or new_w > MAX_ISO_SIDE:
        raise ValueError(f"isotropic grid {new_h}x{new_w} exceeds {MAX_ISO_SIDE} px per side")
    image = np.ascontiguousarray(pixels, dtype=np.float32)
    interpolation = cv2.INTER_AREA if new_h * new_w <= h * w else cv2.INTER_LINEAR
    return cv2.resize(image, (new_w, new_h), interpolation=interpolation)


def canonical_hip(image: np.ndarray, side: HipSide) -> np.ndarray:
    """Mirror so that the pelvis is always on the left of the image.

    Radiological convention puts the pelvis of a right hip on the image right; one canonical
    orientation lets both sides share one model (doubling the training data per criterion).
    """
    return np.ascontiguousarray(image[:, ::-1]) if side is HipSide.RIGHT else image


def robust_line_fit(x: np.ndarray, y: np.ndarray, iterations: int = 10) -> tuple[float, float, np.ndarray]:
    """Fit ``x = slope * y + intercept`` with Tukey-biweight IRLS; returns (slope, intercept, residuals).

    The spine centreline has outliers (sacrum, bright artefacts next to the column); ordinary
    least squares would let a few of them tilt the axis.
    """
    if x.shape != y.shape or x.size < 2:
        raise ValueError("need at least two points of equal-length x and y")
    weights = np.ones_like(x, dtype=np.float64)
    slope = intercept = 0.0
    for _ in range(iterations):
        sw = np.sqrt(weights)
        design = np.column_stack([y, np.ones_like(y)]) * sw[:, None]
        (slope, intercept), *_ = np.linalg.lstsq(design, x * sw, rcond=None)
        residuals = x - (slope * y + intercept)
        scale = 1.4826 * np.median(np.abs(residuals)) + 1e-6
        u = residuals / (4.685 * scale)
        weights = np.where(np.abs(u) < 1, (1 - u**2) ** 2, 0.0)
        if weights.sum() < 2:  # degenerate: everything rejected, keep the last valid fit
            break
    return float(slope), float(intercept), x - (slope * y + intercept)
