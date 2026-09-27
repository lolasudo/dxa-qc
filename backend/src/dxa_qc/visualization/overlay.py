"""Preview and violation overlay images.

Both are rendered on the isotropic grid (true physical proportions) at ``SCALE`` px per mm and
have identical sizes, so the frontend can stack the transparent overlay on the preview.
What is drawn comes from the same measurements the model used - no separate heuristics:

* spine: tracked centreline, fitted axis (red if the axis criterion is violated), vertical
  reference, and foreign-object candidates (red if that criterion is violated, amber otherwise);
* hip: femoral shaft axis (red if positioning is violated), field border (red if the ROI
  criterion is violated), saturated metal (prosthesis) outline.
"""

from __future__ import annotations

from itertools import pairwise

import cv2
import numpy as np

from dxa_qc.dicom_io.reader import DicomImage
from dxa_qc.domain.taxonomy import AnatomicalRegion, HipSide, Violation
from dxa_qc.models.quality.geometry import canonical_hip, measure_hip, measure_spine, to_isotropic
from dxa_qc.models.region.classifier import RegionPrediction
from dxa_qc.pipeline.assessment import QualityAssessment

SCALE = 2  # output pixels per mm

# RGBA
OK = (46, 204, 113, 235)
BAD = (239, 68, 68, 245)
WARN = (245, 158, 11, 220)
GUIDE = (56, 189, 248, 200)
REFERENCE = (255, 255, 255, 150)
METAL = (217, 70, 239, 235)


def _iso(image: DicomImage) -> np.ndarray:
    return to_isotropic(image.pixels, image.spacing)


def render_preview(image: DicomImage) -> np.ndarray:
    """8-bit grayscale preview, physical proportions."""
    iso = _iso(image)
    h, w = iso.shape
    up = cv2.resize(iso, (w * SCALE, h * SCALE), interpolation=cv2.INTER_CUBIC)
    return (np.clip(up, 0, 1) * 255).round().astype(np.uint8)


def _line(
    canvas: np.ndarray, p0: tuple[float, float], p1: tuple[float, float], color, thickness: int = 2
) -> None:
    pts = [(round(x * SCALE), round(y * SCALE)) for x, y in (p0, p1)]
    cv2.line(canvas, pts[0], pts[1], color, thickness, lineType=cv2.LINE_AA)


def _dashed_vertical(canvas: np.ndarray, x: float, y0: float, y1: float, color) -> None:
    step = 6
    for y in np.arange(y0, y1, 2 * step):
        _line(canvas, (x, y), (x, min(y + step, y1)), color, 1)


def _contours(canvas: np.ndarray, mask: np.ndarray, color, thickness: int = 2) -> None:
    if not mask.any():
        return
    up = cv2.resize(
        mask.astype(np.uint8), (mask.shape[1] * SCALE, mask.shape[0] * SCALE), interpolation=cv2.INTER_NEAREST
    )
    # Dilate so single-pixel artefacts get a visible ring around them instead of being covered.
    up = cv2.dilate(up, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(up, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, color, thickness, lineType=cv2.LINE_AA)


def _spine_overlay(iso: np.ndarray, quality: QualityAssessment | None) -> np.ndarray:
    h, w = iso.shape
    canvas = np.zeros((h * SCALE, w * SCALE, 4), np.uint8)
    geo = measure_spine(iso)
    violated = quality.violations if quality else frozenset()

    rows = np.flatnonzero(geo.valid_rows)
    for y0, y1 in pairwise(rows):
        if y1 == y0 + 1:
            _line(canvas, (geo.centreline_x[y0], y0), (geo.centreline_x[y1], y1), GUIDE, 1)
    slope = np.tan(np.radians(geo.axis_tilt_deg))
    top, bottom = float(rows.min()), float(rows.max())
    x_top, x_bottom = slope * top + geo.axis_intercept, slope * bottom + geo.axis_intercept
    _dashed_vertical(canvas, (x_top + x_bottom) / 2, top, bottom, REFERENCE)
    _line(canvas, (x_top, top), (x_bottom, bottom), BAD if Violation.SPINE_AXIS in violated else OK, 3)
    _contours(canvas, geo.artifact_mask, BAD if Violation.FOREIGN_OBJECTS in violated else WARN)
    if Violation.POSITIONING in violated:
        # Coverage criterion concerns the scan limits: highlight top and bottom edges.
        cv2.rectangle(canvas, (0, 0), (w * SCALE - 1, 3), BAD, -1)
        cv2.rectangle(canvas, (0, h * SCALE - 4), (w * SCALE - 1, h * SCALE - 1), BAD, -1)
    return canvas


def _hip_overlay(iso: np.ndarray, side: HipSide, quality: QualityAssessment | None) -> np.ndarray:
    canonical = canonical_hip(iso, side)
    h, w = canonical.shape
    canvas = np.zeros((h * SCALE, w * SCALE, 4), np.uint8)
    geo = measure_hip(canonical)
    violated = quality.violations if quality else frozenset()

    slope, intercept = geo.shaft_line
    y0 = h * 0.35
    _line(
        canvas,
        (slope * y0 + intercept, y0),
        (slope * (h - 1) + intercept, h - 1),
        BAD if Violation.POSITIONING in violated else OK,
        3,
    )
    if Violation.HIP_ROI in violated:
        cv2.rectangle(canvas, (1, 1), (w * SCALE - 2, h * SCALE - 2), BAD, 4)
    if geo.prosthesis_suspected:
        _contours(canvas, canonical > 0.97, METAL)
    # Back to the original orientation of the image.
    return np.ascontiguousarray(canvas[:, ::-1]) if side is HipSide.RIGHT else canvas


def render_overlay(
    image: DicomImage, region: RegionPrediction, quality: QualityAssessment | None
) -> np.ndarray:
    """RGBA overlay with the same size as :func:`render_preview`."""
    iso = _iso(image)
    if region.region is AnatomicalRegion.SPINE:
        return _spine_overlay(iso, quality)
    if region.side is None:
        raise ValueError("hip overlay needs the side")
    return _hip_overlay(iso, region.side, quality)


def encode_png(image: np.ndarray) -> bytes:
    """Grayscale (H, W) or RGBA (H, W, 4) -> PNG bytes."""
    if image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("PNG encoding failed")
    return buf.tobytes()
