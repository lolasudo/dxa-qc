from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from dxa_qc.dicom_io.reader import DicomImage, PixelSpacing, SpacingSource
from dxa_qc.domain.taxonomy import Violation
from dxa_qc.models.region.classifier import RegionClass, RegionPrediction
from dxa_qc.pipeline.assessment import QualityAssessment
from dxa_qc.visualization import SCALE, encode_png, render_overlay, render_preview
from dxa_qc.visualization.overlay import BAD, OK
from tests.unit.test_quality_geometry import synthetic_hip, synthetic_spine

ISO = PixelSpacing(1.0, 1.0, SpacingSource.FALLBACK)


def dicom(pixels: np.ndarray, spacing: PixelSpacing = ISO) -> DicomImage:
    return DicomImage(Path("x.dcm"), "", "", "", None, pixels.astype(np.float32), spacing)


def region(cls: RegionClass) -> RegionPrediction:
    return RegionPrediction(cls, 1.0, {cls: 1.0})


def has_color(img: np.ndarray, rgba) -> bool:
    return bool(np.any(np.all(img[..., :3] == rgba[:3], axis=-1)))


def test_preview_has_physical_proportions() -> None:
    raw = np.zeros((300, 300), np.float32)  # 315 x 180 mm with scanner spacing
    preview = render_preview(dicom(raw, PixelSpacing(1.05, 0.6, SpacingSource.FALLBACK)))
    assert preview.shape == (315 * SCALE, 180 * SCALE)
    assert preview.dtype == np.uint8


def test_overlay_matches_preview_size_and_is_transparent_elsewhere() -> None:
    img = dicom(synthetic_spine(0.0))
    overlay = render_overlay(img, region(RegionClass.SPINE), None)
    assert overlay.shape[:2] == render_preview(img).shape
    assert overlay.shape[2] == 4
    assert (overlay[..., 3] == 0).mean() > 0.9


def test_axis_colour_follows_violation() -> None:
    img = dicom(synthetic_spine(8.0))
    good = render_overlay(img, region(RegionClass.SPINE), QualityAssessment(0, 0.1))
    bad = render_overlay(
        img, region(RegionClass.SPINE), QualityAssessment(1, 0.9, frozenset({Violation.SPINE_AXIS}))
    )
    assert has_color(good, OK) and not has_color(good, BAD)
    assert has_color(bad, BAD)


def test_foreign_object_is_outlined() -> None:
    px = synthetic_spine()
    px[60:64, 20:24] = 1.0
    q = QualityAssessment(1, 0.9, frozenset({Violation.FOREIGN_OBJECTS}))
    overlay = render_overlay(dicom(px), region(RegionClass.SPINE), q)
    ring = overlay[50 * SCALE : 75 * SCALE, 10 * SCALE : 35 * SCALE]
    assert has_color(ring, BAD)


@pytest.mark.parametrize("cls", [RegionClass.HIP_LEFT, RegionClass.HIP_RIGHT])
def test_hip_overlay_is_drawn_in_original_orientation(cls: RegionClass) -> None:
    px = synthetic_hip()
    if cls is RegionClass.HIP_RIGHT:
        px = px[:, ::-1]  # pelvis on the image right, as scanned
    overlay = render_overlay(dicom(px), region(cls), None)
    xs = np.flatnonzero(np.all(overlay[..., :3] == OK[:3], axis=-1).any(axis=0))
    shaft_centre = 125 if cls is RegionClass.HIP_LEFT else px.shape[1] - 1 - 125
    assert abs(np.median(xs) / SCALE - shaft_centre) < 8


def test_encode_png_roundtrip() -> None:
    rgba = np.zeros((10, 12, 4), np.uint8)
    rgba[2, 3] = (255, 0, 0, 255)
    back = cv2.imdecode(np.frombuffer(encode_png(rgba), np.uint8), cv2.IMREAD_UNCHANGED)
    assert back.shape == (10, 12, 4)
    assert tuple(back[2, 3]) == (0, 0, 255, 255)  # BGRA on disk
    assert encode_png(np.zeros((5, 5), np.uint8)).startswith(b"\x89PNG")
