from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dxa_qc.dicom_io.reader import DicomImage, PixelSpacing, SpacingSource
from dxa_qc.domain.taxonomy import ALLOWED_VIOLATIONS, AnatomicalRegion, HipSide
from dxa_qc.models.quality.assessor import EnsembleQualityAssessor, prepare_image
from dxa_qc.models.quality.embeddings import CONVNEXT_S, DINOV2_S, BackboneSpec
from dxa_qc.models.quality.geometry.hip import ROI_FEATURES
from dxa_qc.models.quality.geometry.spine import AXIS_FEATURES
from dxa_qc.models.quality.heads import LinearHead
from dxa_qc.models.quality.model import GEOMETRY, QualityModelFormatError, RegionQualityModel, TargetEnsemble
from dxa_qc.models.region.classifier import RegionClass, RegionPrediction
from tests.unit.test_quality_geometry import synthetic_hip, synthetic_spine

SPACING = PixelSpacing(1.0, 1.0, SpacingSource.FALLBACK)
DIM = 8


class FakeExtractor:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.preloaded: list[str] = []

    def preload(self, spec: BackboneSpec) -> None:
        self.preloaded.append(spec.key)

    def embed(self, images, spec: BackboneSpec, batch_size: int = 16) -> np.ndarray:
        self.calls.append(spec.key)
        return np.ones((len(images), DIM))


def dicom(pixels: np.ndarray) -> DicomImage:
    return DicomImage(Path("x.dcm"), "1.2", "1.2.3", "1.2.4", 1, pixels.astype(np.float32), SPACING)


def head(
    source: str, features: tuple[str, ...] = (), coef: float = 0.0, intercept: float = 0.0
) -> LinearHead:
    d = len(features) if features else DIM
    c = np.zeros(d)
    c[0] = coef
    return LinearHead(source, features, False, np.zeros(d), np.ones(d), c, intercept)


def model(
    region: AnatomicalRegion, backbone: str, geo: tuple[str, ...], axis_coef: float
) -> RegionQualityModel:
    crit = {
        v: TargetEnsemble((head(GEOMETRY, geo, axis_coef, -1.0), head(backbone)))
        for v in ALLOWED_VIOLATIONS[region]
    }
    return RegionQualityModel(region, crit, TargetEnsemble((head(backbone, intercept=-2.0),)))


def assessor() -> tuple[EnsembleQualityAssessor, FakeExtractor]:
    ext = FakeExtractor()
    models = {
        AnatomicalRegion.SPINE: model(AnatomicalRegion.SPINE, DINOV2_S.key, AXIS_FEATURES, axis_coef=1.0),
        AnatomicalRegion.HIP: model(AnatomicalRegion.HIP, CONVNEXT_S.key, ROI_FEATURES, axis_coef=0.0),
    }
    return EnsembleQualityAssessor(models, ext), ext  # type: ignore[arg-type]


def region(cls: RegionClass) -> RegionPrediction:
    return RegionPrediction(cls, 0.99, {cls: 0.99})


def test_tilted_spine_is_flagged_with_measured_angle() -> None:
    a, ext = assessor()
    straight = a.assess(dicom(synthetic_spine(0.0)), region(RegionClass.SPINE))
    tilted = a.assess(dicom(synthetic_spine(9.0)), region(RegionClass.SPINE))
    assert straight.quality_class == 0 and not straight.violations
    assert tilted.quality_class == 1 and tilted.violations
    assert tilted.findings["axis_tilt_deg"] == pytest.approx(9.0, abs=0.7)
    assert set(tilted.criterion_probs) == set(ALLOWED_VIOLATIONS[AnatomicalRegion.SPINE])
    assert ext.calls == [DINOV2_S.key, DINOV2_S.key]


def test_hip_uses_hip_model_and_reports_findings() -> None:
    a, ext = assessor()
    result = a.assess(dicom(synthetic_hip()), region(RegionClass.HIP_LEFT))
    assert ext.calls == [CONVNEXT_S.key]
    assert result.findings["prosthesis_suspected"] is False
    assert result.findings["field_height_mm"] == 330
    assert 0 <= result.quality_prob <= 1


def test_right_hip_is_mirrored_before_measurement() -> None:
    img = synthetic_hip()
    left = prepare_image(dicom(img), AnatomicalRegion.HIP, HipSide.LEFT)
    right = prepare_image(dicom(img[:, ::-1]), AnatomicalRegion.HIP, HipSide.RIGHT)
    assert np.allclose(left.image, right.image)
    assert np.allclose(left.geometry.values, right.geometry.values)


def test_hip_without_side_is_rejected() -> None:
    with pytest.raises(ValueError, match="side"):
        prepare_image(dicom(synthetic_hip()), AnatomicalRegion.HIP, None)


def test_assessor_requires_both_region_models() -> None:
    with pytest.raises(QualityModelFormatError, match="no quality model"):
        EnsembleQualityAssessor({}, FakeExtractor())  # type: ignore[arg-type]


def test_assessor_rejects_unknown_backbone() -> None:
    models = {
        r: model(r, "mystery_net", AXIS_FEATURES if r is AnatomicalRegion.SPINE else ROI_FEATURES, 0.0)
        for r in AnatomicalRegion
    }
    with pytest.raises(QualityModelFormatError, match="unknown backbones"):
        EnsembleQualityAssessor(models, FakeExtractor())  # type: ignore[arg-type]


def test_load_from_empty_dir_fails_cleanly(tmp_path: Path) -> None:
    with pytest.raises(QualityModelFormatError):
        EnsembleQualityAssessor.load(tmp_path, device="cpu")


def test_preload_builds_every_needed_backbone_once() -> None:
    a, ext = assessor()
    a.preload()
    assert ext.preloaded == sorted([CONVNEXT_S.key, DINOV2_S.key])


def _with_hashes(m: RegionQualityModel, hashes: dict[str, str]) -> RegionQualityModel:
    from dataclasses import replace

    return replace(m, metadata={"backbones": {k: {"sha256": h} for k, h in hashes.items()}})


def test_verify_backbone_files_accepts_matching_hash(tmp_path: Path) -> None:
    from dxa_qc.models.quality.assessor import verify_backbone_files
    from dxa_qc.models.quality.embeddings import file_sha256

    (tmp_path / DINOV2_S.weights_file).write_bytes(b"weights")
    good = file_sha256(tmp_path / DINOV2_S.weights_file)
    m = _with_hashes(model(AnatomicalRegion.SPINE, DINOV2_S.key, AXIS_FEATURES, 0.0), {DINOV2_S.key: good})
    verify_backbone_files({AnatomicalRegion.SPINE: m}, tmp_path)
    # Old model files without recorded hashes are still accepted.
    legacy = model(AnatomicalRegion.SPINE, DINOV2_S.key, (), 0.0)
    verify_backbone_files({AnatomicalRegion.SPINE: legacy}, tmp_path)


def test_verify_backbone_files_rejects_swapped_or_missing_weights(tmp_path: Path) -> None:
    from dxa_qc.models.quality.assessor import verify_backbone_files
    from dxa_qc.models.quality.embeddings import BackboneError

    hip = model(AnatomicalRegion.HIP, CONVNEXT_S.key, ROI_FEATURES, 0.0)
    m = _with_hashes(hip, {CONVNEXT_S.key: "0" * 64})
    with pytest.raises(BackboneError, match="not found"):
        verify_backbone_files({AnatomicalRegion.HIP: m}, tmp_path)
    (tmp_path / CONVNEXT_S.weights_file).write_bytes(b"other weights")
    with pytest.raises(BackboneError, match="sha256 mismatch"):
        verify_backbone_files({AnatomicalRegion.HIP: m}, tmp_path)
