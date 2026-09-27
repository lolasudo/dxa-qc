"""Trained quality assessor: geometry + frozen-backbone embeddings -> per-region ensemble model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dxa_qc.dicom_io.reader import DicomImage
from dxa_qc.domain.taxonomy import AnatomicalRegion, HipSide
from dxa_qc.models.quality.embeddings import (
    ALL_SPECS,
    BackboneError,
    BackboneSpec,
    EmbeddingExtractor,
    file_sha256,
)
from dxa_qc.models.quality.geometry import (
    FeatureVector,
    canonical_hip,
    measure_hip,
    measure_spine,
    to_isotropic,
)
from dxa_qc.models.quality.model import ModelInputs, QualityModelFormatError, RegionQualityModel
from dxa_qc.models.region.classifier import RegionPrediction
from dxa_qc.pipeline.assessment import QualityAssessment

MODEL_FILES = {AnatomicalRegion.SPINE: "quality_spine.npz", AnatomicalRegion.HIP: "quality_hip.npz"}
BACKBONES_SUBDIR = "backbones"


@dataclass(frozen=True, slots=True)
class PreparedImage:
    image: np.ndarray  # isotropic 1 mm, canonical orientation
    geometry: FeatureVector
    findings: dict[str, float | bool]


def prepare_image(dicom: DicomImage, region: AnatomicalRegion, side: HipSide | None) -> PreparedImage:
    """Shared by training and inference, so both see exactly the same preprocessing."""
    iso = to_isotropic(dicom.pixels, dicom.spacing)
    if region is AnatomicalRegion.SPINE:
        spine = measure_spine(iso)
        findings: dict[str, float | bool] = {
            "axis_tilt_deg": round(spine.axis_tilt_deg, 2),
            "field_height_mm": float(iso.shape[0]),
            # Iliac crests in the bottom zone of the field. Missing on 5 of 6 badly positioned
            # images and on 2 of 93 good ones.
            "iliac_crests_in_field": spine.features.as_dict()["crest_extent_mm"] > 0,
        }
        return PreparedImage(iso, spine.features, findings)
    if side is None:
        raise ValueError("hip image needs a side to be put into canonical orientation")
    canonical = canonical_hip(iso, side)
    hip = measure_hip(canonical)
    findings = {
        "shaft_tilt_deg": round(hip.shaft_tilt_deg, 2),
        "field_height_mm": float(canonical.shape[0]),
        "field_width_mm": float(canonical.shape[1]),
        "prosthesis_suspected": hip.prosthesis_suspected,
    }
    return PreparedImage(canonical, hip.features, findings)


def model_backbone_specs(model: RegionQualityModel) -> dict[str, BackboneSpec]:
    """Backbone specs recorded in the model file (any timm model a config trained with);
    models trained before specs were recorded fall back to the built-in registry."""
    recorded = model.metadata.get("backbones")
    specs: dict[str, BackboneSpec] = {}
    for key in model.backbones:
        entry = recorded.get(key) if isinstance(recorded, dict) else None
        try:
            if isinstance(entry, dict) and "crop" in entry:
                spec = BackboneSpec.from_dict(entry)
            elif key in ALL_SPECS:
                spec = ALL_SPECS[key]
            else:
                raise KeyError(key)
        except (KeyError, TypeError, ValueError) as exc:
            raise QualityModelFormatError(f"model needs unknown backbones ['{key}']: {exc}") from exc
        if spec.key != key:
            raise QualityModelFormatError(f"backbone spec key mismatch: {spec.key!r} != {key!r}")
        specs[key] = spec
    return specs


def verify_backbone_files(models: dict[AnatomicalRegion, RegionQualityModel], backbones_dir: Path) -> None:
    """Refuse to run with backbone weights other than the ones the heads were trained on:
    a swapped or corrupted file would silently produce wrong (not failing) predictions."""
    checked: dict[str, str] = {}
    for model in models.values():
        recorded = model.metadata.get("backbones")
        if not isinstance(recorded, dict):
            continue
        for key, spec in model_backbone_specs(model).items():
            expected = (recorded.get(key) or {}).get("sha256")
            if not expected:
                continue
            path = backbones_dir / spec.weights_file
            if not path.is_file():
                raise BackboneError(f"backbone weights not found: {path}")
            actual = checked.setdefault(spec.weights_file, file_sha256(path))
            if actual != expected:
                raise BackboneError(f"backbone weights {path.name} do not match the model (sha256 mismatch)")


class EnsembleQualityAssessor:
    name = "ensemble"

    def __init__(
        self, models: dict[AnatomicalRegion, RegionQualityModel], extractor: EmbeddingExtractor
    ) -> None:
        missing = set(MODEL_FILES).difference(models)
        if missing:
            raise QualityModelFormatError(f"no quality model for {sorted(r.value for r in missing)}")
        self.specs: dict[str, BackboneSpec] = {}
        for model in models.values():
            self.specs.update(model_backbone_specs(model))
        self.models = models
        self.extractor = extractor

    @classmethod
    def load(
        cls, weights_dir: Path, device: str | None = None, verify: bool = True
    ) -> EnsembleQualityAssessor:
        weights_dir = Path(weights_dir)
        models = {region: RegionQualityModel.load(weights_dir / name) for region, name in MODEL_FILES.items()}
        if verify:
            verify_backbone_files(models, weights_dir / BACKBONES_SUBDIR)
        return cls(models, EmbeddingExtractor(weights_dir / BACKBONES_SUBDIR, device))

    def preload(self) -> None:
        for key in sorted({k for m in self.models.values() for k in m.backbones}):
            self.extractor.preload(self.specs[key])

    def assess(self, image: DicomImage, region: RegionPrediction) -> QualityAssessment:
        model = self.models[region.region]
        prepared = prepare_image(image, region.region, region.side)
        geometry = {n: np.array([v]) for n, v in prepared.geometry.as_dict().items()}
        embeddings = {key: self.extractor.embed([prepared.image], self.specs[key]) for key in model.backbones}
        pred = model.predict(ModelInputs(geometry, embeddings))[0]
        return QualityAssessment(
            quality_class=pred.quality_class,
            quality_prob=pred.quality_prob,
            violations=pred.violations,
            criterion_probs=pred.criterion_probs,
            findings=prepared.findings,
        )
