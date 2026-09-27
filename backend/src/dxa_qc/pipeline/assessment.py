"""Quality assessor contract, plus the reference baseline all real models are compared against."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from dxa_qc.dicom_io.reader import DicomImage
from dxa_qc.domain.taxonomy import ALLOWED_VIOLATIONS, AnatomicalRegion, Violation
from dxa_qc.models.region.classifier import RegionPrediction


@dataclass(frozen=True, slots=True)
class QualityAssessment:
    quality_class: int
    quality_prob: float
    violations: frozenset[Violation] = field(default_factory=frozenset)
    # Per-criterion probabilities and measured findings (tilt in degrees, field size in mm, ...)
    # for explanations/visualization; not part of the report table.
    criterion_probs: dict[Violation, float] = field(default_factory=dict)
    findings: dict[str, float | bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.quality_class not in (0, 1):
            raise ValueError(f"quality_class must be 0 or 1, got {self.quality_class}")
        if not 0.0 <= self.quality_prob <= 1.0:
            raise ValueError(f"quality_prob must be in [0, 1], got {self.quality_prob}")
        if self.quality_class == 0 and self.violations:
            raise ValueError("violations reported for an image classified as good")


class QualityAssessor(Protocol):
    name: str

    def assess(self, image: DicomImage, region: RegionPrediction) -> QualityAssessment: ...


class PriorBaselineAssessor:
    """Reference baseline, NOT a quality model: predicts the per-region violation prevalence.

    Gives ROC-AUC 0.5 and F1 0 by construction; its purpose is to make the pipeline runnable
    end-to-end and to be the floor every trained assessor must beat.
    Prevalences are from ``разметка.xlsx`` (expert overall verdicts, canonical images).
    """

    name = "prior_baseline"

    def __init__(self, prevalence: dict[AnatomicalRegion, float] | None = None) -> None:
        self.prevalence = prevalence or {AnatomicalRegion.SPINE: 32 / 99, AnatomicalRegion.HIP: 41 / 150}
        missing = set(ALLOWED_VIOLATIONS).difference(self.prevalence)
        if missing:
            raise ValueError(f"prevalence missing for regions: {sorted(r.value for r in missing)}")

    def assess(self, image: DicomImage, region: RegionPrediction) -> QualityAssessment:
        return QualityAssessment(quality_class=0, quality_prob=self.prevalence[region.region])
