"""Proposed corrections for found violations, for a specialist to confirm or reject.

Each proposal says what to change in the positioning or the scan field, and by how much where a
measurement supports it (spine axis tilt), using the thresholds from configs/thresholds.yaml.
Where the image does not give a reliable number, the proposal names the check without an amount.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from dxa_qc.config import Settings
from dxa_qc.domain.taxonomy import ALLOWED_VIOLATIONS, AnatomicalRegion, Violation
from dxa_qc.pipeline.assessment import QualityAssessment


@dataclass(frozen=True, slots=True)
class Correction:
    violation: str
    action: str
    amount: float | None = None  # how much to change, in ``unit``
    unit: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _num(value: float) -> str:
    return f"{value:.1f}".replace(".", ",").removesuffix(",0")


def _spine(v: Violation, f: dict[str, float | bool], s: Settings) -> list[Correction]:
    if v is Violation.SPINE_AXIS:
        tilt = f.get("axis_tilt_deg")
        limit = s.spine.max_axis_tilt_deg
        if isinstance(tilt, (int, float)) and not isinstance(tilt, bool) and abs(tilt) <= limit:
            # Experts also flag 2-5° tilts (pelvis asymmetry): no amount to correct by, only the check.
            return [
                Correction(
                    v.value,
                    f"Наклон оси {_num(abs(tilt))}° в пределах {_num(limit)}°, но укладка признана "
                    "невыровненной: проверьте симметрию таза и уложите пациента по средней линии стола",
                )
            ]
        if isinstance(tilt, (int, float)) and not isinstance(tilt, bool):
            return [
                Correction(
                    v.value,
                    f"Выровнять пациента по средней линии стола: наклон оси {_num(abs(tilt))}° "
                    f"уменьшить до {_num(limit)}° и менее",
                    round(abs(tilt), 1),
                    "°",
                )
            ]
        return [Correction(v.value, f"Выровнять ось позвоночника по вертикали (допустимо до {_num(limit)}°)")]
    if v is Violation.POSITIONING:
        if f.get("iliac_crests_in_field") is False:
            return [
                Correction(
                    v.value,
                    "Сместить нижнюю границу сканирования вниз: "
                    "верхние края подвздошных костей не попали в поле",
                )
            ]
        return [
            Correction(
                v.value, "Проверить верхнюю границу сканирования: в поле должна быть половина тела Th12"
            )
        ]
    return [
        Correction(
            v.value, "Снять металлические предметы одежды из зоны сканирования и повторить исследование"
        )
    ]


def _hip(v: Violation, f: dict[str, float | bool], s: Settings) -> list[Correction]:
    # No amounts for the hip: bone-to-edge distances and shaft tilt do not separate good and bad
    # hips on the dataset (AUC 0.52-0.59, tilt > 5° on 95 of 114 good ones).
    if v is Violation.HIP_ROI:
        return [
            Correction(
                v.value,
                f"Расширить поле сканирования: не менее {_num(s.hip.roi_margin_vertical_cm)} см выше и "
                f"ниже области интереса и {_num(s.hip.roi_margin_lateral_cm)} см латерально",
            )
        ]
    return [
        Correction(
            v.value,
            "Повторить укладку: ротация стопы внутрь 15-25°, малый вертел едва виден; в поле должны быть "
            "большой вертел, шейка бедра и седалищная кость",
        )
    ]


def propose_corrections(
    region: AnatomicalRegion, quality: QualityAssessment, settings: Settings
) -> list[Correction]:
    """Corrections for the violations found, in the report order of the region's criteria."""
    rule = _spine if region is AnatomicalRegion.SPINE else _hip
    found = [v for v in ALLOWED_VIOLATIONS[region] if v in quality.violations]
    return [c for v in found for c in rule(v, dict(quality.findings), settings)]
