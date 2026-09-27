from __future__ import annotations

from dxa_qc.config import Settings
from dxa_qc.domain.taxonomy import AnatomicalRegion, Violation
from dxa_qc.explain import propose_corrections
from dxa_qc.pipeline.assessment import QualityAssessment

SPINE, HIP = AnatomicalRegion.SPINE, AnatomicalRegion.HIP


def _q(violations: set[Violation], **findings: float | bool) -> QualityAssessment:
    return QualityAssessment(
        1 if violations else 0, 0.9 if violations else 0.1, frozenset(violations), findings=findings
    )


def test_good_image_needs_no_correction(settings: Settings) -> None:
    assert propose_corrections(SPINE, _q(set(), axis_tilt_deg=1.0), settings) == []


def test_axis_tilt_beyond_limit_gives_amount(settings: Settings) -> None:
    [c] = propose_corrections(SPINE, _q({Violation.SPINE_AXIS}, axis_tilt_deg=-6.94), settings)
    assert c.violation == Violation.SPINE_AXIS.value
    assert (c.amount, c.unit) == (6.9, "°")
    assert "6,9°" in c.action and "5°" in c.action


def test_axis_flag_within_limit_names_check_without_amount(settings: Settings) -> None:
    # Experts flag 2-5° tilts too: "reduce 3° to 5°" would contradict itself.
    [c] = propose_corrections(SPINE, _q({Violation.SPINE_AXIS}, axis_tilt_deg=3.2), settings)
    assert c.amount is None
    assert "в пределах 5°" in c.action and "симметрию таза" in c.action


def test_axis_without_measurement_falls_back_to_rule(settings: Settings) -> None:
    [c] = propose_corrections(SPINE, _q({Violation.SPINE_AXIS}), settings)
    assert c.amount is None and "5°" in c.action


def test_spine_positioning_depends_on_iliac_crests(settings: Settings) -> None:
    [low] = propose_corrections(SPINE, _q({Violation.POSITIONING}, iliac_crests_in_field=False), settings)
    assert "нижнюю границу" in low.action
    [high] = propose_corrections(SPINE, _q({Violation.POSITIONING}, iliac_crests_in_field=True), settings)
    assert "Th12" in high.action


def test_spine_corrections_follow_criteria_order(settings: Settings) -> None:
    q = _q({Violation.FOREIGN_OBJECTS, Violation.POSITIONING, Violation.SPINE_AXIS}, axis_tilt_deg=7.0)
    got = [c.violation for c in propose_corrections(SPINE, q, settings)]
    assert got == [Violation.POSITIONING.value, Violation.SPINE_AXIS.value, Violation.FOREIGN_OBJECTS.value]


def test_hip_roi_names_the_rule_without_amounts(settings: Settings) -> None:
    # Bone-to-edge distances are unreliable on the dataset: even when present they give no amount.
    q = _q({Violation.HIP_ROI}, field_height_mm=300.0, field_width_mm=160.0)
    [c] = propose_corrections(HIP, q, settings)
    assert c.amount is None and c.unit is None
    assert "3 см выше и ниже" in c.action and "2 см латерально" in c.action


def test_hip_positioning_names_rotation_without_tilt_amount(settings: Settings) -> None:
    # A shaft tilt over 5° is normal anatomy on most correctly positioned hips: not advised on.
    [c] = propose_corrections(HIP, _q({Violation.POSITIONING}, shaft_tilt_deg=-8.26), settings)
    assert c.amount is None and "ротация стопы внутрь" in c.action and "седалищная кость" in c.action


def test_hip_both_violations_in_criteria_order(settings: Settings) -> None:
    q = _q({Violation.HIP_ROI, Violation.POSITIONING})
    got = [c.violation for c in propose_corrections(HIP, q, settings)]
    assert got == [Violation.POSITIONING.value, Violation.HIP_ROI.value]


def test_boolean_findings_are_not_numbers(settings: Settings) -> None:
    [c] = propose_corrections(SPINE, _q({Violation.SPINE_AXIS}, axis_tilt_deg=True), settings)
    assert c.amount is None


def test_to_dict_is_json_ready(settings: Settings) -> None:
    [c] = propose_corrections(SPINE, _q({Violation.SPINE_AXIS}, axis_tilt_deg=9.0), settings)
    assert c.to_dict() == {"violation": c.violation, "action": c.action, "amount": 9.0, "unit": "°"}
