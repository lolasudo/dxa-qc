from __future__ import annotations

import pytest

from dxa_qc.domain.taxonomy import AnatomicalRegion, Violation
from dxa_qc.explain import explain, explain_failure
from dxa_qc.pipeline.assessment import QualityAssessment

SPINE, HIP = AnatomicalRegion.SPINE, AnatomicalRegion.HIP


def test_good_spine_reports_measured_tilt_and_probability() -> None:
    q = QualityAssessment(0, 0.123, findings={"axis_tilt_deg": -1.84})
    text = explain(SPINE, q, 5.0)
    assert text.startswith("Нарушений качества не выявлено.")
    assert "1,8°" in text and "до 5°" in text
    assert text.endswith("Вероятность нарушения качества: 12%.")


def test_violations_listed_in_canonical_order_with_probabilities_and_hints() -> None:
    q = QualityAssessment(
        1,
        0.91,
        frozenset({Violation.FOREIGN_OBJECTS, Violation.POSITIONING}),
        criterion_probs={
            Violation.FOREIGN_OBJECTS: 0.88,
            Violation.POSITIONING: 0.6,
            Violation.SPINE_AXIS: 0.1,
        },
    )
    text = explain(SPINE, q, 5.0)
    assert text.index("некорректная укладка (60%)") < text.index("присутствуют посторонние предметы (88%)")
    assert "подвздошных костей" in text and "металл одежды" in text


def test_bad_without_named_violation_asks_for_review() -> None:
    text = explain(SPINE, QualityAssessment(1, 0.7), 5.0)
    assert "визуальная проверка" in text


def test_hip_prosthesis_and_field_size() -> None:
    q = QualityAssessment(
        0, 0.2, findings={"prosthesis_suspected": True, "field_height_mm": 315.0, "field_width_mm": 180.0}
    )
    text = explain(HIP, q, 5.0)
    assert "эндопротез" in text
    assert "18,0 x 31,5 см" in text


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        ("not a valid DICOM file: File is missing DICOM File Meta", "не является корректным DICOM"),
        ("anatomical region not recognized (confidence 0.41 < 0.6)", "анатомическую область"),
        ("file is too large (99 bytes > 10)", "размер"),
        (None, "Ошибка обработки"),
        ("internal error: KeyError: x", "Ошибка обработки"),
    ],
)
def test_failure_messages_are_russian(error, expected) -> None:
    assert expected in explain_failure(error)
