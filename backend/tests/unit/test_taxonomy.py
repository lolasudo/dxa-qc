from __future__ import annotations

import pytest

from dxa_qc.domain import AnatomicalRegion, Violation, format_violations, parse_violations

SPINE, HIP = AnatomicalRegion.SPINE, AnatomicalRegion.HIP


def test_exact_organizer_strings():
    # Byte-exact strings required by the organizer.
    assert SPINE == "Поясничный отдел позвоночника"
    assert HIP == "Проксимальный отдел бедра"
    assert {v.value for v in Violation} == {
        "Некорректная укладка",
        "Не выравнена ось позвоночника",
        "Присутствуют посторонние предметы",
        "Некорректная область интереса",
    }


def test_no_violations_is_empty_string():
    assert format_violations(SPINE, [], ";") == ""


def test_order_is_canonical_regardless_of_input_order():
    got = format_violations(SPINE, [Violation.FOREIGN_OBJECTS, Violation.POSITIONING], ";")
    assert got == "Некорректная укладка;Присутствуют посторонние предметы"


def test_duplicates_collapse():
    assert (
        format_violations(HIP, [Violation.HIP_ROI, Violation.HIP_ROI], ";") == "Некорректная область интереса"
    )


@pytest.mark.parametrize(
    ("region", "violation"),
    [(HIP, Violation.SPINE_AXIS), (HIP, Violation.FOREIGN_OBJECTS), (SPINE, Violation.HIP_ROI)],
)
def test_violation_outside_region_list_raises(region, violation):
    with pytest.raises(ValueError, match="not allowed"):
        format_violations(region, [violation], ";")


@pytest.mark.parametrize("sep", [";", "; "])
def test_round_trip(sep):
    violations = {Violation.POSITIONING, Violation.SPINE_AXIS, Violation.FOREIGN_OBJECTS}
    assert parse_violations(format_violations(SPINE, violations, sep), ";") == violations


@pytest.mark.parametrize("cell", [None, "", "   "])
def test_parse_empty(cell):
    assert parse_violations(cell, ";") == frozenset()


def test_parse_unknown_value_raises():
    with pytest.raises(ValueError):
        parse_violations("Что-то другое", ";")
