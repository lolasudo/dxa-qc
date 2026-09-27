"""Closed vocabularies fixed by the organizer.

Output strings must match the organizer's wording byte-for-byte, including the
original spelling «выравнена», so they are defined once here and nowhere else.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum


class AnatomicalRegion(StrEnum):
    SPINE = "Поясничный отдел позвоночника"
    HIP = "Проксимальный отдел бедра"


class HipSide(StrEnum):
    """Internal only: needed to map study-level labels to images; never written to the report."""

    RIGHT = "right"
    LEFT = "left"


class Violation(StrEnum):
    POSITIONING = "Некорректная укладка"
    SPINE_AXIS = "Не выравнена ось позвоночника"
    FOREIGN_OBJECTS = "Присутствуют посторонние предметы"
    HIP_ROI = "Некорректная область интереса"


# Order defines the order inside the output string, so results are deterministic.
ALLOWED_VIOLATIONS: dict[AnatomicalRegion, tuple[Violation, ...]] = {
    AnatomicalRegion.SPINE: (Violation.POSITIONING, Violation.SPINE_AXIS, Violation.FOREIGN_OBJECTS),
    AnatomicalRegion.HIP: (Violation.POSITIONING, Violation.HIP_ROI),
}


def format_violations(region: AnatomicalRegion, violations: Iterable[Violation], separator: str) -> str:
    """Build the ``violation_type`` cell; empty string means "no violations".

    Raises ``ValueError`` for a violation not defined for the region - that is a bug
    upstream, and emitting it would produce an answer outside the closed list.
    """
    found = set(violations)
    allowed = ALLOWED_VIOLATIONS[region]
    illegal = found.difference(allowed)
    if illegal:
        names = ", ".join(sorted(v.value for v in illegal))
        raise ValueError(f"Violations not allowed for region {region.value!r}: {names}")
    return separator.join(v.value for v in allowed if v in found)


def parse_violations(cell: str | None, separator: str) -> frozenset[Violation]:
    """Inverse of :func:`format_violations`; tolerant to surrounding whitespace."""
    if not cell or not cell.strip():
        return frozenset()
    return frozenset(Violation(part.strip()) for part in cell.split(separator) if part.strip())
