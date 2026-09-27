"""Parser for the organizer's expert annotation workbook (``разметка.xlsx``).

Semantics established during EDA:
* every criterion cell is a *violation flag*: 1 = violated, 0 = OK, even though the header
  text reads «корректная укладка»;
* a region is annotated iff its «Итог» cell is filled (a study may lack a hip);
* «Итог» is the expert's overall verdict and is NOT always the OR of the criteria
  (scoliosis -> 1 with all criteria 0; a fracture case -> 0 with the axis flag 1), so it is
  kept separately and used as the ``quality_class`` target.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import pandas as pd

from dxa_qc.domain.taxonomy import HipSide, Violation


class LabelsFormatError(ValueError):
    """The workbook does not have the expected layout or contains invalid values."""


class ClinicalTag(StrEnum):
    """Clinically relevant notes from the comment column, used for stratified error analysis."""

    SCOLIOSIS = "scoliosis"
    ENDOPROSTHESIS = "endoprosthesis"
    FRACTURE = "fracture"
    TRANSITIONAL_VERTEBRA = "transitional_vertebra"


_TAG_PATTERNS: dict[ClinicalTag, re.Pattern[str]] = {
    ClinicalTag.SCOLIOSIS: re.compile(r"сколиоз", re.IGNORECASE),
    ClinicalTag.ENDOPROSTHESIS: re.compile(r"эндопротез", re.IGNORECASE),
    ClinicalTag.FRACTURE: re.compile(r"перелом", re.IGNORECASE),
    ClinicalTag.TRANSITIONAL_VERTEBRA: re.compile(r"люмбализ|сакрализ|\bL6\b", re.IGNORECASE),
}


@dataclass(frozen=True, slots=True)
class RegionLabel:
    violations: frozenset[Violation]
    # Expert's overall verdict for the region: 1 = has a quality problem.
    overall: int


@dataclass(frozen=True, slots=True)
class StudyLabels:
    study_id: str  # folder name of the study, NOT the DICOM StudyInstanceUID (they differ)
    spine: RegionLabel | None
    hips: dict[HipSide, RegionLabel] = field(default_factory=dict)
    comment: str = ""
    tags: frozenset[ClinicalTag] = frozenset()


# (column index, expected header substring) - checked to fail fast if the layout changes.
_HEADER_ROW0 = {1: "study", 2: "Позвоночник", 5: "правого бедра", 7: "левого бедра", 9: "Итог"}
_HEADER_ROW1 = {
    2: "укладка",
    3: "ось",
    4: "посторонних",
    5: "позиционирование",
    6: "области интересов",
    7: "позиционирование",
    8: "области интересов",
    9: "Позвоночник",
    10: "правого бедра",
    11: "левого бедра",
    12: "Комментарий",
}
_SPINE_CRITERIA = {2: Violation.POSITIONING, 3: Violation.SPINE_AXIS, 4: Violation.FOREIGN_OBJECTS}
_HIP_CRITERIA = {
    HipSide.RIGHT: {5: Violation.POSITIONING, 6: Violation.HIP_ROI},
    HipSide.LEFT: {7: Violation.POSITIONING, 8: Violation.HIP_ROI},
}
_OVERALL = {"spine": 9, HipSide.RIGHT: 10, HipSide.LEFT: 11}
_COMMENT = 12


def _check_header(raw: pd.DataFrame) -> None:
    for row, expected in ((0, _HEADER_ROW0), (1, _HEADER_ROW1)):
        for col, text in expected.items():
            cell = raw.iat[row, col] if col < raw.shape[1] else None
            if not isinstance(cell, str) or text.lower() not in cell.lower():
                raise LabelsFormatError(
                    f"unexpected header at row {row}, column {col}: {cell!r} (expected {text!r})"
                )


def _flag(value: object, where: str) -> int | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, bool) or not isinstance(value, int | float) or value not in (0, 1):
        raise LabelsFormatError(f"{where}: expected 0/1/empty, got {value!r}")
    return int(value)


def _region(
    row: pd.Series, criteria: dict[int, Violation], overall_col: int, where: str
) -> RegionLabel | None:
    overall = _flag(row.iloc[overall_col], f"{where} overall")
    flags = {v: _flag(row.iloc[c], f"{where} {v.value}") for c, v in criteria.items()}
    if overall is None:
        if any(f is not None for f in flags.values()):
            raise LabelsFormatError(f"{where}: criteria filled but overall verdict is empty")
        return None
    if any(f is None for f in flags.values()):
        raise LabelsFormatError(f"{where}: overall verdict filled but some criteria are empty")
    return RegionLabel(violations=frozenset(v for v, f in flags.items() if f == 1), overall=overall)


def extract_tags(comment: str) -> frozenset[ClinicalTag]:
    return frozenset(tag for tag, pattern in _TAG_PATTERNS.items() if pattern.search(comment))


def load_labels(path: Path) -> dict[str, StudyLabels]:
    """Parse the workbook into ``{study folder name: StudyLabels}``."""
    try:
        raw = pd.read_excel(path, header=None, dtype=object)
    except (OSError, ValueError) as exc:
        raise LabelsFormatError(f"cannot read labels workbook {path}: {exc}") from exc
    if raw.shape[0] < 3 or raw.shape[1] <= _COMMENT:
        raise LabelsFormatError(f"labels workbook is too small: shape {raw.shape}")
    _check_header(raw)

    result: dict[str, StudyLabels] = {}
    for idx in range(2, raw.shape[0]):
        row = raw.iloc[idx]
        study = row.iloc[1]
        if not isinstance(study, str) or not study.strip():
            continue  # trailing empty / summary rows
        study = study.strip()
        where = f"row {idx + 1} ({study})"
        if study in result:
            raise LabelsFormatError(f"{where}: duplicate study")
        comment_cell = row.iloc[_COMMENT]
        comment = comment_cell.strip() if isinstance(comment_cell, str) else ""
        hips = {
            side: label
            for side, crit in _HIP_CRITERIA.items()
            if (label := _region(row, crit, _OVERALL[side], f"{where} {side.value} hip")) is not None
        }
        result[study] = StudyLabels(
            study_id=study,
            spine=_region(row, _SPINE_CRITERIA, _OVERALL["spine"], f"{where} spine"),
            hips=hips,
            comment=comment,
            tags=extract_tags(comment),
        )
    if not result:
        raise LabelsFormatError("labels workbook contains no studies")
    return result
