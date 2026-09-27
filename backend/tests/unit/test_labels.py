from __future__ import annotations

from pathlib import Path

import pytest
from openpyxl import Workbook

from dxa_qc.data import ClinicalTag, LabelsFormatError, extract_tags, load_labels
from dxa_qc.domain import HipSide, Violation

HEADER0 = [
    "№",
    "study",
    "Позвоночник",
    None,
    None,
    "Проксимальный отдел правого бедра",
    None,
    "Проксимальный отдел левого бедра",
    None,
    "Итог",
    None,
    None,
    None,
]
HEADER1 = [
    None,
    None,
    "корректная укладка",
    "правильно выравнена ось позвоночника (до 5)",
    "наличие посторонних предметов, выраженных артефактов или наложений",
    "позиционирование/ротация",
    "корректности области интересов",
    "позиционирование/ротация",
    "корректности области интересов",
    "Позвоночник",
    "Проксимальный отдел правого бедра",
    "Проксимальный отдел левого бедра",
    "Комментарий",
]


def workbook(tmp_path: Path, rows: list[list], header1=HEADER1) -> Path:
    wb = Workbook()
    ws = wb.active
    ws.append(HEADER0)
    ws.append(header1)
    for r in rows:
        ws.append(r)
    path = tmp_path / "labels.xlsx"
    wb.save(path)
    return path


#            №  study   sp_pos axis fo  rh_pos rh_roi lh_pos lh_roi sp_tot rh_tot lh_tot comment
ROW_FULL = [1, "2.25.1", 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, None]
ROW_NO_HIPS = [2, "2.25.2", 0, 0, 0, None, None, None, None, 1, None, None, "сколиоз"]


def test_parses_violations_and_overall(tmp_path):
    labels = load_labels(workbook(tmp_path, [ROW_FULL, ROW_NO_HIPS]))
    full = labels["2.25.1"]
    assert full.spine.violations == {Violation.SPINE_AXIS}
    assert full.hips[HipSide.RIGHT].violations == {Violation.POSITIONING}
    assert full.hips[HipSide.LEFT].violations == {Violation.HIP_ROI}
    assert full.spine.overall == 1


def test_missing_hips_and_overall_independent_of_criteria(tmp_path):
    lab = load_labels(workbook(tmp_path, [ROW_NO_HIPS]))["2.25.2"]
    assert lab.hips == {}
    # Scoliosis case: expert overall = 1 although no criterion flagged.
    assert lab.spine.violations == frozenset() and lab.spine.overall == 1
    assert lab.tags == {ClinicalTag.SCOLIOSIS}


def test_trailing_empty_rows_ignored(tmp_path):
    assert list(load_labels(workbook(tmp_path, [ROW_FULL, [None] * 13]))) == ["2.25.1"]


def test_study_id_whitespace_stripped(tmp_path):
    row = list(ROW_FULL)
    row[1] = " 2.25.1 "
    assert "2.25.1" in load_labels(workbook(tmp_path, [row]))


def test_changed_header_fails_fast(tmp_path):
    bad = list(HEADER1)
    bad[3] = "что-то другое"
    with pytest.raises(LabelsFormatError, match="unexpected header"):
        load_labels(workbook(tmp_path, [ROW_FULL], header1=bad))


@pytest.mark.parametrize("value", [2, "да", -1])
def test_non_binary_flag_rejected(tmp_path, value):
    row = list(ROW_FULL)
    row[3] = value
    with pytest.raises(LabelsFormatError, match="expected 0/1"):
        load_labels(workbook(tmp_path, [row]))


def test_partial_region_rejected(tmp_path):
    row = list(ROW_FULL)
    row[6] = None  # right hip ROI missing while overall is filled
    with pytest.raises(LabelsFormatError, match="some criteria are empty"):
        load_labels(workbook(tmp_path, [row]))


def test_criteria_without_overall_rejected(tmp_path):
    row = list(ROW_NO_HIPS)
    row[5] = 1
    with pytest.raises(LabelsFormatError, match="overall verdict is empty"):
        load_labels(workbook(tmp_path, [row]))


def test_duplicate_study_rejected(tmp_path):
    with pytest.raises(LabelsFormatError, match="duplicate"):
        load_labels(workbook(tmp_path, [ROW_FULL, ROW_FULL]))


def test_unreadable_file(tmp_path):
    bad = tmp_path / "x.xlsx"
    bad.write_bytes(b"not excel")
    with pytest.raises(LabelsFormatError, match="cannot read"):
        load_labels(bad)


@pytest.mark.parametrize(
    ("comment", "tags"),
    [
        ("Сколиоз, не верня разметка ", {ClinicalTag.SCOLIOSIS}),
        ("правое бедро эндопротезирования", {ClinicalTag.ENDOPROSTHESIS}),
        ("эндопротезирования ТБС", {ClinicalTag.ENDOPROSTHESIS}),
        ("L6, люмбализация", {ClinicalTag.TRANSITIONAL_VERTEBRA}),
        ("перелом", {ClinicalTag.FRACTURE}),
        ("Хороший пример ротации", set()),
        ("L56 not a vertebra", set()),
    ],
)
def test_tags(comment, tags):
    assert extract_tags(comment) == tags
