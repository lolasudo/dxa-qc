from __future__ import annotations

import csv
import shutil
from pathlib import Path, PurePosixPath

import numpy as np
import pytest

from dxa_qc.data import (
    AnnotationsFormatError,
    DatasetIndexError,
    RegionAnnotation,
    RegionLabel,
    StudyLabels,
    build_index,
    load_region_annotations,
    write_index_csv,
)
from dxa_qc.dicom_io import DicomReadError, pixel_digest
from dxa_qc.domain import HipSide, Violation
from dxa_qc.models.region import RegionClass
from tests.factories import make_dicom

HEADER = "pixel_sha256_16,study_id,example_path,region,side,note\n"


def px(value: int) -> np.ndarray:
    return np.full((32, 32), value, np.uint8)


# --- pixel digest ---


def test_digest_ignores_headers_but_not_pixels(tmp_path):
    a = make_dicom(tmp_path / "a.dcm", px(10), study_uid="1.1")
    b = make_dicom(tmp_path / "b.dcm", px(10), study_uid="2.2")
    c = make_dicom(tmp_path / "c.dcm", px(11))
    assert pixel_digest(a) == pixel_digest(b) != pixel_digest(c)


def test_digest_on_non_dicom(tmp_path):
    bad = tmp_path / "x.dcm"
    bad.write_bytes(b"junk")
    with pytest.raises(DicomReadError):
        pixel_digest(bad)


# --- annotations loader ---


def write_ann(tmp_path: Path, body: str, header: str = HEADER) -> Path:
    p = tmp_path / "ann.csv"
    p.write_text(header + body, encoding="utf-8")
    return p


def test_annotations_parse(tmp_path):
    anns = load_region_annotations(
        write_ann(tmp_path, "aa,s1,s1/a.dcm,spine,,\nbb,s1,s1/b.dcm,hip,left,note\n")
    )
    assert anns["aa"].label is RegionClass.SPINE
    assert anns["bb"].label is RegionClass.HIP_LEFT and anns["bb"].note == "note"


@pytest.mark.parametrize(
    ("body", "match"),
    [
        ("aa,s1,s1/a.dcm,spine,left,\n", "invalid region/side"),
        ("aa,s1,s1/a.dcm,hip,,\n", "invalid region/side"),
        ("aa,s1,../../etc/passwd,spine,,\n", "relative"),
        ("aa,s1,s1/a.dcm,spine,,\naa,s1,s1/b.dcm,spine,,\n", "duplicate"),
        ("", "no annotations"),
    ],
)
def test_annotations_invalid(tmp_path, body, match):
    with pytest.raises(AnnotationsFormatError, match=match):
        load_region_annotations(write_ann(tmp_path, body))


def test_annotations_wrong_columns(tmp_path):
    with pytest.raises(AnnotationsFormatError, match="columns"):
        load_region_annotations(write_ann(tmp_path, "aa,s1\n", header="hash,study\n"))


# --- index ---


@pytest.fixture
def mini_dataset(tmp_path):
    root = tmp_path / "studies"
    spine = make_dicom(root / "s1" / "ser" / "0.dcm", px(10))
    shutil.copy(spine, root / "s1" / "ser" / "1.dcm")  # byte-identical repeat
    hip = make_dicom(root / "s1" / "ser" / "2.dcm", px(20))
    anns = {
        pixel_digest(spine): RegionAnnotation(
            pixel_digest(spine), "s1", PurePosixPath("s1/ser/0.dcm"), RegionClass.SPINE, ""
        ),
        pixel_digest(hip): RegionAnnotation(
            pixel_digest(hip), "s1", PurePosixPath("s1/ser/2.dcm"), RegionClass.HIP_RIGHT, ""
        ),
    }
    labels = {"s1": StudyLabels("s1", spine=RegionLabel(frozenset({Violation.SPINE_AXIS}), 1), hips={})}
    return root, anns, labels


def test_index_marks_duplicates_and_attaches_labels(mini_dataset):
    root, anns, labels = mini_dataset
    idx = build_index(root, anns, labels)
    assert [(i.path.name, i.region_class, i.is_duplicate) for i in idx] == [
        ("0.dcm", RegionClass.SPINE, False),
        ("1.dcm", RegionClass.SPINE, True),
        ("2.dcm", RegionClass.HIP_RIGHT, False),
    ]
    assert idx[0].label.violations == {Violation.SPINE_AXIS}
    assert idx[2].label is None  # right hip imaged but not annotated by experts


def test_index_attaches_hip_label_by_side(mini_dataset):
    root, anns, labels = mini_dataset
    labels["s1"] = StudyLabels("s1", spine=None, hips={HipSide.RIGHT: RegionLabel(frozenset(), 0)})
    assert build_index(root, anns, labels)[2].label == RegionLabel(frozenset(), 0)


def test_index_fails_on_unannotated_image(mini_dataset):
    root, anns, labels = mini_dataset
    make_dicom(root / "s1" / "ser" / "3.dcm", px(30))
    with pytest.raises(DatasetIndexError, match="no region annotation"):
        build_index(root, anns, labels)


def test_index_fails_on_study_without_labels(mini_dataset):
    root, anns, _ = mini_dataset
    with pytest.raises(DatasetIndexError, match="without expert labels"):
        build_index(root, anns, {})


def test_index_fails_when_annotation_from_other_study(mini_dataset):
    root, anns, labels = mini_dataset
    shutil.copytree(root / "s1", root / "s2")
    labels["s2"] = labels["s1"]
    with pytest.raises(DatasetIndexError, match="another study"):
        build_index(root, anns, labels)


def test_index_csv(mini_dataset, tmp_path):
    root, anns, labels = mini_dataset
    out = tmp_path / "index.csv"
    write_index_csv(build_index(root, anns, labels), out, root)
    with out.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["path"] == "s1/ser/0.dcm"
    assert rows[0]["v_spine_axis"] == "1" and rows[0]["v_hip_roi"] == ""  # not applicable to spine
    assert rows[2]["labeled"] == "0" and rows[2]["overall"] == ""
