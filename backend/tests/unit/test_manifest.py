from __future__ import annotations

import shutil
from pathlib import Path, PurePosixPath

import pytest

from dxa_qc.data import RegionAnnotation, RegionLabel, StudyLabels
from dxa_qc.data.manifest import COLUMNS, ManifestError, load_manifest, write_manifest
from dxa_qc.dicom_io import pixel_digest
from dxa_qc.domain.taxonomy import AnatomicalRegion, HipSide, Violation
from dxa_qc.models.region import RegionClass
from dxa_qc.training import make_manifest
from tests.factories import make_dicom
from tests.unit.test_dataset_index import px

SPINE_OK = {
    "path": "a/1.dcm",
    "study_id": "s1",
    "region": "spine",
    "overall": "1",
    "positioning": "0",
    "spine_axis": "1",
    "foreign_objects": "0",
}
HIP_OK = {
    "path": "a/2.dcm",
    "study_id": "s1",
    "region": "hip",
    "side": "left",
    "overall": "0",
    "positioning": "0",
    "hip_roi": "0",
}


def write(tmp_path: Path, *rows: dict[str, str]) -> Path:
    path = tmp_path / "manifest.csv"
    write_manifest(rows, path)
    return path


def test_valid_rows_parse(tmp_path: Path) -> None:
    rows = load_manifest(write(tmp_path, SPINE_OK, HIP_OK, {"path": "b/3.dcm", "study_id": "s2"}))
    spine, hip, unlabeled = rows
    assert spine.line == 2 and spine.region is AnatomicalRegion.SPINE and spine.side is None
    assert spine.violations == {Violation.SPINE_AXIS} and spine.overall == 1
    assert spine.path == (tmp_path / "a" / "1.dcm").resolve()
    assert hip.side is HipSide.LEFT and hip.violations == frozenset() and hip.labeled
    assert not unlabeled.labeled and unlabeled.region is None


def test_images_root_overrides_manifest_dir(tmp_path: Path) -> None:
    root = tmp_path / "images"
    rows = load_manifest(write(tmp_path, SPINE_OK), images_root=root)
    assert rows[0].path == (root / "a" / "1.dcm").resolve()


def test_bom_and_windows_separators_are_accepted(tmp_path: Path) -> None:
    path = tmp_path / "m.csv"
    path.write_text(",".join(COLUMNS) + "\r\na\\b\\1.dcm,s1,,,,,,,,\r\n", encoding="utf-8-sig")
    assert load_manifest(path)[0].path == (tmp_path / "a" / "b" / "1.dcm").resolve()


@pytest.mark.parametrize(
    ("patch", "match"),
    [
        ({"path": "../x.dcm"}, "without '..'"),
        ({"path": "a/../../x.dcm"}, "without '..'"),
        ({"path": "/etc/passwd"}, "without '..'"),
        ({"path": "C:/x.dcm"}, "without '..'"),
        ({"path": ""}, "without '..'"),
        ({"study_id": " "}, "study_id is required"),
        ({"region": "knee"}, "region must be"),
        ({"side": "left"}, "spine rows must not have a side"),
        ({"overall": "2"}, "expected 0, 1 or empty"),
        ({"spine_axis": ""}, "spine_axis is required"),
        ({"hip_roi": "0"}, "hip_roi does not apply to spine"),
        ({"region": ""}, "explicit region"),
        ({"overall": ""}, "criteria filled but overall is empty"),
    ],
)
def test_invalid_rows_are_rejected_with_line(tmp_path: Path, patch: dict[str, str], match: str) -> None:
    with pytest.raises(ManifestError, match=match) as err:
        load_manifest(write(tmp_path, HIP_OK, {**SPINE_OK, **patch}))
    assert "line 3" in str(err.value)


def test_hip_side_value_is_validated(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="side must be"):
        load_manifest(write(tmp_path, {**HIP_OK, "side": "both"}))


def test_duplicate_paths_are_rejected(tmp_path: Path) -> None:
    dup = {**HIP_OK, "path": "a/./1.dcm"}  # same file, different spelling
    with pytest.raises(ManifestError, match="duplicate path"):
        load_manifest(write(tmp_path, SPINE_OK, dup))


def test_missing_columns_and_empty_manifest(tmp_path: Path) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_text("path,study_id\nx.dcm,s1\n", encoding="utf-8")
    with pytest.raises(ManifestError, match="lacks columns"):
        load_manifest(bad)
    with pytest.raises(ManifestError, match="no rows"):
        load_manifest(write(tmp_path))


def test_symlink_escaping_root_is_rejected(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "root"
    root.mkdir()
    try:
        (root / "link").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are not permitted here")
    with pytest.raises(ManifestError, match="escapes"):
        load_manifest(write(root, {"path": "link/x.dcm", "study_id": "s1"}))


# --- make_manifest: organizer format -> manifest -----------------------------------------------


def test_make_manifest_round_trips_through_loader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    studies = tmp_path / "studies"
    spine = make_dicom(studies / "s1" / "ser" / "0.dcm", px(10))
    shutil.copy(spine, studies / "s1" / "ser" / "1.dcm")  # byte-identical repeat: dropped
    hip = make_dicom(studies / "s1" / "ser" / "2.dcm", px(20))
    anns = {
        pixel_digest(spine): RegionAnnotation(
            pixel_digest(spine), "s1", PurePosixPath("s1/ser/0.dcm"), RegionClass.SPINE, ""
        ),
        pixel_digest(hip): RegionAnnotation(
            pixel_digest(hip), "s1", PurePosixPath("s1/ser/2.dcm"), RegionClass.HIP_RIGHT, ""
        ),
    }
    labels = {"s1": StudyLabels("s1", spine=RegionLabel(frozenset({Violation.SPINE_AXIS}), 1), hips={})}
    monkeypatch.setattr(make_manifest, "load_region_annotations", lambda _: anns)
    monkeypatch.setattr(make_manifest, "load_labels", lambda _: labels)

    out = tmp_path / "out" / "manifest.csv"
    out.parent.mkdir()
    argv = ["--studies", str(studies), "--labels", "x.xlsx", "--annotations", "a.csv", "--output", str(out)]
    with pytest.raises(SystemExit):  # studies are outside the default root (the manifest's dir)
        make_manifest.main(argv)
    assert make_manifest.main([*argv, "--images-root", str(tmp_path)]) == 0
    assert "../" not in out.read_text(encoding="utf-8")
    spine_row, hip_row = load_manifest(out, images_root=tmp_path)
    assert spine_row.path == (studies / "s1" / "ser" / "0.dcm").resolve()
    assert spine_row.violations == {Violation.SPINE_AXIS} and spine_row.overall == 1
    assert hip_row.region is AnatomicalRegion.HIP and hip_row.side is HipSide.RIGHT
    assert not hip_row.labeled  # imaged, but experts did not label the right hip
