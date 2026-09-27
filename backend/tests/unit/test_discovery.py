from __future__ import annotations

import os
import zipfile

import pytest

from dxa_qc.ingest import UnsafeArchiveError, collect_inputs
from tests.factories import make_dicom


@pytest.fixture
def limits(settings):
    return settings.ingest


def test_folder_walk_sorted_and_filtered(tmp_path, limits):
    root = tmp_path / "batch"
    make_dicom(root / "s2" / "b.dcm")
    make_dicom(root / "s1" / "a.dcm")
    make_dicom(root / "s1" / "no_extension")  # DICOM detected by content
    (root / "s1" / "notes.txt").write_text("not an image")
    (root / "s1" / "broken.dcm").write_bytes(b"garbage")  # kept: must yield a Failure row
    (root / "DICOMDIR").write_bytes(b"\0" * 128 + b"DICM")
    make_dicom(root / "__MACOSX" / "x.dcm")
    make_dicom(root / ".hidden" / "x.dcm")

    items = collect_inputs(root, tmp_path / "work", limits)
    assert [i.display_path for i in items] == [
        "batch/s1/a.dcm",
        "batch/s1/broken.dcm",
        "batch/s1/no_extension",
        "batch/s2/b.dcm",
    ]
    assert all(i.path.is_absolute() for i in items)


def test_single_file(tmp_path, limits):
    f = make_dicom(tmp_path / "one.dcm")
    items = collect_inputs(f, tmp_path / "work", limits)
    assert [(i.path, i.display_path) for i in items] == [(f.resolve(), "one.dcm")]


def test_zip_input_uses_archive_relative_paths(tmp_path, limits):
    d = make_dicom(tmp_path / "src" / "x.dcm")
    arc = tmp_path / "upload.zip"
    with zipfile.ZipFile(arc, "w") as zf:
        zf.write(d, "study/x.dcm")
    items = collect_inputs(arc, tmp_path / "work", limits)
    assert [i.display_path for i in items] == ["upload.zip/study/x.dcm"]
    assert str(tmp_path / "work") not in items[0].display_path


def test_malicious_zip_propagates(tmp_path, limits):
    arc = tmp_path / "evil.zip"
    with zipfile.ZipFile(arc, "w") as zf:
        zf.writestr("../../etc/passwd", "x")
    with pytest.raises(UnsafeArchiveError):
        collect_inputs(arc, tmp_path / "work", limits)


def test_missing_input(tmp_path, limits):
    with pytest.raises(FileNotFoundError):
        collect_inputs(tmp_path / "nope", tmp_path / "work", limits)


@pytest.mark.skipif(os.name == "nt", reason="symlink creation needs admin rights on Windows")
def test_symlinks_are_not_followed(tmp_path, limits):
    outside = make_dicom(tmp_path / "outside" / "secret.dcm")
    root = tmp_path / "batch"
    root.mkdir()
    (root / "link.dcm").symlink_to(outside)
    (root / "linkdir").symlink_to(outside.parent, target_is_directory=True)
    assert collect_inputs(root, tmp_path / "work", limits) == []
