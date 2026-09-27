from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from dxa_qc.data.manifest import load_manifest, write_manifest
from dxa_qc.models.region import RegionClass
from dxa_qc.training import region as region_training
from dxa_qc.training.region import load_manifest_images, manifest_class
from tests.factories import make_dicom
from tests.unit.test_dataset_index import px


def dataset(tmp_path: Path) -> Path:
    for i, (name, value) in enumerate([("spine", 10), ("hl", 20), ("hr", 30)]):
        make_dicom(tmp_path / "s1" / f"{name}.dcm", px(value + i))
    shutil.copy(tmp_path / "s1" / "spine.dcm", tmp_path / "s1" / "spine_copy.dcm")
    (tmp_path / "s1" / "broken.dcm").write_bytes(b"not a dicom")
    rows = [
        {"path": "s1/spine.dcm", "study_id": "s1", "region": "spine"},
        {"path": "s1/spine_copy.dcm", "study_id": "s1", "region": "spine"},  # byte-identical: once
        {
            "path": "s1/hl.dcm",
            "study_id": "s1",
            "region": "hip",
            "side": "left",
            "overall": "0",
            "positioning": "0",
            "hip_roi": "0",
        },
        {"path": "s1/hr.dcm", "study_id": "s1", "region": "hip", "side": "right"},
        {"path": "s1/hip_no_side.dcm", "study_id": "s2", "region": "hip"},  # no side: not a class
        {"path": "s1/broken.dcm", "study_id": "s1", "region": "spine"},  # skipped with a warning
    ]
    manifest = tmp_path / "manifest.csv"
    write_manifest(rows, manifest)
    return manifest


def test_manifest_class_needs_explicit_region_and_side(tmp_path: Path) -> None:
    rows = load_manifest(dataset(tmp_path))
    assert [manifest_class(r) for r in rows] == [
        RegionClass.SPINE,
        RegionClass.SPINE,
        RegionClass.HIP_LEFT,
        RegionClass.HIP_RIGHT,
        None,
        RegionClass.SPINE,
    ]


def test_load_manifest_images_dedups_and_skips_unreadable(tmp_path: Path) -> None:
    items = load_manifest_images(load_manifest(dataset(tmp_path)))
    assert [(i.path.name, i.label) for i in items] == [
        ("spine.dcm", RegionClass.SPINE),
        ("hl.dcm", RegionClass.HIP_LEFT),
        ("hr.dcm", RegionClass.HIP_RIGHT),
    ]
    assert all(i.pixels.ndim == 2 for i in items)


def test_load_manifest_images_requires_every_class(tmp_path: Path) -> None:
    rows = [r for r in load_manifest(dataset(tmp_path)) if r.path.name != "hr.dcm"]
    with pytest.raises(ValueError, match="hip_right"):
        load_manifest_images(rows)


def test_main_trains_from_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = dataset(tmp_path)
    monkeypatch.setattr(
        region_training,
        "cross_validate",
        lambda items, c: {"accuracy": 1.0, "n_images": len(items), "n_studies": 1, "errors": []},
    )
    out = tmp_path / "region.npz"
    assert region_training.main(["--manifest", str(manifest), "--output", str(out)]) == 0
    from dxa_qc.models.region import RegionClassifier

    model = RegionClassifier.load(out)
    assert model.predict(np.full((32, 32), 21, np.float32)).label in set(RegionClass)
    with pytest.raises(SystemExit):
        region_training.main(["--output", str(out)])
