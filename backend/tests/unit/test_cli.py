from __future__ import annotations

import csv
import zipfile

import numpy as np
import pytest

from dxa_qc.cli.batch_process import EXIT_INPUT_ERROR, EXIT_OK, main
from dxa_qc.models.region import RegionClass, RegionClassifier
from tests.factories import make_dicom
from tests.unit.test_region_classifier import dataset


@pytest.fixture(scope="module")
def weights_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("weights")
    images, labels = dataset(10, seed=7)
    RegionClassifier.fit(images, labels, seed=0).save(d / "region_classifier.npz")
    return d


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def test_cli_folder_to_csv(tmp_path, weights_dir):
    from tests.unit.test_region_classifier import synthetic

    img = (synthetic(RegionClass.SPINE, np.random.default_rng(0)) * 255).astype(np.uint8)
    make_dicom(tmp_path / "in" / "x.dcm", img)
    (tmp_path / "in" / "bad.dcm").write_bytes(b"junk")
    code = main(
        [
            "--input",
            str(tmp_path / "in"),
            "--output",
            str(tmp_path / "out"),
            "--format",
            "csv",
            "--weights-dir",
            str(weights_dir),
            "--assessor",
            "prior_baseline",
        ]
    )
    assert code == EXIT_OK
    rows = read_csv(tmp_path / "out" / "results.csv")
    assert [r["processing_status"] for r in rows] == ["Failure", "Success"]
    assert rows[1]["anatomical_region"] == "Поясничный отдел позвоночника"
    assert (tmp_path / "out" / "results_errors.csv").exists()


def test_cli_missing_input(tmp_path, weights_dir):
    code = main(
        [
            "--input",
            str(tmp_path / "nope"),
            "--output",
            str(tmp_path / "out"),
            "--weights-dir",
            str(weights_dir),
        ]
    )
    assert code == EXIT_INPUT_ERROR


def test_cli_unsafe_zip(tmp_path, weights_dir):
    arc = tmp_path / "evil.zip"
    with zipfile.ZipFile(arc, "w") as zf:
        zf.writestr("../evil.dcm", "x")
    code = main(["--input", str(arc), "--output", str(tmp_path / "out"), "--weights-dir", str(weights_dir)])
    assert code == EXIT_INPUT_ERROR
    assert not (tmp_path / "out").exists()


def test_cli_missing_weights(tmp_path):
    make_dicom(tmp_path / "in" / "x.dcm")
    code = main(
        [
            "--input",
            str(tmp_path / "in"),
            "--output",
            str(tmp_path / "out"),
            "--weights-dir",
            str(tmp_path / "no_weights"),
        ]
    )
    assert code == EXIT_INPUT_ERROR


def test_cli_bad_config_dir(tmp_path, weights_dir):
    make_dicom(tmp_path / "in" / "x.dcm")
    code = main(
        [
            "--input",
            str(tmp_path / "in"),
            "--output",
            str(tmp_path / "out"),
            "--weights-dir",
            str(weights_dir),
            "--config-dir",
            str(tmp_path / "no_cfg"),
        ]
    )
    assert code == EXIT_INPUT_ERROR


def test_cli_ensemble_without_quality_models_is_input_error(tmp_path, weights_dir):
    """Region weights exist, quality models don't: refuse to run instead of guessing."""
    make_dicom(tmp_path / "in" / "x.dcm")
    code = main(
        [
            "--input",
            str(tmp_path / "in"),
            "--output",
            str(tmp_path / "out"),
            "--weights-dir",
            str(weights_dir),
        ]
    )
    assert code == EXIT_INPUT_ERROR
    assert not (tmp_path / "out").exists()


def test_cli_series_outputs_are_reproducible(tmp_path, weights_dir):
    import pydicom

    from tests.unit.test_region_classifier import synthetic

    img = (synthetic(RegionClass.SPINE, np.random.default_rng(0)) * 255).astype(np.uint8)
    make_dicom(tmp_path / "in" / "x.dcm", img)
    (tmp_path / "in" / "bad.dcm").write_bytes(b"junk")
    uids = []
    for run in ("a", "b"):
        out = tmp_path / run
        args = ["--input", str(tmp_path / "in"), "--output", str(out), "--format", "csv"]
        assert (
            main([*args, "--weights-dir", str(weights_dir), "--assessor", "prior_baseline", "--series"]) == 0
        )
        details = read_csv_bom(out / "results_details.csv")
        assert [d["processing_status"] for d in details] == ["Failure", "Success"]
        assert details[1]["projection"] == "Прямая (AP)"
        # Failed row: no side outputs; index 1 is the processed image.
        assert sorted(p.name for p in (out / "dicom").iterdir()) == ["00001_sc.dcm", "00001_sr.dcm"]
        assert sorted(p.name for p in (out / "visualization").iterdir()) == [
            "00001_overlay.png",
            "00001_preview.png",
        ]
        uids.append(
            [pydicom.dcmread(out / "dicom" / n).SOPInstanceUID for n in ("00001_sr.dcm", "00001_sc.dcm")]
        )
    assert uids[0] == uids[1]


def test_cli_without_series_writes_only_the_table(tmp_path, weights_dir):
    make_dicom(tmp_path / "in" / "bad.dcm").write_bytes(b"junk")
    args = ["--input", str(tmp_path / "in"), "--output", str(tmp_path / "out"), "--format", "csv"]
    assert main([*args, "--weights-dir", str(weights_dir), "--assessor", "prior_baseline"]) == 0
    assert not (tmp_path / "out" / "dicom").exists()
    assert not (tmp_path / "out" / "results_details.csv").exists()


def read_csv_bom(path):
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))
