from __future__ import annotations

import json
import time

import pytest

from dxa_qc.cli.batch_process import EXIT_OK, main
from dxa_qc.models.quality.assessor import MODEL_FILES
from tests.conftest import BACKEND_ROOT, TEST_SAMPLE_DIR, require

pytestmark = pytest.mark.dataset

WEIGHTS = BACKEND_ROOT / "weights"
# The limit is 3 min per study; this is a per-image budget on a laptop CPU/GPU.
MAX_SECONDS_PER_IMAGE = 20.0


@pytest.fixture(scope="module")
def trained() -> None:
    for name in MODEL_FILES.values():
        require(WEIGHTS / name)
    pytest.importorskip("torch")


def _run(tmp_path, out_name: str) -> list[dict[str, str]]:
    import csv

    out = tmp_path / out_name
    code = main(
        [
            "--input",
            str(require(TEST_SAMPLE_DIR)),
            "--output",
            str(out),
            "--format",
            "csv",
            "--weights-dir",
            str(WEIGHTS),
        ]
    )
    assert code == EXIT_OK
    with (out / "results.csv").open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def test_organizer_sample_end_to_end_is_reproducible(trained, tmp_path):
    started = time.perf_counter()
    first = _run(tmp_path, "a")
    elapsed = time.perf_counter() - started
    assert len(first) == 3
    assert all(r["processing_status"] == "Success" for r in first)
    for r in first:
        assert 0.0 <= float(r["quality_prob"]) <= 1.0
        assert (r["quality_class"] == "0") <= (r["violation_type"] == "")  # class 0 => no violations
    assert elapsed / 3 < MAX_SECONDS_PER_IMAGE
    second = _run(tmp_path, "b")
    strip = [{k: v for k, v in r.items() if k != "time_of_processing"} for r in first]
    assert strip == [{k: v for k, v in r.items() if k != "time_of_processing"} for r in second]


def test_trained_models_carry_cv_evidence(trained):
    from dxa_qc.models.quality.model import RegionQualityModel

    for name in MODEL_FILES.values():
        model = RegionQualityModel.load(WEIGHTS / name)
        report = model.metadata["cv_report"]
        assert isinstance(report, dict)
        auc = report["quality_class"]["roc_auc"]["value"]
        # Regression floor: the prior baseline is 0.5; a retrain that drops near it is a bug.
        assert auc > 0.65, f"{name}: CV ROC-AUC {auc}"
        json.dumps(model.metadata)


def test_real_series_outputs(tmp_path, trained) -> None:
    """Organizer's sample through the real models: every processed image gets a readable SR and
    Secondary Capture filed into its own study, and corrections match the reported violations."""
    import csv

    import pydicom

    out = tmp_path / "series"
    args = ["--input", str(require(TEST_SAMPLE_DIR)), "--output", str(out), "--format", "csv"]
    assert main([*args, "--weights-dir", str(WEIGHTS), "--series"]) == EXIT_OK
    with (out / "results_details.csv").open(encoding="utf-8-sig", newline="") as fh:
        details = list(csv.DictReader(fh))
    ok = [(i, d) for i, d in enumerate(details) if d["processing_status"] == "Success"]
    assert ok
    for i, d in ok:
        sr = pydicom.dcmread(out / "dicom" / f"{i:05d}_sr.dcm")
        sc = pydicom.dcmread(out / "dicom" / f"{i:05d}_sc.dcm")
        assert sr.StudyInstanceUID == sc.StudyInstanceUID
        texts = {item.ConceptNameCodeSequence[0].CodeValue: item for item in sr.ContentSequence}
        assert str(texts["DXAQC-REGION"].TextValue) == d["anatomical_region"]
        assert str(texts["DXAQC-PROJECTION"].TextValue) == d["projection"]
        assert sc.pixel_array.ndim == 3 and sc.pixel_array.shape[2] == 3
        # A proposal for every reported violation, none for a good image.
        assert bool(d["corrections"]) == bool(d["violation_type"])
