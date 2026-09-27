from __future__ import annotations

import zipfile

import numpy as np
import pytest

from dxa_qc.domain import AnatomicalRegion, Violation
from dxa_qc.ingest import UnsafeArchiveError
from dxa_qc.models.region import RegionClass, RegionPrediction
from dxa_qc.pipeline import ImageProcessor, PriorBaselineAssessor, QualityAssessment, run_batch
from dxa_qc.report import ProcessingStatus, ReportFormat
from tests.factories import make_dicom


class FakeRegionModel:
    def __init__(self, label=RegionClass.SPINE, confidence=0.99):
        self.label, self.confidence = label, confidence

    def predict(self, pixels):
        return RegionPrediction(self.label, self.confidence, {self.label: self.confidence})


class FixedAssessor:
    name = "fixed"

    def __init__(self, result=None, exc: Exception | None = None):
        self.result = result or QualityAssessment(1, 0.9, frozenset({Violation.SPINE_AXIS}))
        self.exc = exc

    def assess(self, image, region):
        if self.exc:
            raise self.exc
        return self.result


@pytest.fixture
def batch_dir(tmp_path):
    root = tmp_path / "in"
    make_dicom(root / "a.dcm", study_uid="1.1", sop_uid="1.1.1")
    (root / "b_broken.dcm").write_bytes(b"not dicom")
    make_dicom(root / "c.dcm", study_uid="1.1", sop_uid="1.1.2")
    return root


def processor(settings, region=None, assessor=None):
    return ImageProcessor(settings, region or FakeRegionModel(), assessor or FixedAssessor())


def test_bad_file_does_not_stop_batch(settings, batch_dir, tmp_path):
    result = run_batch(batch_dir, tmp_path / "out", ReportFormat.CSV, processor(settings))
    statuses = [(r.path_to_study, r.processing_status) for r in result.rows]
    assert statuses == [
        ("in/a.dcm", ProcessingStatus.SUCCESS),
        ("in/b_broken.dcm", ProcessingStatus.FAILURE),
        ("in/c.dcm", ProcessingStatus.SUCCESS),
    ]
    assert result.n_failed == 1
    assert "not a valid DICOM" in result.rows[1].error
    assert result.report.table.exists() and result.report.errors.exists()


def test_success_row_content(settings, batch_dir, tmp_path):
    row = run_batch(batch_dir, tmp_path / "out", ReportFormat.CSV, processor(settings)).rows[0]
    assert (row.study_uid, row.image_uid) == ("1.1", "1.1.1")
    assert row.anatomical_region is AnatomicalRegion.SPINE
    assert (row.quality_class, row.quality_prob) == (1, 0.9)
    assert row.violation_type == "Не выравнена ось позвоночника"
    assert row.time_of_processing > 0


def test_model_crash_becomes_failure_row(settings, batch_dir, tmp_path):
    proc = processor(settings, assessor=FixedAssessor(exc=RuntimeError("CUDA out of memory")))
    rows = run_batch(batch_dir, tmp_path / "out", ReportFormat.CSV, proc).rows
    assert all(r.processing_status is ProcessingStatus.FAILURE for r in rows)
    assert rows[0].error == "internal error: RuntimeError (details in the service log)"
    assert "CUDA out of memory" not in rows[0].error


def test_violation_outside_region_list_is_failure_not_bad_output(settings, batch_dir, tmp_path, caplog):
    # A hip must never be reported with a spine-only violation.
    bad = FixedAssessor(QualityAssessment(1, 0.8, frozenset({Violation.SPINE_AXIS})))
    proc = processor(settings, region=FakeRegionModel(RegionClass.HIP_LEFT), assessor=bad)
    row = run_batch(batch_dir, tmp_path / "out", ReportFormat.CSV, proc).rows[0]
    assert row.processing_status is ProcessingStatus.FAILURE and row.error.startswith("internal error")
    assert "not allowed" in caplog.text  # the details go to the service log


def test_low_region_confidence_is_failure(settings, batch_dir, tmp_path):
    proc = processor(settings, region=FakeRegionModel(confidence=0.4))
    row = run_batch(batch_dir, tmp_path / "out", ReportFormat.CSV, proc).rows[0]
    assert row.processing_status is ProcessingStatus.FAILURE
    assert "region not recognized" in row.error


def test_zip_input_end_to_end(settings, batch_dir, tmp_path):
    arc = tmp_path / "batch.zip"
    with zipfile.ZipFile(arc, "w") as zf:
        for f in sorted(batch_dir.iterdir()):
            zf.write(f, f"study/{f.name}")
    progress = []
    result = run_batch(
        arc,
        tmp_path / "out",
        ReportFormat.XLSX,
        processor(settings),
        on_progress=lambda n, total: progress.append((n, total)),
    )
    assert [r.path_to_study for r in result.rows] == [
        "batch.zip/study/a.dcm",
        "batch.zip/study/b_broken.dcm",
        "batch.zip/study/c.dcm",
    ]
    assert progress == [(0, 3), (1, 3), (2, 3), (3, 3)]  # (0, n): total known before the first image
    assert result.report.table.suffix == ".xlsx"


def test_unsafe_archive_rejected_as_a_whole(settings, tmp_path):
    arc = tmp_path / "evil.zip"
    with zipfile.ZipFile(arc, "w") as zf:
        zf.writestr("../../x.dcm", "x")
    with pytest.raises(UnsafeArchiveError):
        run_batch(arc, tmp_path / "out", ReportFormat.CSV, processor(settings))


def test_results_are_reproducible(settings, batch_dir, tmp_path):
    def table(i):
        rows = run_batch(batch_dir, tmp_path / f"o{i}", ReportFormat.CSV, processor(settings)).rows
        return [r.model_dump(exclude={"time_of_processing"}) for r in rows]

    assert table(1) == table(2)


# --- assessment contract ---


@pytest.mark.parametrize(
    ("cls", "prob", "violations"),
    [(2, 0.5, set()), (1, 1.5, set()), (0, 0.1, {Violation.POSITIONING})],
)
def test_quality_assessment_invariants(cls, prob, violations):
    with pytest.raises(ValueError):
        QualityAssessment(cls, prob, frozenset(violations))


def test_prior_baseline(settings):
    base = PriorBaselineAssessor()
    spine = base.assess(None, RegionPrediction(RegionClass.SPINE, 1.0, {}))
    hip = base.assess(None, RegionPrediction(RegionClass.HIP_RIGHT, 1.0, {}))
    assert spine.quality_class == 0 and spine.quality_prob == pytest.approx(32 / 99)
    assert hip.quality_prob == pytest.approx(41 / 150)
    with pytest.raises(ValueError, match="missing"):
        PriorBaselineAssessor({AnatomicalRegion.SPINE: 0.3})


def test_fake_pixels_are_used(settings, tmp_path):
    # Sanity: the processor passes decoded pixels (not the path) to the region model.
    seen = {}

    class Spy(FakeRegionModel):
        def predict(self, pixels):
            seen["shape"] = pixels.shape
            return super().predict(pixels)

    make_dicom(tmp_path / "in" / "x.dcm", np.zeros((40, 30), np.uint8))
    run_batch(tmp_path / "in", tmp_path / "out", ReportFormat.CSV, processor(settings, region=Spy()))
    assert seen["shape"] == (40, 30)


def test_on_result_gets_intermediates_and_is_isolated(settings, batch_dir, tmp_path):
    seen = []

    def on_result(index, processed):
        seen.append((index, processed.row.processing_status, processed.image is not None))
        if index == 0:
            raise RuntimeError("preview writer crashed")

    result = run_batch(
        batch_dir, tmp_path / "out", ReportFormat.CSV, processor(settings), on_result=on_result
    )
    assert len(result.rows) == 3  # a failing side output must not lose rows
    assert [s[0] for s in seen] == [0, 1, 2]
    assert seen[1] == (1, ProcessingStatus.FAILURE, False)
    assert seen[0][2] and seen[2][2]
