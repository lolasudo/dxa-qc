from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pydicom
import pytest

from dxa_qc.report import dicom as dcm
from tests.factories import make_dicom

ROW = {
    "anatomical_region": "Поясничный отдел позвоночника",
    "quality_class": 1,
    "quality_prob": 0.8734,
    "violation_type": "Не выравнена ось позвоночника;Присутствуют посторонние предметы",
    "explanation": "Выявлено нарушение качества.",
    "findings": {"axis_tilt_deg": -6.9, "field_height_mm": 210.0, "iliac_crests_in_field": True},
}
CORRECTIONS = [{"violation": "Не выравнена ось позвоночника", "action": "Выровнять пациента", "amount": 6.9}]
NOW = datetime(2026, 9, 26, 12, 30, 5)


@pytest.fixture
def source(tmp_path: Path) -> Path:
    return make_dicom(
        tmp_path / "src.dcm",
        study_uid="1.2.3.4",
        extra={
            "PatientID": "ANON-1",
            "PatientName": "Anonymized",
            "PatientBirthDate": "Anonymized",  # non-conformant: must not be copied
            "StudyDate": "20250101",
            "AccessionNumber": "A1",
        },
    )


def _items(ds: pydicom.Dataset) -> list[tuple[str, str]]:
    out = []
    for item in ds.ContentSequence:
        code = item.ConceptNameCodeSequence[0].CodeValue
        if item.ValueType == "TEXT":
            out.append((code, str(item.TextValue)))
        else:
            out.append((code, str(item.MeasuredValueSequence[0].NumericValue)))
    return out


def test_source_ref_keeps_study_and_drops_nonconformant_values(source: Path) -> None:
    ref = dcm.SourceRef.from_file(source)
    assert ref.study_uid == "1.2.3.4"
    assert ref.attributes["PatientID"] == "ANON-1"
    assert ref.attributes["StudyDate"] == "20250101"
    assert "PatientBirthDate" not in ref.attributes


def test_sr_content_and_references(source: Path) -> None:
    ref = dcm.SourceRef.from_file(source)
    ds = dcm.build_sr(ref, ROW, series_uid="1.2.9", instance_number=3, corrections=CORRECTIONS, now=NOW)
    assert ds.SOPClassUID == dcm.ENHANCED_SR
    assert ds.Modality == "SR" and ds.StudyInstanceUID == "1.2.3.4" and ds.InstanceNumber == 3
    assert ds.VerificationFlag == "UNVERIFIED" and ds.CompletionFlag == "COMPLETE"
    assert ds.PatientBirthDate == ""  # Type 2: present, empty
    evidence = ds.CurrentRequestedProcedureEvidenceSequence[0]
    assert evidence.ReferencedSeriesSequence[0].ReferencedSOPSequence[0].ReferencedSOPInstanceUID == (
        ref.sop_instance_uid
    )
    items = _items(ds)
    assert ("DXAQC-VERDICT", "Есть нарушение") in items
    assert ("DXAQC-PROB", "87.34") in items
    assert [v for c, v in items if c == "DXAQC-VIOLATION"] == [
        "Не выравнена ось позвоночника",
        "Присутствуют посторонние предметы",
    ]
    assert ("DXAQC-AXIS", "6.9") in items  # absolute value, degrees
    assert ("DXAQC-FH", "210") in items
    assert ("DXAQC-CORRECTION", "Выровнять пациента") in items
    # Boolean findings are not measurements.
    assert not any(code.startswith("DXAQC-ILIAC") for code, _ in items)


def test_sr_review_marks_report_verified(source: Path) -> None:
    ref = dcm.SourceRef.from_file(source)
    review = dcm.Review("corrected", 0, (), "Артефакт вне зоны ROI", NOW)
    ds = dcm.build_sr(ref, ROW, series_uid="1.2.9", instance_number=1, corrections=[], review=review, now=NOW)
    assert ds.VerificationFlag == "VERIFIED"
    assert ds.VerifyingObserverSequence[0].VerificationDateTime == "20260926123005"
    text = dict(_items(ds))["DXAQC-REVIEW"]
    assert text == "Решение специалиста: исправлено; качественное. Артефакт вне зоны ROI"


def test_outputs_are_valid_files_with_reproducible_uids(source: Path, tmp_path: Path) -> None:
    ref = dcm.SourceRef.from_file(source)
    preview = np.full((40, 30), 128, np.uint8)
    overlay = np.zeros((40, 30, 4), np.uint8)
    overlay[5:10, 5:10] = (255, 0, 0, 255)
    paths = [
        dcm.write_outputs(
            ref, ROW, CORRECTIONS, preview, overlay, tmp_path / f"out{i}", series_key="job", index=0
        )
        for i in range(2)
    ]
    first = [pydicom.dcmread(p) for p in paths[0]]
    second = [pydicom.dcmread(p) for p in paths[1]]
    for a, b in zip(first, second, strict=True):
        assert (a.SOPInstanceUID, a.SeriesInstanceUID) == (b.SOPInstanceUID, b.SeriesInstanceUID)
        assert a.file_meta.MediaStorageSOPInstanceUID == a.SOPInstanceUID
    sr, sc = first
    assert sr.SeriesInstanceUID != sc.SeriesInstanceUID
    assert sc.SOPClassUID == dcm.SECONDARY_CAPTURE and sc.PhotometricInterpretation == "RGB"
    pixels = sc.pixel_array
    assert pixels.shape == (40, 30, 3)
    assert tuple(pixels[7, 7]) == (255, 0, 0) and tuple(pixels[0, 0]) == (128, 128, 128)
    assert sc.SourceImageSequence[0].ReferencedSOPInstanceUID == ref.sop_instance_uid
    assert sr.ContentSequence[0].TextValue == ROW["anatomical_region"]  # UTF-8 round trip


def test_other_job_gets_other_uids(source: Path) -> None:
    ref = dcm.SourceRef.from_file(source)
    a = dcm.build_sr(
        ref, ROW, series_uid=dcm.uid("job-a", ref.study_uid, "sr"), instance_number=1, corrections=[]
    )
    b = dcm.build_sr(
        ref, ROW, series_uid=dcm.uid("job-b", ref.study_uid, "sr"), instance_number=1, corrections=[]
    )
    assert a.SOPInstanceUID != b.SOPInstanceUID


def test_composite_blends_by_alpha() -> None:
    preview = np.full((1, 2), 100, np.uint8)
    overlay = np.array([[[200, 0, 0, 0], [200, 0, 0, 255]]], np.uint8)
    assert dcm.composite(preview, overlay).tolist() == [[[100, 100, 100], [200, 0, 0]]]


def test_secondary_capture_rejects_non_rgb(source: Path) -> None:
    ref = dcm.SourceRef.from_file(source)
    with pytest.raises(ValueError, match="RGB"):
        dcm.build_secondary_capture(ref, np.zeros((4, 4), np.uint8), series_uid="1.2", instance_number=1)
