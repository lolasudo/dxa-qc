"""DICOM outputs: the report of the result (Enhanced SR: text plus measured numbers)
and a derived series with the violation overlay (Secondary Capture), both filed into the source study.

UIDs are derived from stable inputs (job, study, source image), so a repeated run on the same data
produces the same identifiers. Patient and study attributes are copied from the source
header as they are: the service neither needs nor adds personal data.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import PYDICOM_IMPLEMENTATION_UID, ExplicitVRLittleEndian, generate_uid

# Enhanced, not Basic Text SR: the report carries NUM items, which Basic Text SR does not allow.
ENHANCED_SR = "1.2.840.10008.5.1.4.1.1.88.22"
SECONDARY_CAPTURE = "1.2.840.10008.5.1.4.1.1.7"
SCHEME = "99DXAQC"  # private coding scheme of this service
MANUFACTURER = "DXA QC"

# Patient / study attributes copied from the source image so the outputs file into the same study.
_COPIED = (
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "PatientSex",
    "StudyDate",
    "StudyTime",
    "StudyID",
    "AccessionNumber",
    "ReferringPhysicianName",
    "StudyDescription",
)
_TYPE2 = ("PatientName", "PatientID", "PatientBirthDate", "PatientSex", "StudyDate", "StudyTime",
          "StudyID", "AccessionNumber", "ReferringPhysicianName")  # fmt: skip


_FORMATS = {
    "PatientBirthDate": re.compile(r"\d{8}"),
    "StudyDate": re.compile(r"\d{8}"),
    "StudyTime": re.compile(r"\d{2}(\d{2}(\d{2}(\.\d{1,6})?)?)?"),
    "PatientSex": re.compile(r"[MFO]"),
}


def _conforms(key: str, value: Any) -> bool:
    """Values like "Anonymized" in date/time/code fields would make the outputs non-conformant."""
    pattern = _FORMATS.get(key)
    return pattern is None or bool(pattern.fullmatch(str(value)))


@dataclass(frozen=True, slots=True)
class SourceRef:
    """What the outputs need from the source image; kept after the source file is deleted."""

    study_uid: str
    sop_class_uid: str
    sop_instance_uid: str
    series_uid: str
    attributes: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: Path) -> SourceRef:
        with warnings.catch_warnings():  # anonymized sources often carry non-conformant values
            warnings.simplefilter("ignore")
            ds = pydicom.dcmread(path, stop_before_pixels=True)
            attrs = {k: ds.get(k) for k in _COPIED if k in ds and _conforms(k, ds.get(k))}
        return cls(
            study_uid=str(ds.get("StudyInstanceUID", "")) or generate_uid(),
            sop_class_uid=str(ds.get("SOPClassUID", "")),
            sop_instance_uid=str(ds.get("SOPInstanceUID", "")),
            series_uid=str(ds.get("SeriesInstanceUID", "")),
            attributes={k: v for k, v in attrs.items() if v is not None},
        )


@dataclass(frozen=True, slots=True)
class Review:
    """Specialist's decision (see the review endpoint); turns the SR into a verified report."""

    decision: str  # "confirmed" | "corrected"
    quality_class: int
    violations: tuple[str, ...]
    comment: str
    reviewed_at: datetime


_F = TypeVar("_F", bound=Callable[..., Any])


def _quiet(fn: _F) -> _F:
    """pydicom warns on the source study's non-conformant UIDs; they are kept on purpose."""

    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return fn(*args, **kwargs)

    return wrapper  # type: ignore[return-value]


def uid(*parts: str) -> str:
    return generate_uid(entropy_srcs=["dxa-qc", *parts])


def _base(
    ref: SourceRef, sop_class: str, sop_uid: str, series_uid: str, modality: str, now: datetime
) -> Dataset:
    ds = Dataset()
    ds.SpecificCharacterSet = "ISO_IR 192"  # UTF-8: Russian text
    ds.SOPClassUID = sop_class
    ds.SOPInstanceUID = sop_uid
    for key in _TYPE2:
        setattr(ds, key, "")
    for key, value in ref.attributes.items():
        setattr(ds, key, value)
    ds.StudyInstanceUID = ref.study_uid
    ds.SeriesInstanceUID = series_uid
    ds.Modality = modality
    ds.Manufacturer = MANUFACTURER
    ds.InstanceCreationDate = now.strftime("%Y%m%d")
    ds.InstanceCreationTime = now.strftime("%H%M%S")
    ds.ContentDate = ds.InstanceCreationDate
    ds.ContentTime = ds.InstanceCreationTime
    return ds


def _code(value: str, meaning: str, scheme: str = SCHEME) -> Dataset:
    c = Dataset()
    c.CodeValue = value
    c.CodingSchemeDesignator = scheme
    c.CodeMeaning = meaning
    return c


def _text(code: str, meaning: str, text: str) -> Dataset:
    item = Dataset()
    item.RelationshipType = "CONTAINS"
    item.ValueType = "TEXT"
    item.ConceptNameCodeSequence = Sequence([_code(code, meaning)])
    item.TextValue = text
    return item


def _num(code: str, meaning: str, value: float, unit: str, unit_meaning: str) -> Dataset:
    measured = Dataset()
    measured.NumericValue = f"{value:.2f}".rstrip("0").rstrip(".") or "0"
    measured.MeasurementUnitsCodeSequence = Sequence([_code(unit, unit_meaning, "UCUM")])
    item = Dataset()
    item.RelationshipType = "CONTAINS"
    item.ValueType = "NUM"
    item.ConceptNameCodeSequence = Sequence([_code(code, meaning)])
    item.MeasuredValueSequence = Sequence([measured])
    return item


def _evidence(ref: SourceRef) -> Sequence:
    image = Dataset()
    image.ReferencedSOPClassUID = ref.sop_class_uid
    image.ReferencedSOPInstanceUID = ref.sop_instance_uid
    series = Dataset()
    series.SeriesInstanceUID = ref.series_uid
    series.ReferencedSOPSequence = Sequence([image])
    study = Dataset()
    study.StudyInstanceUID = ref.study_uid
    study.ReferencedSeriesSequence = Sequence([series])
    return Sequence([study])


_MEASUREMENTS = (
    ("axis_tilt_deg", "DXAQC-AXIS", "Наклон оси позвоночника", "deg", "degree", True),
    ("shaft_tilt_deg", "DXAQC-SHAFT", "Наклон оси бедренной кости", "deg", "degree", True),
    ("field_height_mm", "DXAQC-FH", "Высота поля сканирования", "mm", "millimeter", False),
    ("field_width_mm", "DXAQC-FW", "Ширина поля сканирования", "mm", "millimeter", False),
)


@_quiet
def build_sr(
    ref: SourceRef,
    row: dict[str, Any],
    *,
    series_uid: str,
    instance_number: int,
    corrections: list[dict[str, Any]],
    review: Review | None = None,
    now: datetime | None = None,
) -> Dataset:
    """Enhanced SR with the verdict, violations, explanation, measurements and corrections."""
    now = now or datetime.now()
    # A verified report is a new document: it must not reuse the UID of the unverified one.
    revision = [review.decision, review.reviewed_at.isoformat()] if review else []
    sop_uid = uid(series_uid, ref.sop_instance_uid, "sr", *revision)
    ds = _base(ref, ENHANCED_SR, sop_uid, series_uid, "SR", now)
    ds.SeriesNumber = 9901
    ds.SeriesDescription = "Контроль качества DXA: заключение"
    ds.InstanceNumber = instance_number
    ds.ReferencedPerformedProcedureStepSequence = Sequence()
    ds.PerformedProcedureCodeSequence = Sequence()
    ds.CurrentRequestedProcedureEvidenceSequence = _evidence(ref)
    ds.CompletionFlag = "COMPLETE"
    ds.ValueType = "CONTAINER"
    ds.ContinuityOfContent = "SEPARATE"
    ds.ConceptNameCodeSequence = Sequence([_code("DXAQC-REPORT", "Контроль качества денситометрии")])

    items = [
        _text("DXAQC-REGION", "Анатомическая область", str(row.get("anatomical_region") or "")),
        *([_text("DXAQC-PROJECTION", "Проекция", str(row["projection"]))] if row.get("projection") else []),
        _text(
            "DXAQC-VERDICT", "Вердикт", "Есть нарушение" if row.get("quality_class") == 1 else "Качественное"
        ),
    ]
    prob = row.get("quality_prob")
    if isinstance(prob, (int, float)):
        items.append(_num("DXAQC-PROB", "Вероятность нарушения", 100 * float(prob), "%", "percent"))
    violations = [v for v in str(row.get("violation_type") or "").split(";") if v.strip()]
    items += [_text("DXAQC-VIOLATION", "Нарушение", v.strip()) for v in violations]
    if row.get("explanation"):
        items.append(_text("DXAQC-CONCLUSION", "Заключение", str(row["explanation"])))
    findings = row.get("findings") or {}
    for key, code, meaning, unit, unit_meaning, absolute in _MEASUREMENTS:
        value = findings.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            items.append(_num(code, meaning, abs(value) if absolute else value, unit, unit_meaning))
    items += [_text("DXAQC-CORRECTION", "Рекомендация по коррекции", str(c["action"])) for c in corrections]

    if review is None:
        ds.VerificationFlag = "UNVERIFIED"
    else:
        ds.VerificationFlag = "VERIFIED"
        observer = Dataset()
        observer.VerifyingObserverName = "Специалист"
        observer.VerifyingOrganization = ""
        observer.VerificationDateTime = review.reviewed_at.strftime("%Y%m%d%H%M%S")
        observer.VerifyingObserverIdentificationCodeSequence = Sequence()
        ds.VerifyingObserverSequence = Sequence([observer])
        decision = "подтверждено" if review.decision == "confirmed" else "исправлено"
        verdict = "есть нарушение" if review.quality_class == 1 else "качественное"
        text = f"Решение специалиста: {decision}; {verdict}"
        if review.violations:
            text += f" ({'; '.join(review.violations)})"
        if review.comment:
            text += f". {review.comment}"
        items.append(_text("DXAQC-REVIEW", "Решение специалиста", text))
    ds.ContentSequence = Sequence(items)
    return ds


@_quiet
def build_secondary_capture(
    ref: SourceRef, rgb: np.ndarray, *, series_uid: str, instance_number: int, now: datetime | None = None
) -> Dataset:
    """Derived RGB image (snapshot with the violation overlay) as a new series of the source study."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError(f"expected an 8-bit RGB image, got {rgb.dtype} {rgb.shape}")
    now = now or datetime.now()
    ds = _base(ref, SECONDARY_CAPTURE, uid(series_uid, ref.sop_instance_uid, "sc"), series_uid, "OT", now)
    ds.SeriesNumber = 9902
    ds.SeriesDescription = "Контроль качества DXA: разметка нарушений"
    ds.InstanceNumber = instance_number
    ds.PatientOrientation = ""
    ds.ConversionType = "WSD"
    ds.ImageType = ["DERIVED", "SECONDARY"]
    ds.DerivationDescription = "Снимок с разметкой нарушений качества, изотропная сетка 0,5 мм/пиксель"
    source = Dataset()
    source.ReferencedSOPClassUID = ref.sop_class_uid
    source.ReferencedSOPInstanceUID = ref.sop_instance_uid
    ds.SourceImageSequence = Sequence([source])
    ds.SamplesPerPixel = 3
    ds.PhotometricInterpretation = "RGB"
    ds.PlanarConfiguration = 0
    ds.Rows, ds.Columns = int(rgb.shape[0]), int(rgb.shape[1])
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = np.ascontiguousarray(rgb).tobytes()
    return ds


def composite(preview: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    """Grayscale preview (H, W) + RGBA overlay (H, W, 4) -> RGB (H, W, 3), alpha-blended."""
    gray = np.repeat(preview[..., None].astype(np.float32), 3, axis=2)
    alpha = overlay[..., 3:4].astype(np.float32) / 255.0
    out = gray * (1 - alpha) + overlay[..., :3].astype(np.float32) * alpha
    return np.clip(out.round(), 0, 255).astype(np.uint8)


def write(ds: Dataset, path: Path) -> Path:
    """Save as a DICOM file (Part 10). UIDs of the source study are kept as they are, even when
    they are not strictly conformant (leading zeros): the outputs must file into that study."""
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = ds.SOPClassUID
    meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.ImplementationClassUID = PYDICOM_IMPLEMENTATION_UID
    ds.file_meta = meta
    path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds.save_as(path, enforce_file_format=True)
    return path


def output_paths(out_dir: Path, index: int) -> tuple[Path, Path]:
    """File names of the SR and the Secondary Capture of the ``index``-th image."""
    return out_dir / f"{index:05d}_sr.dcm", out_dir / f"{index:05d}_sc.dcm"


def write_sr(
    ref: SourceRef,
    row: dict[str, Any],
    corrections: list[dict[str, Any]],
    out_dir: Path,
    *,
    series_key: str,
    index: int,
    review: Review | None = None,
) -> Path:
    sr = build_sr(
        ref,
        row,
        series_uid=uid(series_key, ref.study_uid, "sr"),
        instance_number=index + 1,
        corrections=corrections,
        review=review,
    )
    return write(sr, output_paths(out_dir, index)[0])


def write_outputs(
    ref: SourceRef,
    row: dict[str, Any],
    corrections: list[dict[str, Any]],
    preview: np.ndarray,
    overlay: np.ndarray,
    out_dir: Path,
    *,
    series_key: str,
    index: int,
) -> tuple[Path, Path]:
    """SR + Secondary Capture of one processed image; one SR and one SC series per study."""
    sr_path = write_sr(ref, row, corrections, out_dir, series_key=series_key, index=index)
    sc = build_secondary_capture(
        ref,
        composite(preview, overlay),
        series_uid=uid(series_key, ref.study_uid, "sc"),
        instance_number=index + 1,
    )
    return sr_path, write(sc, output_paths(out_dir, index)[1])
