"""DICOM reading with validation.

Every failure mode surfaces as :class:`DicomReadError` with a human-readable reason, so the
batch pipeline can record ``processing_status = Failure`` instead of crashing.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import FileDataset
from pydicom.errors import InvalidDicomError
from pydicom.pixels import apply_modality_lut

from dxa_qc.config.settings import PixelSpacingConfig

# The scanner writes non-conformant UIDs and over-long LO values. They are harmless for reading,
# but default validation logs a warning per element, flooding batch logs; structural errors
# still raise regardless of this setting.
pydicom.config.settings.reading_validation_mode = pydicom.config.IGNORE

_PREAMBLE_LEN = 128
_MAGIC = b"DICM"
# Guards against decompressing absurd images; real DXA frames are ~300x300.
# Callers pass the configured limits; these defaults keep any caller safe that forgets to.
_MIN_SIDE = 16
DEFAULT_MAX_SIDE = 4096
DEFAULT_PHYSICAL_RANGE_MM = (10.0, 2500.0)


class DicomReadError(Exception):
    """The file cannot be used as a single-frame grayscale DICOM image."""


class SpacingSource(StrEnum):
    PIXEL_SPACING = "PixelSpacing"
    IMAGER_PIXEL_SPACING = "ImagerPixelSpacing"
    FALLBACK = "config_fallback"


@dataclass(frozen=True, slots=True)
class PixelSpacing:
    row_mm: float
    column_mm: float
    source: SpacingSource


@dataclass(frozen=True, slots=True)
class DicomImage:
    path: Path
    study_uid: str
    image_uid: str
    series_uid: str
    instance_number: int | None
    # float32 in [0, 1], 2-D, bright = dense regardless of the original photometric interpretation.
    pixels: np.ndarray
    spacing: PixelSpacing

    @property
    def shape(self) -> tuple[int, int]:
        return self.pixels.shape  # type: ignore[return-value]


def has_dicom_preamble(path: Path) -> bool:
    """Cheap content-based check; file extensions are not trusted (closed test naming is unknown)."""
    try:
        with path.open("rb") as fh:
            head = fh.read(_PREAMBLE_LEN + len(_MAGIC))
    except OSError:
        return False
    return head[_PREAMBLE_LEN:] == _MAGIC


def pixel_digest(path: Path) -> str:
    """Short SHA-256 of the raw PixelData bytes.

    Identifies byte-identical images regardless of headers: the dataset repeats the same scan
    under several instance numbers, and duplicates must be neither double-weighted in training
    nor split across CV folds.
    """
    try:
        ds = pydicom.dcmread(path, force=False)
        data = ds.PixelData
    except (InvalidDicomError, AttributeError, OSError, EOFError, ValueError, KeyError, TypeError) as exc:
        raise DicomReadError(f"cannot read pixel data of {Path(path).name}: {_reason(exc, path)}") from exc
    return hashlib.sha256(data).hexdigest()[:16]


def _uid(ds: FileDataset, keyword: str) -> str:
    value = ds.get(keyword)
    return str(value).strip() if value is not None else ""


def _instance_number(ds: FileDataset) -> int | None:
    try:
        value = ds.get("InstanceNumber")
        return int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _parse_spacing_tag(value: object) -> tuple[float, float] | None:
    try:
        row, col = (float(v) for v in value)  # type: ignore[union-attr]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) and 0 < v <= 10 for v in (row, col)):
        return None
    return row, col


def resolve_pixel_spacing(ds: FileDataset, fallback: PixelSpacingConfig) -> PixelSpacing:
    """Tag first, config fallback only when the tag is absent or invalid."""
    for keyword, source in (
        ("PixelSpacing", SpacingSource.PIXEL_SPACING),
        ("ImagerPixelSpacing", SpacingSource.IMAGER_PIXEL_SPACING),
    ):
        if keyword in ds:
            parsed = _parse_spacing_tag(ds.get(keyword))
            if parsed is not None:
                return PixelSpacing(*parsed, source=source)
    return PixelSpacing(fallback.row_mm, fallback.column_mm, SpacingSource.FALLBACK)


def _validate_geometry(ds: FileDataset, max_side: int) -> None:
    samples = int(ds.get("SamplesPerPixel", 1) or 1)
    if samples != 1:
        raise DicomReadError(f"expected grayscale image, got SamplesPerPixel={samples}")
    frames = int(ds.get("NumberOfFrames", 1) or 1)
    if frames != 1:
        raise DicomReadError(f"expected single-frame image, got NumberOfFrames={frames}")
    rows, cols = ds.get("Rows"), ds.get("Columns")
    if rows is None or cols is None:
        raise DicomReadError("Rows/Columns tags are missing")
    if not (_MIN_SIDE <= int(rows) <= max_side and _MIN_SIDE <= int(cols) <= max_side):
        raise DicomReadError(f"unsupported image size {rows}x{cols}")
    photometric = str(ds.get("PhotometricInterpretation", "")).upper()
    if photometric not in ("MONOCHROME1", "MONOCHROME2"):
        raise DicomReadError(f"unsupported PhotometricInterpretation {photometric!r}")


def _normalize(ds: FileDataset, raw: np.ndarray) -> np.ndarray:
    has_rescale = "RescaleSlope" in ds or "RescaleIntercept" in ds or "ModalityLUTSequence" in ds
    data = apply_modality_lut(raw, ds).astype(np.float32) if has_rescale else raw.astype(np.float32)
    if not np.isfinite(data).all():
        raise DicomReadError("pixel data contains non-finite values")

    if not has_rescale and int(ds.get("PixelRepresentation", 0)) == 0:
        # Unsigned stored values: scale by the nominal range to keep absolute brightness comparable
        # across images (min-max stretching would hide e.g. how bright a metal artefact is).
        bits = int(ds.get("BitsStored", ds.get("BitsAllocated", 8)))
        data /= float((1 << bits) - 1)
    else:
        lo, hi = float(data.min()), float(data.max())
        data = (data - lo) / (hi - lo) if hi > lo else np.zeros_like(data)

    data = np.clip(data, 0.0, 1.0)
    if str(ds.PhotometricInterpretation).upper() == "MONOCHROME1":
        data = 1.0 - data
    return data


def _validate_physical_size(
    rows: int, cols: int, spacing: PixelSpacing, allowed: tuple[float, float]
) -> None:
    """The field must have a plausible size in mm: this also bounds the isotropic resampling
    downstream, so a forged PixelSpacing cannot inflate the image to billions of pixels."""
    lo, hi = allowed
    height, width = rows * spacing.row_mm, cols * spacing.column_mm
    if not (lo <= height <= hi and lo <= width <= hi):
        raise DicomReadError(
            f"implausible physical image size {height:.0f} x {width:.0f} mm "
            f"(pixel spacing {spacing.row_mm} x {spacing.column_mm} mm, {spacing.source.value})"
        )


def _reason(exc: BaseException, path: Path | str) -> str:
    """Error text safe to show to clients: no server-side directories (they reveal the job layout)."""
    if isinstance(exc, OSError) and exc.strerror:
        return exc.strerror
    text = str(exc)
    full = str(path)
    return text.replace(full, Path(full).name) if full else text


def read_dicom(
    path: Path,
    spacing_fallback: PixelSpacingConfig,
    max_file_bytes: int,
    *,
    max_side: int = DEFAULT_MAX_SIDE,
    physical_range_mm: tuple[float, float] = DEFAULT_PHYSICAL_RANGE_MM,
) -> DicomImage:
    """Read and validate one DICOM image. Raises :class:`DicomReadError` on any problem."""
    path = Path(path)
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise DicomReadError(f"cannot access file: {exc.strerror or exc}") from exc
    if size > max_file_bytes:
        raise DicomReadError(f"file is too large ({size} bytes > {max_file_bytes})")

    try:
        # force=False: refuse files without a DICOM header rather than guessing.
        ds = pydicom.dcmread(path, force=False)
    except InvalidDicomError as exc:
        raise DicomReadError(f"not a valid DICOM file: {_reason(exc, path)}") from exc
    except (OSError, EOFError, ValueError, KeyError, TypeError) as exc:
        raise DicomReadError(f"corrupted DICOM file: {_reason(exc, path)}") from exc

    if "PixelData" not in ds:
        raise DicomReadError("DICOM has no PixelData")
    _validate_geometry(ds, max_side)
    # Checked before decoding: rejecting costs nothing, decoding a hostile frame does.
    spacing = resolve_pixel_spacing(ds, spacing_fallback)
    _validate_physical_size(int(ds.Rows), int(ds.Columns), spacing, physical_range_mm)

    try:
        raw = ds.pixel_array
    except Exception as exc:  # decoder errors come from many plugins with unrelated types
        raise DicomReadError(f"cannot decode pixel data: {exc}") from exc
    if raw.ndim != 2:
        raise DicomReadError(f"expected 2-D pixel array, got shape {raw.shape}")

    return DicomImage(
        path=path,
        study_uid=_uid(ds, "StudyInstanceUID"),
        image_uid=_uid(ds, "SOPInstanceUID"),
        series_uid=_uid(ds, "SeriesInstanceUID"),
        instance_number=_instance_number(ds),
        pixels=_normalize(ds, raw),
        spacing=spacing,
    )
