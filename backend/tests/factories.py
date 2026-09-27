"""Builders for synthetic DICOM files, so unit tests don't depend on the real dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import (
    PYDICOM_IMPLEMENTATION_UID,
    ComputedRadiographyImageStorage,
    ExplicitVRLittleEndian,
    generate_uid,
)


def make_dicom(
    path: Path,
    pixels: np.ndarray | None = None,
    *,
    photometric: str = "MONOCHROME2",
    study_uid: str | None = None,
    sop_uid: str | None = None,
    extra: dict[str, Any] | None = None,
    with_pixels: bool = True,
) -> Path:
    if pixels is None:
        pixels = np.tile(np.arange(64, dtype=np.uint8) * 4, (48, 1))
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = ComputedRadiographyImageStorage
    meta.MediaStorageSOPInstanceUID = sop_uid or generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.ImplementationClassUID = PYDICOM_IMPLEMENTATION_UID

    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID = ComputedRadiographyImageStorage
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = study_uid or generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.InstanceNumber = 1
    ds.Modality = "CR"
    ds.Rows, ds.Columns = pixels.shape[:2]
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = photometric
    bits = 8 if pixels.dtype == np.uint8 else 16
    ds.BitsAllocated = bits
    ds.BitsStored = bits
    ds.HighBit = bits - 1
    ds.PixelRepresentation = 0
    if with_pixels:
        ds.PixelData = pixels.tobytes()
    for key, value in (extra or {}).items():
        setattr(ds, key, value)

    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(path, enforce_file_format=True)
    return path
