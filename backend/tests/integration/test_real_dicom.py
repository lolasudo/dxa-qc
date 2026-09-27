from __future__ import annotations

import pytest

from dxa_qc.dicom_io import SpacingSource, read_dicom
from tests.conftest import STUDIES_DIR, TEST_SAMPLE_DIR, require

pytestmark = pytest.mark.dataset


def test_organizer_sample_files_read(settings):
    files = sorted(require(TEST_SAMPLE_DIR).glob("*.dcm"))
    assert len(files) == 3
    for f in files:
        img = read_dicom(f, settings.pixel_spacing, settings.ingest.max_file_bytes)
        assert img.study_uid and img.image_uid
        assert img.spacing.source == SpacingSource.FALLBACK
        assert 0.0 <= img.pixels.min() < img.pixels.max() <= 1.0


def test_every_training_file_reads(settings):
    files = sorted(require(STUDIES_DIR).rglob("*.dcm"))
    assert len(files) == 499
    for f in files:
        read_dicom(f, settings.pixel_spacing, settings.ingest.max_file_bytes)
