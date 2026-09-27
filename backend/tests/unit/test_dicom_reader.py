from __future__ import annotations

import numpy as np
import pytest

from dxa_qc.dicom_io import DicomReadError, SpacingSource, has_dicom_preamble, read_dicom
from dxa_qc.dicom_io.reader import PixelSpacing
from tests.factories import make_dicom

MAX_BYTES = 64 * 1024 * 1024


@pytest.fixture
def read(settings):
    return lambda p, max_bytes=MAX_BYTES: read_dicom(p, settings.pixel_spacing, max_bytes)


def test_reads_uids_and_normalizes_8bit(tmp_path, read):
    px = np.zeros((32, 40), np.uint8)
    px[0, 0], px[1, 1] = 255, 51
    path = make_dicom(tmp_path / "a.dcm", px, study_uid="1.2.3", sop_uid="1.2.3.4")
    img = read(path)
    assert (img.study_uid, img.image_uid) == ("1.2.3", "1.2.3.4")
    assert img.shape == (32, 40)
    assert img.pixels.dtype == np.float32
    # Absolute scaling by bit depth, not min-max: 51/255 stays 0.2.
    assert img.pixels[0, 0] == pytest.approx(1.0)
    assert img.pixels[1, 1] == pytest.approx(0.2)


def test_16bit_uses_bits_stored(tmp_path, read):
    px = np.full((32, 32), 4095, np.uint16)
    path = make_dicom(tmp_path / "a.dcm", px, extra={"BitsStored": 12, "HighBit": 11})
    assert read(path).pixels.max() == pytest.approx(1.0)


def test_monochrome1_is_inverted(tmp_path, read):
    px = np.zeros((32, 32), np.uint8)
    img = read(make_dicom(tmp_path / "a.dcm", px, photometric="MONOCHROME1"))
    assert img.pixels.min() == pytest.approx(1.0)


def test_rescale_slope_triggers_minmax(tmp_path, read):
    px = np.tile(np.arange(32, dtype=np.uint8), (32, 1))
    img = read(make_dicom(tmp_path / "a.dcm", px, extra={"RescaleSlope": 2, "RescaleIntercept": -100}))
    assert img.pixels.min() == pytest.approx(0.0)
    assert img.pixels.max() == pytest.approx(1.0)


def test_spacing_fallback_when_tag_missing(tmp_path, read):
    sp = read(make_dicom(tmp_path / "a.dcm")).spacing
    assert (sp.row_mm, sp.column_mm, sp.source) == (1.05, 0.6, SpacingSource.FALLBACK)


def test_spacing_tag_wins_over_fallback(tmp_path, read):
    sp = read(make_dicom(tmp_path / "a.dcm", extra={"PixelSpacing": [0.5, 0.25]})).spacing
    assert (sp.row_mm, sp.column_mm, sp.source) == (0.5, 0.25, SpacingSource.PIXEL_SPACING)


def test_imager_pixel_spacing_used_second(tmp_path, read):
    sp = read(make_dicom(tmp_path / "a.dcm", extra={"ImagerPixelSpacing": [0.7, 0.7]})).spacing
    assert sp.source == SpacingSource.IMAGER_PIXEL_SPACING


def test_invalid_spacing_tag_falls_back(tmp_path, read):
    sp = read(make_dicom(tmp_path / "a.dcm", extra={"PixelSpacing": [0, 0.6]})).spacing
    assert sp.source == SpacingSource.FALLBACK


def test_missing_uids_do_not_fail(tmp_path, read):
    path = make_dicom(tmp_path / "a.dcm")
    import pydicom

    ds = pydicom.dcmread(path)
    del ds.StudyInstanceUID
    ds.save_as(path)
    assert read(path).study_uid == ""


# --- failure modes: every one must be a DicomReadError, never a raw exception ---


def test_non_dicom_file(tmp_path, read):
    path = tmp_path / "x.dcm"
    path.write_bytes(b"hello world" * 100)
    with pytest.raises(DicomReadError, match="not a valid DICOM"):
        read(path)


def test_truncated_file(tmp_path, read):
    path = make_dicom(tmp_path / "a.dcm")
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])
    with pytest.raises(DicomReadError):
        read(path)


def test_empty_file(tmp_path, read):
    path = tmp_path / "empty.dcm"
    path.write_bytes(b"")
    with pytest.raises(DicomReadError):
        read(path)


def test_missing_file(tmp_path, read):
    with pytest.raises(DicomReadError, match="cannot access"):
        read(tmp_path / "nope.dcm")


def test_size_limit(tmp_path, read):
    path = make_dicom(tmp_path / "a.dcm")
    with pytest.raises(DicomReadError, match="too large"):
        read(path, max_bytes=100)


def test_no_pixel_data(tmp_path, read):
    with pytest.raises(DicomReadError, match="no PixelData"):
        read(make_dicom(tmp_path / "a.dcm", with_pixels=False))


def test_rgb_rejected(tmp_path, read):
    path = make_dicom(tmp_path / "a.dcm", extra={"SamplesPerPixel": 3})
    with pytest.raises(DicomReadError, match="grayscale"):
        read(path)


def test_tiny_image_rejected(tmp_path, read):
    with pytest.raises(DicomReadError, match="unsupported image size"):
        read(make_dicom(tmp_path / "a.dcm", np.zeros((4, 4), np.uint8)))


def test_unsupported_photometric(tmp_path, read):
    with pytest.raises(DicomReadError, match="PhotometricInterpretation"):
        read(make_dicom(tmp_path / "a.dcm", photometric="PALETTE COLOR"))


def test_pixel_data_length_mismatch(tmp_path, read):
    path = make_dicom(tmp_path / "a.dcm", extra={"Rows": 200})
    with pytest.raises(DicomReadError, match="decode"):
        read(path)


def test_preamble_detection(tmp_path):
    assert has_dicom_preamble(make_dicom(tmp_path / "no_extension"))
    junk = tmp_path / "junk.dcm"
    junk.write_bytes(b"x" * 10)
    assert not has_dicom_preamble(junk)
    assert not has_dicom_preamble(tmp_path / "missing")


# ---- resource-exhaustion guards ---------------------------------------------------------------


def test_forged_pixel_spacing_is_rejected_before_decoding(tmp_path, settings, monkeypatch):
    """PixelSpacing=10 mm on a 400 px frame claims a 4 m field: reject without decoding pixels."""
    path = make_dicom(tmp_path / "x.dcm", np.zeros((400, 400), np.uint8), extra={"PixelSpacing": [10, 10]})
    import pydicom.dataset

    def boom(self):  # decoding must not even be attempted
        raise AssertionError("pixel data decoded for a rejected file")

    monkeypatch.setattr(pydicom.dataset.Dataset, "pixel_array", property(boom))
    with pytest.raises(DicomReadError, match="implausible physical image size"):
        read_dicom(path, settings.pixel_spacing, MAX_BYTES)


def test_oversized_frame_is_rejected(tmp_path, settings):
    path = make_dicom(tmp_path / "x.dcm", np.zeros((300, 300), np.uint8))
    with pytest.raises(DicomReadError, match="unsupported image size"):
        read_dicom(path, settings.pixel_spacing, MAX_BYTES, max_side=256)


def test_isotropic_grid_is_bounded():
    from dxa_qc.models.quality.geometry import to_isotropic

    with pytest.raises(ValueError, match="exceeds"):
        to_isotropic(np.zeros((500, 500), np.float32), PixelSpacing(10.0, 10.0, SpacingSource.PIXEL_SPACING))


def test_read_errors_do_not_reveal_server_directories(tmp_path, settings):
    from dxa_qc.dicom_io.reader import DicomReadError, pixel_digest, read_dicom

    secret_dir = tmp_path / "jobs" / "0123456789abcdef"
    secret_dir.mkdir(parents=True)
    bad = secret_dir / "broken.dcm"
    bad.write_bytes(b"\x00" * 300)
    for call in (
        lambda: read_dicom(bad, settings.pixel_spacing, 10**6),
        lambda: read_dicom(secret_dir / "missing.dcm", settings.pixel_spacing, 10**6),
        lambda: pixel_digest(bad),
    ):
        with pytest.raises(DicomReadError) as err:
            call()
        assert "0123456789abcdef" not in str(err.value) and str(tmp_path) not in str(err.value)
