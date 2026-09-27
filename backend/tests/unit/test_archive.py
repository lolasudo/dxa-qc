from __future__ import annotations

import io
import stat
import zipfile
from pathlib import Path

import pytest

from dxa_qc.config.settings import IngestLimits
from dxa_qc.ingest.archive import UnsafeArchiveError, _copy_bounded, decode_member_name, safe_extract

LIMITS = IngestLimits(
    max_file_bytes=10 * 1024 * 1024,
    max_archive_members=50,
    max_archive_uncompressed_bytes=20 * 1024 * 1024,
    max_compression_ratio=100,
)


def build_zip(path: Path, members: dict[str, bytes], compression=zipfile.ZIP_DEFLATED) -> Path:
    with zipfile.ZipFile(path, "w", compression) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return path


def test_extracts_nested_files(tmp_path):
    arc = build_zip(tmp_path / "a.zip", {"study/1.dcm": b"x", "study/sub/2.dcm": b"y", "study/": b""})
    out = safe_extract(arc, tmp_path / "out", LIMITS)
    rels = sorted(p.relative_to((tmp_path / "out").resolve()).as_posix() for p in out)
    assert rels == ["study/1.dcm", "study/sub/2.dcm"]


@pytest.mark.parametrize(
    "evil",
    ["../../etc/passwd", "a/../../evil.dcm", "/etc/passwd", "C:/Windows/evil.dcm", "..\\..\\evil.dcm"],
)
def test_zip_slip_rejected_and_nothing_left_behind(tmp_path, evil):
    arc = build_zip(tmp_path / "a.zip", {"ok.dcm": b"x", evil: b"pwned"})
    dest = tmp_path / "work" / "out"
    with pytest.raises(UnsafeArchiveError):
        safe_extract(arc, dest, LIMITS)
    assert not dest.exists()
    assert not any(p.name in {"passwd", "evil.dcm"} for p in tmp_path.rglob("*"))


def test_symlink_member_rejected(tmp_path):
    arc = tmp_path / "a.zip"
    with zipfile.ZipFile(arc, "w") as zf:
        info = zipfile.ZipInfo("link.dcm")
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        zf.writestr(info, "/etc/passwd")
    with pytest.raises(UnsafeArchiveError, match="symlink"):
        safe_extract(arc, tmp_path / "out", LIMITS)


def test_too_many_members(tmp_path):
    arc = build_zip(tmp_path / "a.zip", {f"{i}.dcm": b"x" for i in range(LIMITS.max_archive_members + 1)})
    with pytest.raises(UnsafeArchiveError, match="too many"):
        safe_extract(arc, tmp_path / "out", LIMITS)


def test_compression_bomb_ratio(tmp_path):
    arc = build_zip(tmp_path / "a.zip", {"bomb.dcm": b"\0" * (5 * 1024 * 1024)})
    with pytest.raises(UnsafeArchiveError, match="compression ratio"):
        safe_extract(arc, tmp_path / "out", LIMITS)


def test_total_size_limit(tmp_path):
    limits = LIMITS.model_copy(update={"max_archive_uncompressed_bytes": 1000})
    arc = build_zip(tmp_path / "a.zip", {"a.dcm": b"x" * 600, "b.dcm": b"x" * 600})
    with pytest.raises(UnsafeArchiveError, match="expands"):
        safe_extract(arc, tmp_path / "out", limits)


def test_bounded_copy_stops_at_declared_size():
    # Defense in depth: zipfile already truncates at file_size, but the extractor must not rely on it.
    with pytest.raises(UnsafeArchiveError, match="larger than declared"):
        _copy_bounded(io.BytesIO(b"A" * 5000), io.BytesIO(), declared_size=100, name="a.dcm")


def test_bounded_copy_exact_size_ok():
    dst = io.BytesIO()
    _copy_bounded(io.BytesIO(b"A" * 100), dst, declared_size=100, name="a.dcm")
    assert dst.getvalue() == b"A" * 100


def test_not_a_zip(tmp_path):
    bad = tmp_path / "a.zip"
    bad.write_bytes(b"not a zip at all")
    with pytest.raises(UnsafeArchiveError, match="cannot read"):
        safe_extract(bad, tmp_path / "out", LIMITS)


def test_nonempty_destination_refused(tmp_path):
    arc = build_zip(tmp_path / "a.zip", {"a.dcm": b"x"})
    dest = tmp_path / "out"
    dest.mkdir()
    (dest / "keep.txt").write_text("important")
    with pytest.raises(ValueError, match="empty"):
        safe_extract(arc, dest, LIMITS)
    assert (dest / "keep.txt").exists()


def test_cp866_names_decoded(tmp_path):
    # Emulate a Windows archiver: cp866 bytes, UTF-8 flag not set.
    info = zipfile.ZipInfo("x")
    info.filename = "Для теста/CR000000_ПОП.dcm".encode("cp866").decode("cp437")
    info.flag_bits = 0
    assert decode_member_name(info) == "Для теста/CR000000_ПОП.dcm"


def test_utf8_flagged_names_untouched(tmp_path):
    arc = build_zip(tmp_path / "a.zip", {"Датасет/файл.dcm": b"x"})
    out = safe_extract(arc, tmp_path / "out", LIMITS)
    assert out[0].name == "файл.dcm"


def test_extraction_refused_when_disk_would_fill(tmp_path, monkeypatch):
    import shutil as _shutil
    from collections import namedtuple

    from dxa_qc.config import load_settings
    from tests.conftest import BACKEND_ROOT

    limits = load_settings(BACKEND_ROOT / "configs").ingest.model_copy(update={"min_free_disk_bytes": 1000})
    arc = tmp_path / "a.zip"
    with zipfile.ZipFile(arc, "w") as zf:
        zf.writestr("x.dcm", b"0" * 5000)
    Usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr(_shutil, "disk_usage", lambda _: Usage(10**6, 10**6, 5500))
    with pytest.raises(UnsafeArchiveError, match="not enough disk space"):
        safe_extract(arc, tmp_path / "out", limits)
    assert not (tmp_path / "out").exists()
