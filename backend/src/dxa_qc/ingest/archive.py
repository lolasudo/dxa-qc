"""Safe ZIP extraction for uploaded study archives.

Protects against: zip-slip (``../`` / absolute paths / drive letters), symlink members,
zip bombs (member count, total size, compression ratio, and sizes lying in the header),
and encrypted members. Anything suspicious aborts the whole archive: a partially
extracted malicious archive is worse than a clear error.
"""

from __future__ import annotations

import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath

from dxa_qc.config.settings import IngestLimits

_UTF8_FLAG = 0x800
_ENCRYPTED_FLAG = 0x1
_CHUNK = 1 << 20
# Ratio check is meaningless for tiny members (a 4 KiB run of zeros compresses 1000x legitimately).
_RATIO_MIN_BYTES = 1 << 20


class UnsafeArchiveError(Exception):
    """Archive rejected for security reasons or because it is not a readable ZIP."""


def decode_member_name(info: zipfile.ZipInfo) -> str:
    """Recover Cyrillic names from archives made by Windows tools.

    Without the UTF-8 flag, :mod:`zipfile` decodes names as cp437, but Russian Windows
    archivers (including the organizer's) actually write cp866. cp437 round-trips bytes
    losslessly, so re-decoding as cp866 is safe for ASCII names too.
    """
    if info.flag_bits & _UTF8_FLAG:
        return info.filename
    return info.filename.encode("cp437").decode("cp866")


def _safe_relative_path(name: str) -> PurePosixPath | None:
    """Validated relative path of a member, ``None`` for directory entries."""
    normalized = name.replace("\\", "/")
    if normalized.endswith("/"):
        return None
    path = PurePosixPath(normalized)
    if path.is_absolute() or normalized.startswith("/"):
        raise UnsafeArchiveError(f"absolute path in archive: {name!r}")
    if any(part == ".." for part in path.parts):
        raise UnsafeArchiveError(f"path traversal in archive: {name!r}")
    if path.parts and ":" in path.parts[0]:
        raise UnsafeArchiveError(f"drive-qualified path in archive: {name!r}")
    if "\x00" in normalized:
        raise UnsafeArchiveError(f"NUL byte in archive member name: {name!r}")
    return path


def _is_symlink(info: zipfile.ZipInfo) -> bool:
    return stat.S_ISLNK(info.external_attr >> 16)


def _check_metadata(infos: list[zipfile.ZipInfo], limits: IngestLimits) -> int:
    """Validate member metadata; returns the declared total uncompressed size."""
    if len(infos) > limits.max_archive_members:
        raise UnsafeArchiveError(f"too many archive members ({len(infos)} > {limits.max_archive_members})")
    total = 0
    for info in infos:
        if info.flag_bits & _ENCRYPTED_FLAG:
            raise UnsafeArchiveError(f"encrypted member: {info.filename!r}")
        if _is_symlink(info):
            raise UnsafeArchiveError(f"symlink member: {info.filename!r}")
        if info.file_size > limits.max_file_bytes:
            raise UnsafeArchiveError(f"member too large: {info.filename!r} ({info.file_size} bytes)")
        if info.file_size >= _RATIO_MIN_BYTES:
            ratio = info.file_size / max(info.compress_size, 1)
            if ratio > limits.max_compression_ratio:
                raise UnsafeArchiveError(f"suspicious compression ratio {ratio:.0f} for {info.filename!r}")
        total += info.file_size
    if total > limits.max_archive_uncompressed_bytes:
        raise UnsafeArchiveError(
            f"archive expands to {total} bytes, limit {limits.max_archive_uncompressed_bytes}"
        )
    return total


def _check_free_space(destination: Path, needed: int, keep_free: int) -> None:
    # Declared sizes are enforced during copying (_copy_bounded), so they are a safe upper bound.
    free = shutil.disk_usage(destination).free
    if free - needed < keep_free:
        raise UnsafeArchiveError(
            f"not enough disk space to extract: needs {needed} bytes, {free} free, {keep_free} must stay free"
        )


def _copy_bounded(src, dst, declared_size: int, name: str) -> None:
    # Header sizes are attacker-controlled; count real bytes and stop at the declared size.
    written = 0
    while chunk := src.read(_CHUNK):
        written += len(chunk)
        if written > declared_size:
            raise UnsafeArchiveError(f"member {name!r} is larger than declared in the header")
        dst.write(chunk)


def safe_extract(archive: Path, destination: Path, limits: IngestLimits) -> list[Path]:
    """Extract ``archive`` into ``destination`` and return the extracted file paths.

    ``destination`` must be empty or absent; on any error it is removed entirely.
    """
    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f"destination must be empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()

    try:
        with zipfile.ZipFile(archive) as zf:
            infos = zf.infolist()
            total = _check_metadata(infos, limits)
            _check_free_space(destination, total, limits.min_free_disk_bytes)
            extracted: list[Path] = []
            for info in infos:
                rel = _safe_relative_path(decode_member_name(info))
                if rel is None:
                    continue
                target = (root / Path(*rel.parts)).resolve()
                # Final guard after resolution: covers anything the lexical checks missed.
                if not target.is_relative_to(root):
                    raise UnsafeArchiveError(f"member escapes destination: {info.filename!r}")
                if target.exists():
                    raise UnsafeArchiveError(f"duplicate member path: {info.filename!r}")
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, target.open("xb") as dst:
                    _copy_bounded(src, dst, info.file_size, info.filename)
                extracted.append(target)
            return extracted
    except UnsafeArchiveError:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    except (zipfile.BadZipFile, zipfile.LargeZipFile, NotImplementedError, EOFError, OSError) as exc:
        shutil.rmtree(destination, ignore_errors=True)
        raise UnsafeArchiveError(f"cannot read archive: {exc}") from exc
