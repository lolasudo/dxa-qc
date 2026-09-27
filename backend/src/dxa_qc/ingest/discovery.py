"""Turn a CLI/API input (file, folder or ZIP) into an ordered list of images to process."""

from __future__ import annotations

import os
import zipfile
from dataclasses import dataclass
from pathlib import Path

from dxa_qc.config.settings import IngestLimits
from dxa_qc.dicom_io.reader import has_dicom_preamble
from dxa_qc.ingest.archive import safe_extract

# DICOMDIR is a media index, not an image; OS/archiver metadata is never a study.
_SKIP_NAMES = frozenset({"DICOMDIR", "Thumbs.db", ".DS_Store", "desktop.ini"})
_SKIP_DIRS = frozenset({"__MACOSX"})


@dataclass(frozen=True, slots=True)
class InputItem:
    path: Path
    # Path shown in the report: relative to the user's input, never an internal temp dir.
    display_path: str


def _is_candidate(path: Path) -> bool:
    if path.name in _SKIP_NAMES or path.name.startswith("."):
        return False
    # A ".dcm" that fails the preamble check is still returned: it must become a Failure row,
    # not silently disappear from the report.
    return path.suffix.lower() == ".dcm" or has_dicom_preamble(path)


def _walk(root: Path) -> list[Path]:
    found: list[Path] = []
    # followlinks=False + skipping file symlinks: an input folder must not be able to pull in
    # files from elsewhere on the host.
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS and not d.startswith("."))
        for name in sorted(filenames):
            path = Path(dirpath) / name
            if not path.is_symlink() and path.is_file() and _is_candidate(path):
                found.append(path)
    return found


def _display(path: Path, root: Path, prefix: str) -> str:
    rel = path.relative_to(root).as_posix()
    return f"{prefix}/{rel}" if prefix else rel


def collect_inputs(source: Path, workdir: Path, limits: IngestLimits) -> list[InputItem]:
    """Resolve ``source`` into image files, extracting ZIPs into ``workdir``.

    Output order is deterministic (sorted paths) so repeated runs produce identical reports.
    """
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"input does not exist: {source}")

    if source.is_dir():
        root = source.resolve()
        return [InputItem(p, _display(p, root, source.name)) for p in _walk(root)]

    if zipfile.is_zipfile(source):
        extract_root = Path(workdir) / "extracted"
        safe_extract(source, extract_root, limits)
        root = extract_root.resolve()
        return [InputItem(p, _display(p, root, source.name)) for p in _walk(root)]

    return [InputItem(source.resolve(), source.name)]
