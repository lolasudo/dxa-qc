"""Team-verified region/side annotations (``data/region_annotations.csv``).

The organizer labels quality per study and per region, but not which file is which region.
This table, produced by visual review of every unique image, closes that gap. Rows are keyed
by the pixel digest so they cover all byte-identical copies of an image.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from dxa_qc.models.region.classifier import RegionClass

_COLUMNS = ("pixel_sha256_16", "study_id", "example_path", "region", "side", "note")


class AnnotationsFormatError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RegionAnnotation:
    pixel_digest: str
    study_id: str
    example_path: PurePosixPath  # relative to the studies directory
    label: RegionClass
    note: str


def _label(region: str, side: str) -> RegionClass:
    if region == "spine" and not side:
        return RegionClass.SPINE
    if region == "hip" and side == "left":
        return RegionClass.HIP_LEFT
    if region == "hip" and side == "right":
        return RegionClass.HIP_RIGHT
    raise AnnotationsFormatError(f"invalid region/side combination: {region!r}/{side!r}")


def load_region_annotations(path: Path) -> dict[str, RegionAnnotation]:
    with Path(path).open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if tuple(reader.fieldnames or ()) != _COLUMNS:
            raise AnnotationsFormatError(f"unexpected columns {reader.fieldnames}, expected {_COLUMNS}")
        result: dict[str, RegionAnnotation] = {}
        for line, row in enumerate(reader, start=2):
            rel = PurePosixPath(row["example_path"])
            if rel.is_absolute() or ".." in rel.parts:
                raise AnnotationsFormatError(f"line {line}: example_path must be relative: {rel}")
            digest = row["pixel_sha256_16"]
            if digest in result:
                raise AnnotationsFormatError(f"line {line}: duplicate pixel digest {digest}")
            try:
                label = _label(row["region"], row["side"])
            except AnnotationsFormatError as exc:
                raise AnnotationsFormatError(f"line {line}: {exc}") from None
            result[digest] = RegionAnnotation(digest, row["study_id"], rel, label, row["note"])
    if not result:
        raise AnnotationsFormatError(f"no annotations in {path}")
    return result
