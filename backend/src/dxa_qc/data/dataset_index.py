"""Join files, verified region annotations and expert labels into one training index.

One row per file. ``is_duplicate`` marks byte-identical repeats of an earlier file in the same
study: training should use only canonical rows, while all rows share the study as CV group.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from dxa_qc.data.annotations import RegionAnnotation
from dxa_qc.data.labels import ClinicalTag, RegionLabel, StudyLabels
from dxa_qc.dicom_io.reader import pixel_digest
from dxa_qc.domain.taxonomy import ALLOWED_VIOLATIONS, AnatomicalRegion
from dxa_qc.models.region.classifier import RegionClass


class DatasetIndexError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class IndexedImage:
    study_id: str
    path: Path
    pixel_digest: str
    region_class: RegionClass
    is_duplicate: bool
    # None when the experts did not annotate this region (e.g. endoprosthesis hips).
    label: RegionLabel | None
    tags: frozenset[ClinicalTag]

    @property
    def region(self) -> AnatomicalRegion:
        return self.region_class.region


def _region_label(study: StudyLabels, cls: RegionClass) -> RegionLabel | None:
    if cls is RegionClass.SPINE:
        return study.spine
    side = cls.side
    if side is None:
        raise DatasetIndexError(f"hip class without a side: {cls}")
    return study.hips.get(side)


def build_index(
    studies_dir: Path,
    annotations: dict[str, RegionAnnotation],
    labels: dict[str, StudyLabels],
) -> list[IndexedImage]:
    """Index every ``*.dcm`` under ``studies_dir/<study_id>/...``; fails on any unannotated image."""
    studies_dir = Path(studies_dir)
    index: list[IndexedImage] = []
    seen: set[tuple[str, str]] = set()
    for path in sorted(studies_dir.rglob("*.dcm")):
        study_id = path.relative_to(studies_dir).parts[0]
        if study_id not in labels:
            raise DatasetIndexError(f"study folder without expert labels: {study_id}")
        digest = pixel_digest(path)
        ann = annotations.get(digest)
        if ann is None:
            raise DatasetIndexError(f"image has no region annotation (re-run the review): {path}")
        if ann.study_id != study_id:
            raise DatasetIndexError(f"annotation for {path} belongs to another study ({ann.study_id})")
        key = (study_id, digest)
        index.append(
            IndexedImage(
                study_id=study_id,
                path=path,
                pixel_digest=digest,
                region_class=ann.label,
                is_duplicate=key in seen,
                label=_region_label(labels[study_id], ann.label),
                tags=labels[study_id].tags,
            )
        )
        seen.add(key)
    return index


def write_index_csv(index: list[IndexedImage], path: Path, studies_dir: Path) -> None:
    """Flat CSV for notebooks/training: one 0/1 column per violation, empty when unlabeled."""
    violation_columns = sorted({v for vs in ALLOWED_VIOLATIONS.values() for v in vs}, key=lambda v: v.name)
    header = [
        "study_id",
        "path",
        "pixel_digest",
        "region_class",
        "is_duplicate",
        "labeled",
        "overall",
        *(f"v_{v.name.lower()}" for v in violation_columns),
        "tags",
    ]
    with Path(path).open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for item in index:
            allowed = ALLOWED_VIOLATIONS[item.region]
            flags = [
                "" if item.label is None or v not in allowed else int(v in item.label.violations)
                for v in violation_columns
            ]
            writer.writerow(
                [
                    item.study_id,
                    item.path.relative_to(studies_dir).as_posix(),
                    item.pixel_digest,
                    item.region_class.value,
                    int(item.is_duplicate),
                    int(item.label is not None),
                    "" if item.label is None else item.label.overall,
                    *flags,
                    ";".join(sorted(t.value for t in item.tags)),
                ]
            )
