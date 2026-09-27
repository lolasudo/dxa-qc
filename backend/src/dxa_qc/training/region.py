"""Train and evaluate the region/side classifier on the verified annotations.

Usage::

    python -m dxa_qc.training.region --studies ../Датасет/Исследования \
        --annotations data/region_annotations.csv --output weights/region_classifier.npz
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dxa_qc.config import load_settings
from dxa_qc.data.annotations import load_region_annotations
from dxa_qc.data.manifest import ManifestRow, load_manifest
from dxa_qc.dicom_io import DicomReadError, pixel_digest, read_dicom
from dxa_qc.models.region import RegionClass, RegionClassifier

log = logging.getLogger(__name__)

SEED = 20260924


@dataclass(frozen=True, slots=True)
class AnnotatedImage:
    study_id: str
    path: Path
    label: RegionClass
    pixels: np.ndarray


def load_annotated_images(annotations_csv: Path, studies_dir: Path) -> list[AnnotatedImage]:
    """One entry per unique image; the stored pixel digest guards against a stale annotation."""
    settings = load_settings()
    items: list[AnnotatedImage] = []
    for ann in load_region_annotations(annotations_csv).values():
        path = Path(studies_dir, *ann.example_path.parts)
        if pixel_digest(path) != ann.pixel_digest:
            raise ValueError(f"pixel digest mismatch for {path}: annotation is stale")
        img = read_dicom(path, settings.pixel_spacing, settings.ingest.max_file_bytes)
        items.append(AnnotatedImage(ann.study_id, path, ann.label, img.pixels))
    return items


def manifest_class(row: ManifestRow) -> RegionClass | None:
    """Region class a manifest row states explicitly; ``None`` if region or hip side is left empty."""
    if row.region is None:
        return None
    for cls in RegionClass:
        if cls.region is row.region and cls.side is row.side:
            return cls
    return None  # hip without a side


def load_manifest_images(rows: list[ManifestRow]) -> list[AnnotatedImage]:
    """Rows with an explicit region class; byte-identical repeats inside a study are counted once.

    Unreadable files are skipped with a warning, as in quality training.
    """
    settings = load_settings()
    limits = settings.ingest
    items: list[AnnotatedImage] = []
    seen: set[tuple[str, str]] = set()
    skipped = 0
    for row in rows:
        label = manifest_class(row)
        if label is None:
            continue
        try:
            key = (row.study_id, pixel_digest(row.path))
            if key in seen:
                continue
            img = read_dicom(
                row.path,
                settings.pixel_spacing,
                limits.max_file_bytes,
                max_side=limits.max_image_side,
                physical_range_mm=(limits.min_physical_mm, limits.max_physical_mm),
            )
        except (DicomReadError, OSError) as exc:
            skipped += 1
            log.warning("skipping %s: %s", row.path, exc)
            continue
        seen.add(key)
        items.append(AnnotatedImage(row.study_id, row.path, label, img.pixels))
    if skipped:
        log.warning("%d image(s) skipped (see warnings above)", skipped)
    missing = {c.value for c in RegionClass} - {i.label.value for i in items}
    if missing:
        raise ValueError(f"manifest has no images of class(es) {sorted(missing)}")
    return items


def cross_validate(items: list[AnnotatedImage], n_splits: int = 5, c: float = 0.1) -> dict[str, object]:
    """Study-grouped stratified CV: images of one study never appear in both train and test."""
    from sklearn.model_selection import StratifiedGroupKFold

    labels = np.array([i.label.value for i in items])
    groups = np.array([i.study_id for i in items])
    predicted = np.empty_like(labels)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    for train_idx, test_idx in cv.split(np.zeros(len(items)), labels, groups):
        model = RegionClassifier.fit(
            [items[i].pixels for i in train_idx], [items[i].label for i in train_idx], c=c, seed=SEED
        )
        for i in test_idx:
            predicted[i] = model.predict(items[i].pixels).label.value
    errors = [
        {"path": str(items[i].path), "true": labels[i], "predicted": predicted[i]}
        for i in np.flatnonzero(predicted != labels)
    ]
    return {
        "n_images": len(items),
        "n_studies": len(set(groups)),
        "n_splits": n_splits,
        "accuracy": float((predicted == labels).mean()),
        "per_class_counts": {c.value: int((labels == c.value).sum()) for c in RegionClass},
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = parser.add_argument_group("dataset: a manifest, or the organizer's format")
    src.add_argument("--manifest", type=Path, help="training manifest CSV (see dxa_qc.data.manifest)")
    src.add_argument("--images-root", type=Path, help="root for manifest paths (default: the manifest's dir)")
    src.add_argument("--studies", type=Path, help="organizer format: studies folder")
    src.add_argument("--annotations", type=Path, help="organizer format: region_annotations.csv")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--c", type=float, default=0.1, help="inverse L2 regularization strength")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.manifest:
        items = load_manifest_images(load_manifest(args.manifest, args.images_root))
        source = args.manifest
    elif args.studies and args.annotations:
        items = load_annotated_images(args.annotations, args.studies)
        source = args.annotations
    else:
        parser.error("give --manifest, or both --studies and --annotations")
    report = cross_validate(items, c=args.c)
    log.info(
        "CV accuracy %.4f on %d images / %d studies",
        report["accuracy"],
        report["n_images"],
        report["n_studies"],
    )
    for err in report["errors"]:
        log.warning("CV error: %s", err)

    model = RegionClassifier.fit([i.pixels for i in items], [i.label for i in items], c=args.c, seed=SEED)
    dataset_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    model.save(
        args.output,
        metadata={
            "c": args.c,
            "seed": SEED,
            "dataset_sha256": dataset_sha,
            "cv_accuracy": report["accuracy"],
        },
    )
    args.output.with_suffix(".cv.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info("saved %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
