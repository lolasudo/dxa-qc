"""Convert the organizer's dataset (``разметка.xlsx`` + verified region table) into a manifest.

Usage::

    python -m dxa_qc.training.make_manifest --studies ../Датасет/Исследования \
        --labels ../Датасет/разметка.xlsx --annotations data/region_annotations.csv \
        --output ../Датасет/manifest.csv

Byte-identical duplicates inside a study are dropped (they would double-weight an image).
Paths are written relative to ``--images-root`` (default: the manifest's directory), which must
contain the studies folder; pass the same ``--images-root`` to training if it is not the default.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dxa_qc.data import build_index, load_labels, load_region_annotations
from dxa_qc.data.manifest import CRITERION_COLUMNS, write_manifest
from dxa_qc.domain.taxonomy import ALLOWED_VIOLATIONS

log = logging.getLogger(__name__)


def build_rows(
    studies: Path, labels_xlsx: Path, annotations_csv: Path, images_root: Path
) -> list[dict[str, str]]:
    root = Path(images_root).resolve()
    if not Path(studies).resolve().is_relative_to(root):
        raise ValueError(f"studies folder {studies} is not inside the images root {root}")
    index = build_index(studies, load_region_annotations(annotations_csv), load_labels(labels_xlsx))
    rows = []
    for item in index:
        if item.is_duplicate:
            continue
        row = {
            "path": Path(item.path).resolve().relative_to(root).as_posix(),
            "study_id": item.study_id,
            "region": item.region.name.lower(),
            "side": item.region_class.side.value if item.region_class.side else "",
            "comment": ";".join(sorted(t.value for t in item.tags)),
        }
        if item.label is not None:
            row["overall"] = str(item.label.overall)
            for v in ALLOWED_VIOLATIONS[item.region]:
                row[CRITERION_COLUMNS[v]] = str(int(v in item.label.violations))
        rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--studies", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, help="root for written paths (default: output's dir)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = args.images_root or args.output.resolve().parent
    try:
        rows = build_rows(args.studies, args.labels, args.annotations, root)
    except ValueError as exc:
        parser.error(str(exc))
    write_manifest(rows, args.output)
    log.info(
        "wrote %d images (%d labeled) to %s",
        len(rows),
        sum(bool(r.get("overall")) for r in rows),
        args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
