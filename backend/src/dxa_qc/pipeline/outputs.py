"""Per-image outputs beyond the report table, shared by the REST API and the CLI.

* details: explanation, per-criterion probabilities, measurements, projection and proposed
  corrections;
* files: PNG preview with a transparent overlay of the found violations, and DICOM series filed
  into the source study - Enhanced SR with the conclusion and a Secondary Capture with the
  overlay.

Side outputs never cost the row: a failure to render or write them is logged and skipped.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dxa_qc.config.settings import Settings
from dxa_qc.domain.taxonomy import ALLOWED_VIOLATIONS
from dxa_qc.explain import describe_projection, explain, explain_failure, propose_corrections
from dxa_qc.pipeline.batch import ProcessedImage
from dxa_qc.report.dicom import SourceRef, write_outputs
from dxa_qc.report.schema import ProcessingStatus
from dxa_qc.report.writer import neutralize_formula
from dxa_qc.visualization import encode_png, render_overlay, render_preview

log = logging.getLogger(__name__)

IMAGE_KINDS = ("preview", "overlay")
DICOM_KINDS = ("sr", "sc")


def image_file(images_dir: Path, index: int, kind: str) -> Path:
    return images_dir / f"{index:05d}_{kind}.png"


def dicom_file(dicom_dir: Path, index: int, kind: str) -> Path:
    return dicom_dir / f"{index:05d}_{kind}.dcm"


def describe(p: ProcessedImage, settings: Settings, index: int) -> dict[str, Any]:
    """Report row plus everything a specialist needs to check it."""
    row: dict[str, Any] = p.row.table_values()
    row.update(index=index, error_reason=None, projection=None, criteria=[], findings={}, corrections=[])
    if p.row.processing_status is ProcessingStatus.FAILURE:
        row["error_reason"] = explain_failure(p.row.error)
        row["explanation"] = row["error_reason"]
        return row
    if p.region is None or p.quality is None:  # success rows always carry them
        raise RuntimeError(f"successful row without intermediates: {p.row.path_to_study}")
    region, quality = p.region.region, p.quality
    row["projection"] = describe_projection(region, p.region.side)
    row["explanation"] = explain(region, quality, settings.spine.max_axis_tilt_deg)
    row["criteria"] = [
        {
            "violation": v.value,
            "probability": round(quality.criterion_probs[v], 4) if v in quality.criterion_probs else None,
            "violated": v in quality.violations,
        }
        for v in ALLOWED_VIOLATIONS[region]
    ]
    row["findings"] = dict(quality.findings)
    row["corrections"] = [c.to_dict() for c in propose_corrections(region, quality, settings)]
    return row


@dataclass(frozen=True, slots=True)
class WrittenOutputs:
    images: dict[str, Path]
    dicom: dict[str, Path]
    # Kept to rewrite the SR after the specialist's review, when the source file is gone.
    source: SourceRef | None = None


def write_image_outputs(
    p: ProcessedImage,
    details: dict[str, Any],
    *,
    images_dir: Path | None,
    dicom_dir: Path | None,
    series_key: str,
) -> WrittenOutputs:
    """Must run while ``p.item.path`` still exists (the DICOM outputs copy study attributes)."""
    if p.image is None or p.region is None or p.quality is None:
        return WrittenOutputs({}, {})
    index = int(details["index"])
    try:
        preview = render_preview(p.image)
        overlay = render_overlay(p.image, p.region, p.quality)
    except Exception:
        log.exception("visualization failed for %s", p.row.path_to_study)
        return WrittenOutputs({}, {})

    images: dict[str, Path] = {}
    if images_dir is not None:
        try:
            images_dir.mkdir(parents=True, exist_ok=True)
            for kind, pixels in zip(IMAGE_KINDS, (preview, overlay), strict=True):
                path = image_file(images_dir, index, kind)
                path.write_bytes(encode_png(pixels))
                images[kind] = path
        except Exception:
            log.exception("PNG output failed for %s", p.row.path_to_study)

    dicom: dict[str, Path] = {}
    source: SourceRef | None = None
    if dicom_dir is not None:
        try:
            source = SourceRef.from_file(p.item.path)
            sr, sc = write_outputs(
                source, details, details["corrections"], preview, overlay, dicom_dir,
                series_key=series_key, index=index,
            )  # fmt: skip
            dicom = {"sr": sr, "sc": sc}
        except Exception:
            log.exception("DICOM output failed for %s", p.row.path_to_study)
    return WrittenOutputs(images, dicom, source)


DETAIL_COLUMNS = (
    "path_to_study",
    "anatomical_region",
    "projection",
    "quality_class",
    "quality_prob",
    "violation_type",
    "processing_status",
    "explanation",
)


def _violation_names() -> list[str]:
    return [v.value for v in dict.fromkeys(v for vs in ALLOWED_VIOLATIONS.values() for v in vs)]


def write_details_csv(path: Path, details: list[dict[str, Any]]) -> Path:
    """Explanations, per-criterion probabilities, measurements and corrections, one row per image."""
    names = _violation_names()
    with path.open("w", encoding="utf-8-sig", newline="") as fh:  # BOM: Excel opens Cyrillic correctly
        writer = csv.writer(fh)
        writer.writerow([*DETAIL_COLUMNS, *(f"p: {n}" for n in names), "corrections", "findings"])
        for d in details:
            probs = {c["violation"]: c["probability"] for c in d.get("criteria", [])}
            writer.writerow(
                [neutralize_formula("" if d.get(c) is None else d.get(c)) for c in DETAIL_COLUMNS]
                + ["" if probs.get(n) is None else probs[n] for n in names]
                + [neutralize_formula(" | ".join(c["action"] for c in d.get("corrections", [])))]
                + [json.dumps(d.get("findings", {}), ensure_ascii=False)]
            )
    return path
