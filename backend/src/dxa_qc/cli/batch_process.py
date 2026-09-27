"""Batch quality control of DXA DICOM studies.

Example::

    python -m dxa_qc.cli.batch_process --input /data/studies.zip --output /data/results --format xlsx

With ``--series`` the output also gets ``results_details.csv`` (explanations, measurements, proposed
corrections), ``visualization/*.png`` and ``dicom/*.dcm`` (SR with the conclusion + Secondary
Capture with the violation overlay, filed into the source study).

Exit codes: 0 - report written (individual files may still have Failure status);
2 - the input as a whole is unusable (missing, unsafe archive) or configuration/weights are invalid.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

from dxa_qc.config import ConfigError, load_settings
from dxa_qc.ingest import UnsafeArchiveError
from dxa_qc.models.region import ModelFormatError, RegionClassifier
from dxa_qc.pipeline import ImageProcessor, PriorBaselineAssessor, ProcessedImage, QualityAssessor, run_batch
from dxa_qc.report import ReportFormat

WEIGHTS_DIR_ENV = "DXA_QC_WEIGHTS_DIR"
_DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[3] / "weights"
EXIT_OK, EXIT_INPUT_ERROR = 0, 2
# Fixed, so a repeated run on the same data produces the same DICOM UIDs.
SERIES_KEY = "dxa-qc-cli"

log = logging.getLogger("dxa_qc.cli")


ENSEMBLE = "ensemble"


class AssessorLoadError(RuntimeError):
    pass


def _build_assessor(name: str, weights_dir: Path, device: str | None) -> QualityAssessor:
    if name == PriorBaselineAssessor.name:
        return PriorBaselineAssessor()
    if name == ENSEMBLE:
        # Imported lazily: torch is heavy and not needed for the baseline or for --help.
        from dxa_qc.models.quality.assessor import EnsembleQualityAssessor
        from dxa_qc.models.quality.embeddings import BackboneError
        from dxa_qc.models.quality.model import QualityModelFormatError

        try:
            return EnsembleQualityAssessor.load(weights_dir, device)
        except (QualityModelFormatError, BackboneError) as exc:
            raise AssessorLoadError(str(exc)) from exc
    raise ValueError(f"unknown assessor {name!r}")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m dxa_qc.cli.batch_process",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True, help="DICOM file, folder or ZIP archive")
    parser.add_argument("--output", type=Path, required=True, help="directory for the report")
    parser.add_argument("--format", choices=[f.value for f in ReportFormat], default=ReportFormat.XLSX.value)
    parser.add_argument("--config-dir", type=Path, default=None, help="overrides DXA_QC_CONFIG_DIR")
    parser.add_argument("--weights-dir", type=Path, default=None, help=f"overrides {WEIGHTS_DIR_ENV}")
    parser.add_argument(
        "--assessor",
        choices=[ENSEMBLE, PriorBaselineAssessor.name],
        default=ENSEMBLE,
        help="trained ensemble (default) or the prior baseline used as a reference floor",
    )
    parser.add_argument(
        "--device", default=None, help="torch device for the ensemble (default: cuda if available)"
    )
    parser.add_argument(
        "--series",
        action="store_true",
        help="also write details, PNG visualizations and DICOM series (SR + Secondary Capture)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    weights_dir = Path(args.weights_dir or os.environ.get(WEIGHTS_DIR_ENV) or _DEFAULT_WEIGHTS_DIR)
    try:
        settings = load_settings(args.config_dir)
        region_model = RegionClassifier.load(weights_dir / "region_classifier.npz")
        assessor = _build_assessor(args.assessor, weights_dir, args.device)
    except (ConfigError, ModelFormatError, AssessorLoadError) as exc:
        log.error("%s", exc)
        return EXIT_INPUT_ERROR

    if isinstance(assessor, PriorBaselineAssessor):
        log.warning("using the prior baseline assessor: quality_class/quality_prob are NOT model predictions")

    processor = ImageProcessor(settings, region_model, assessor)
    details: list[dict[str, Any]] = []

    def on_result(index: int, processed: ProcessedImage) -> None:
        # Imported here: rendering pulls in OpenCV, which the plain table does not need.
        from dxa_qc.pipeline.outputs import describe, write_image_outputs

        row = describe(processed, settings, index)
        details.append(row)
        write_image_outputs(
            processed,
            row,
            images_dir=args.output / "visualization",
            dicom_dir=args.output / "dicom",
            series_key=SERIES_KEY,
        )

    try:
        result = run_batch(
            args.input,
            args.output,
            ReportFormat(args.format),
            processor,
            on_result=on_result if args.series else None,
        )
    except (FileNotFoundError, UnsafeArchiveError) as exc:
        log.error("input rejected: %s", exc)
        return EXIT_INPUT_ERROR

    log.info(
        "processed %d image(s), %d failed; report: %s",
        len(result.rows),
        result.n_failed,
        result.report.table,
    )
    if result.report.errors:
        log.info("failure details: %s", result.report.errors)
    if args.series:
        from dxa_qc.pipeline.outputs import write_details_csv

        path = write_details_csv(args.output / "results_details.csv", details)
        log.info("details: %s; visualizations and DICOM series: %s", path, args.output)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
