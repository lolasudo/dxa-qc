"""Batch processing core shared by the CLI and (next stage) the REST API.

Contract: every discovered image yields exactly one report row; a bad
file becomes a ``Failure`` row and never aborts the batch.
"""

from __future__ import annotations

import logging
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from dxa_qc.config.settings import Settings
from dxa_qc.dicom_io.reader import DicomImage, DicomReadError, read_dicom
from dxa_qc.domain.taxonomy import format_violations
from dxa_qc.ingest.discovery import InputItem, collect_inputs
from dxa_qc.models.region.classifier import RegionClassifier, RegionPrediction
from dxa_qc.pipeline.assessment import QualityAssessment, QualityAssessor
from dxa_qc.report.schema import ProcessingStatus, ResultRow
from dxa_qc.report.writer import ReportFormat, WrittenReport, write_report

log = logging.getLogger(__name__)


class RegionNotRecognizedError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class BatchResult:
    rows: list[ResultRow]
    report: WrittenReport

    @property
    def n_failed(self) -> int:
        return sum(r.processing_status is ProcessingStatus.FAILURE for r in self.rows)


@dataclass(frozen=True, slots=True)
class ProcessedImage:
    """A report row plus the intermediate results needed for previews and explanations.

    ``image``/``region``/``quality`` are ``None`` when processing failed before that stage.
    """

    item: InputItem
    row: ResultRow
    image: DicomImage | None = None
    region: RegionPrediction | None = None
    quality: QualityAssessment | None = None


class ImageProcessor:
    def __init__(self, settings: Settings, region_model: RegionClassifier, assessor: QualityAssessor) -> None:
        self.settings = settings
        self.region_model = region_model
        self.assessor = assessor

    def _process(self, item: InputItem) -> ProcessedImage:
        started = time.perf_counter()
        limits = self.settings.ingest
        image = read_dicom(
            item.path,
            self.settings.pixel_spacing,
            limits.max_file_bytes,
            max_side=limits.max_image_side,
            physical_range_mm=(limits.min_physical_mm, limits.max_physical_mm),
        )
        region = self.region_model.predict(image.pixels)
        if region.confidence < self.settings.region.min_confidence:
            raise RegionNotRecognizedError(
                f"anatomical region not recognized (confidence {region.confidence:.2f} "
                f"< {self.settings.region.min_confidence})"
            )
        quality = self.assessor.assess(image, region)
        violation_type = format_violations(
            region.region, quality.violations, self.settings.report.violation_separator
        )
        row = ResultRow(
            path_to_study=item.display_path,
            study_uid=image.study_uid,
            image_uid=image.image_uid,
            anatomical_region=region.region,
            quality_class=quality.quality_class,
            quality_prob=quality.quality_prob,
            violation_type=violation_type,
            processing_status=ProcessingStatus.SUCCESS,
            time_of_processing=time.perf_counter() - started,
        )
        return ProcessedImage(item, row, image, region, quality)

    def process_detailed(self, item: InputItem) -> ProcessedImage:
        started = time.perf_counter()
        try:
            return self._process(item)
        except (DicomReadError, RegionNotRecognizedError) as exc:
            error = str(exc)
            log.warning("failed %s: %s", item.display_path, error)
        except Exception as exc:
            # Broad on purpose: a bug in a model or decoder must cost one row, not the batch.
            # The report is handed to users: the exception text (paths, internals) stays in the log.
            error = f"internal error: {type(exc).__name__} (details in the service log)"
            log.exception("unexpected error on %s", item.display_path)
        row = ResultRow(
            path_to_study=item.display_path,
            processing_status=ProcessingStatus.FAILURE,
            time_of_processing=time.perf_counter() - started,
            error=error,
        )
        return ProcessedImage(item, row)

    def process(self, item: InputItem) -> ResultRow:
        return self.process_detailed(item).row


def run_batch(
    source: Path,
    output_dir: Path,
    fmt: ReportFormat,
    processor: ImageProcessor,
    on_progress: Callable[[int, int], None] | None = None,
    on_result: Callable[[int, ProcessedImage], None] | None = None,
) -> BatchResult:
    """Process everything under ``source`` (file, folder or ZIP) and write the report.

    Raises only for problems with the input as a whole (missing path, unsafe archive).
    ``on_result(index, processed)`` receives every image's intermediate results in order.
    """
    with tempfile.TemporaryDirectory(prefix="dxa_qc_") as workdir:
        items = collect_inputs(source, Path(workdir), processor.settings.ingest)
        log.info("discovered %d image(s) in %s", len(items), source)
        rows: list[ResultRow] = []
        if on_progress:
            on_progress(0, len(items))
        for n, item in enumerate(items, start=1):
            processed = processor.process_detailed(item)
            rows.append(processed.row)
            if on_result:
                # Side outputs (previews) run while the extracted file still exists; their
                # failure must not lose the row, so they are isolated here.
                try:
                    on_result(n - 1, processed)
                except Exception:
                    log.exception("side output failed for %s", item.display_path)
            if on_progress:
                on_progress(n, len(items))
    report = write_report(rows, output_dir, fmt)
    return BatchResult(rows=rows, report=report)
