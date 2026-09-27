"""Batch job store for the REST API (no Celery/Redis - one worker, in-process).

One worker thread runs every job (batches and single studies): the models hold one GPU and are
not meant for concurrent calls, and the time budget is per study, not per concurrent user. Job state
lives in memory; files live under ``data_dir/<job_id>/`` and are removed after the TTL.
"""

from __future__ import annotations

import contextlib
import csv
import logging
import os
import shutil
import threading
import uuid
import zipfile
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

from dxa_qc.pipeline.batch import ImageProcessor, ProcessedImage, run_batch
from dxa_qc.pipeline.outputs import (
    DICOM_KINDS,
    IMAGE_KINDS,
    describe,
    dicom_file,
    image_file,
    write_details_csv,
    write_image_outputs,
)
from dxa_qc.report.dicom import Review, SourceRef, write_sr
from dxa_qc.report.schema import REPORT_COLUMNS, ProcessingStatus
from dxa_qc.report.writer import ReportFormat, neutralize_formula, write_report

log = logging.getLogger(__name__)

JOB_ID_PATTERN = r"^[0-9a-f]{32}$"
REVIEWS_FILE = "results_reviewed.csv"
__all__ = ["DICOM_KINDS", "IMAGE_KINDS", "JOB_ID_PATTERN", "Job", "JobKind", "JobStatus", "JobStore",
           "QueueFullError", "ReviewError", "StorageFullError"]  # fmt: skip


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class JobKind(StrEnum):
    BATCH = "batch"
    STUDY = "study"


class QueueFullError(RuntimeError):
    pass


class StorageFullError(RuntimeError):
    pass


class ReviewError(ValueError):
    """The specialist's decision contradicts the row (unknown violation, failed row, ...)."""


def directory_size(path: Path) -> int:
    total = 0
    for dirpath, _, filenames in os.walk(path, followlinks=False):
        for name in filenames:
            with contextlib.suppress(OSError):  # file removed concurrently
                total += (Path(dirpath) / name).lstat().st_size
    return total


def _now() -> datetime:
    return datetime.now(UTC)


@dataclass
class Job:
    id: str
    kind: JobKind
    source_name: str
    directory: Path
    status: JobStatus = JobStatus.QUEUED
    processed: int = 0
    total: int | None = None
    error: str | None = None
    created_at: datetime = field(default_factory=_now)
    updated_at: datetime = field(default_factory=_now)
    rows: list[dict[str, Any]] = field(default_factory=list)
    # Source references of processed images, to reissue their SR after a review.
    sources: dict[int, SourceRef] = field(default_factory=dict)
    # Disk usage, measured once when the job finishes (its files never change afterwards).
    size_bytes: int | None = None

    @property
    def upload_path(self) -> Path:
        return self.directory / "upload" / self.source_name

    @property
    def report_dir(self) -> Path:
        return self.directory / "report"

    @property
    def images_dir(self) -> Path:
        return self.directory / "images"

    @property
    def dicom_dir(self) -> Path:
        return self.directory / "dicom"

    def image_path(self, index: int, kind: str) -> Path:
        return image_file(self.images_dir, index, kind)

    def dicom_path(self, index: int, kind: str) -> Path:
        return dicom_file(self.dicom_dir, index, kind)

    def status_payload(self) -> dict[str, Any]:
        progress = (
            100.0
            if self.status is JobStatus.DONE
            else (round(100.0 * self.processed / self.total, 1) if self.total else 0.0)
        )
        return {
            "job_id": self.id,
            "status": self.status.value,
            "progress": progress,
            "processed": self.processed,
            "total": self.total,
            "error": self.error,
            "source_name": self.source_name,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class JobStore:
    def __init__(
        self,
        data_dir: Path,
        processor_factory: Callable[[], ImageProcessor],
        *,
        max_queued_jobs: int,
        job_ttl: timedelta,
        max_pending_studies: int = 4,
        max_storage_bytes: int = 20 * 1024**3,
        min_free_disk_bytes: int = 0,
        api_prefix: str = "/api/v1",
    ) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._processor_factory = processor_factory
        self._processor: ImageProcessor | None = None
        self.max_queued_jobs = max_queued_jobs
        self.max_pending_studies = max_pending_studies
        self.max_storage_bytes = max_storage_bytes
        self.min_free_disk_bytes = min_free_disk_bytes
        self.job_ttl = job_ttl
        self.api_prefix = api_prefix
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._archive_lock = threading.Lock()
        self._review_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dxa-qc-worker")
        # Job metadata is in memory, so leftovers of a previous process are orphans.
        for stale in self.data_dir.iterdir():
            shutil.rmtree(stale, ignore_errors=True) if stale.is_dir() else stale.unlink(missing_ok=True)

    # ---- lifecycle --------------------------------------------------------------------------

    def warm_up(self) -> Future[None]:
        """Load models on the worker thread (the thread that will use them)."""
        return self._executor.submit(self._get_processor)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _get_processor(self) -> ImageProcessor:
        if self._processor is None:
            self._processor = self._processor_factory()
        return self._processor

    # ---- jobs ---------------------------------------------------------------------------------

    def storage_used(self) -> int:
        with self._lock:
            jobs = list(self._jobs.values())
        return sum(j.size_bytes if j.size_bytes is not None else directory_size(j.directory) for j in jobs)

    def _check_capacity(self) -> None:
        free = shutil.disk_usage(self.data_dir).free
        if free < self.min_free_disk_bytes:
            raise StorageFullError(f"only {free} bytes free on the data volume")
        used = self.storage_used()
        if used >= self.max_storage_bytes:
            raise StorageFullError(f"job storage quota exhausted ({used} bytes)")

    def new_job(self, kind: JobKind, source_name: str) -> Job:
        """Reserve a job and its directory; the caller writes the upload to ``job.upload_path``.

        Raises :class:`QueueFullError` / :class:`StorageFullError` instead of accepting work the
        server cannot hold: back-pressure is how a flood of uploads is kept from taking it down.
        """
        self.cleanup()
        self._check_capacity()
        with self._lock:
            active = [j for j in self._jobs.values() if j.status in (JobStatus.QUEUED, JobStatus.RUNNING)]
            same_kind = sum(j.kind is kind for j in active)
            limit = self.max_queued_jobs if kind is JobKind.BATCH else self.max_pending_studies
            if same_kind >= limit:
                raise QueueFullError(f"too many active {kind.value} jobs ({same_kind}), retry later")
            job_id = uuid.uuid4().hex
            job = Job(job_id, kind, _safe_name(source_name), self.data_dir / job_id)
            job.upload_path.parent.mkdir(parents=True)
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def discard(self, job: Job) -> None:
        with self._lock:
            self._jobs.pop(job.id, None)
        shutil.rmtree(job.directory, ignore_errors=True)

    def start(self, job: Job) -> Future[None]:
        return self._executor.submit(self._run, job)

    def cleanup(self) -> None:
        cutoff = _now() - self.job_ttl
        with self._lock:
            expired = [
                j
                for j in self._jobs.values()
                if j.status in (JobStatus.DONE, JobStatus.FAILED) and j.updated_at < cutoff
            ]
            for job in expired:
                del self._jobs[job.id]
        for job in expired:
            shutil.rmtree(job.directory, ignore_errors=True)

    # ---- worker -------------------------------------------------------------------------------

    def _touch(self, job: Job, **changes: Any) -> None:
        with self._lock:
            for key, value in changes.items():
                setattr(job, key, value)
            job.updated_at = _now()

    def _run(self, job: Job) -> None:
        self._touch(job, status=JobStatus.RUNNING)
        try:
            processor = self._get_processor()
            details: list[dict[str, Any]] = []

            def on_result(index: int, processed: ProcessedImage) -> None:
                details.append(self._describe(job, index, processed, processor))

            result = run_batch(
                job.upload_path,
                job.report_dir,
                ReportFormat.XLSX,
                processor,
                on_progress=lambda n, total: self._touch(job, processed=n, total=total),
                on_result=on_result,
            )
            write_report(result.rows, job.report_dir, ReportFormat.CSV)
            write_details_csv(job.report_dir / "results_details.csv", details)
            job.upload_path.unlink(missing_ok=True)  # the original is not kept longer than needed
            self._touch(job, status=JobStatus.DONE, rows=details, processed=len(details), total=len(details))
        except Exception as exc:  # the job fails as a whole only for input-level problems
            log.warning("job %s failed: %s", job.id, exc)
            job.upload_path.unlink(missing_ok=True)
            self._touch(job, status=JobStatus.FAILED, error=_input_error_message(exc))
        finally:
            job.size_bytes = directory_size(job.directory)

    def _describe(self, job: Job, index: int, p: ProcessedImage, processor: ImageProcessor) -> dict[str, Any]:
        row = describe(p, processor.settings, index)
        row["review"] = None
        written = write_image_outputs(
            p, row, images_dir=job.images_dir, dicom_dir=job.dicom_dir, series_key=job.id
        )
        for kind in written.images:
            row[f"{kind}_url"] = self.image_url(job, index, kind)
        for kind in written.dicom:
            row[f"{kind}_url"] = self.dicom_url(job, index, kind)
        if written.source is not None:
            job.sources[index] = written.source
        return row

    def image_url(self, job: Job, index: int, kind: str) -> str:
        return f"{self.api_prefix}/batch/{job.id}/images/{index}/{kind}.png"

    def dicom_url(self, job: Job, index: int, kind: str) -> str:
        return f"{self.api_prefix}/batch/{job.id}/dicom/{index}/{kind}.dcm"

    # ---- specialist review -----------------------------------------------------------------

    def review(
        self,
        job: Job,
        index: int,
        *,
        decision: str,
        quality_class: int | None,
        violations: list[str],
        comment: str,
    ) -> dict[str, Any]:
        """Record the specialist's decision on one image: the verified verdict goes to
        ``results_reviewed.csv`` and the image's SR is reissued as VERIFIED."""
        if not 0 <= index < len(job.rows):
            raise KeyError(index)
        row = job.rows[index]
        if row["processing_status"] != ProcessingStatus.SUCCESS.value:
            raise ReviewError("Строку с ошибкой обработки нельзя подтвердить")
        if decision == "confirmed":
            quality_class = int(row["quality_class"])
            violations = [v for v in str(row["violation_type"] or "").split(";") if v]
        elif decision == "corrected":
            if quality_class not in (0, 1):
                raise ReviewError("Укажите итоговый класс качества: 0 или 1")
            allowed = [c["violation"] for c in row["criteria"]]
            unknown = [v for v in violations if v not in allowed]
            if unknown:
                raise ReviewError(f"Нарушения не относятся к этой области: {'; '.join(unknown)}")
            if quality_class == 0 and violations:
                raise ReviewError("У качественного исследования не может быть нарушений")
            violations = [v for v in allowed if v in violations]  # canonical order, no duplicates
        else:
            raise ReviewError(f"Неизвестное решение: {decision}")
        reviewed = Review(decision, int(quality_class), tuple(violations), comment.strip(), _now())
        with self._review_lock:
            source = job.sources.get(index)
            if source is not None:
                write_sr(source, row, row["corrections"], job.dicom_dir,
                         series_key=job.id, index=index, review=reviewed)  # fmt: skip
            row["review"] = {
                "decision": reviewed.decision,
                "quality_class": reviewed.quality_class,
                "violation_type": ";".join(reviewed.violations),
                "comment": reviewed.comment,
                "reviewed_at": reviewed.reviewed_at.isoformat(),
            }
            _write_reviews_csv(job.report_dir / REVIEWS_FILE, job.rows)
            self._touch(job)
            with self._archive_lock:  # the archive must not serve the pre-review report
                (job.directory / "result.zip").unlink(missing_ok=True)
                job.size_bytes = directory_size(job.directory)
        return row

    # ---- downloads ----------------------------------------------------------------------------

    def result_archive(self, job: Job) -> Path:
        """ZIP: report (xlsx + csv), errors, details, reviews, PNG visualizations and the DICOM
        series (SR + Secondary Capture) of every processed image."""
        archive = job.directory / "result.zip"
        # One builder at a time: concurrent downloads must not write the same temp file.
        with self._archive_lock:
            if archive.exists():
                return archive
            tmp = archive.with_suffix(".tmp")
            with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for f in sorted(job.report_dir.glob("*")):
                    zf.write(f, f.name)
                for f in sorted(job.images_dir.glob("*.png")):
                    zf.write(f, f"visualization/{f.name}")
                for f in sorted(job.dicom_dir.glob("*.dcm")):
                    zf.write(f, f"dicom/{f.name}")
            tmp.replace(archive)
            job.size_bytes = directory_size(job.directory)  # the archive counts towards the quota
        return archive


def _safe_name(name: str) -> str:
    """Only the base name, ASCII-safe characters kept; never a path."""
    base = Path(name.replace("\\", "/")).name or "upload"
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in base)[:120].lstrip(".")
    return cleaned or "upload"


def _input_error_message(exc: Exception) -> str:
    from dxa_qc.ingest import UnsafeArchiveError

    if isinstance(exc, UnsafeArchiveError):
        if "disk space" in str(exc):
            return "Недостаточно места на сервере для распаковки архива, повторите позже"
        return f"Архив отклонён как небезопасный: {exc}"
    if isinstance(exc, FileNotFoundError):
        return "Загруженный файл не найден"
    if isinstance(exc, zipfile.BadZipFile):
        return "Файл повреждён или не является ZIP-архивом"
    return f"Внутренняя ошибка обработки: {type(exc).__name__}"


_REVIEW_KEYS = ("decision", "quality_class", "violation_type", "comment", "reviewed_at")


def _write_reviews_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Every report row with the specialist's verdict next to the model's (empty if not reviewed)."""
    with path.open("w", encoding="utf-8-sig", newline="") as fh:  # BOM: Excel opens Cyrillic correctly
        writer = csv.writer(fh)
        writer.writerow([*REPORT_COLUMNS, *(f"review_{k}" for k in _REVIEW_KEYS)])
        for r in rows:
            review = r.get("review") or {}
            values = [r.get(c) for c in REPORT_COLUMNS] + [review.get(k) for k in _REVIEW_KEYS]
            writer.writerow([neutralize_formula("" if v is None else v) for v in values])
