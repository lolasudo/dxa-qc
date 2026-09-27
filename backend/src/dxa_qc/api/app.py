"""REST API over the same core as the CLI - no logic is duplicated here.

Run: ``python -m dxa_qc.api`` (single process: the models are loaded once per process).

Endpoints (prefix ``/api/v1``):

* ``GET  /health`` - liveness + whether the models are loaded;
* ``POST /batch`` - upload a ZIP (or a single DICOM), returns ``job_id`` (202);
* ``GET  /batch/{id}/status`` - queued/running/done/failed + progress;
* ``GET  /batch/{id}/results`` - report rows + explanation, criteria, image and DICOM URLs;
* ``GET  /batch/{id}/result`` - ZIP: xlsx/csv report, errors, details, visualizations;
* ``GET  /batch/{id}/images/{n}/{preview|overlay}.png``;
* ``GET  /batch/{id}/dicom/{n}/{sr|sc}.dcm`` - DICOM SR with the conclusion / Secondary Capture;
* ``POST /batch/{id}/rows/{n}/review`` - specialist confirms or corrects the result;
* ``POST /study`` - one DICOM, processed synchronously, returns its row.
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi import Path as PathParam
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from dxa_qc.api.jobs import (
    DICOM_KINDS,
    IMAGE_KINDS,
    JOB_ID_PATTERN,
    Job,
    JobKind,
    JobStatus,
    JobStore,
    QueueFullError,
    ReviewError,
    StorageFullError,
)
from dxa_qc.api.limits import BodySizeLimitMiddleware
from dxa_qc.config import Settings, load_settings
from dxa_qc.pipeline.batch import ImageProcessor

log = logging.getLogger(__name__)
# Access to patient images and results (HIPAA §164.312(b) audit controls): who, what, which job.
audit = logging.getLogger("dxa_qc.audit")

API_PREFIX = "/api/v1"
DATA_DIR_ENV = "DXA_QC_DATA_DIR"
WEIGHTS_DIR_ENV = "DXA_QC_WEIGHTS_DIR"
DEVICE_ENV = "DXA_QC_DEVICE"
CORS_ENV = "DXA_QC_CORS_ORIGINS"  # comma-separated; empty = same-origin only (nginx / vite proxy)
DOCS_ENV = "DXA_QC_API_DOCS"  # "0" hides /docs and /openapi.json (e.g. on an exposed deployment)
# Multipart framing (boundaries, part headers) on top of the file itself.
_MULTIPART_OVERHEAD = 64 * 1024
_BACKEND_ROOT = Path(__file__).resolve().parents[3]
_CHUNK = 1 << 20
_ZIP_MAGIC = (b"PK\x03\x04", b"PK\x05\x06")
_RETRY_AFTER_S = "30"

JobId = Annotated[str, PathParam(pattern=JOB_ID_PATTERN)]
RowIndex = Annotated[int, PathParam(ge=0, le=10**6)]


class ReviewRequest(BaseModel):
    """Specialist's decision: ``confirmed`` keeps the model's verdict, ``corrected`` replaces it."""

    model_config = ConfigDict(extra="forbid")

    decision: Literal["confirmed", "corrected"]
    quality_class: Literal[0, 1] | None = None
    violations: list[Annotated[str, Field(max_length=100)]] = Field(default_factory=list, max_length=10)
    comment: str = Field(default="", max_length=1000)


def default_processor_factory(settings: Settings, weights_dir: Path, device: str | None) -> ImageProcessor:
    # Imported lazily: torch is heavy and tests inject their own processor.
    from dxa_qc.models.quality.assessor import EnsembleQualityAssessor
    from dxa_qc.models.region import RegionClassifier

    region = RegionClassifier.load(weights_dir / "region_classifier.npz")
    assessor = EnsembleQualityAssessor.load(weights_dir, device)
    assessor.preload()  # "models_ready" in /health then means the first image is fast too
    return ImageProcessor(settings, region, assessor)


def create_app(
    *,
    settings: Settings | None = None,
    data_dir: Path | None = None,
    processor_factory: Callable[[], ImageProcessor] | None = None,
) -> FastAPI:
    settings = settings or load_settings()
    data_dir = Path(data_dir or os.environ.get(DATA_DIR_ENV) or _BACKEND_ROOT / "var" / "jobs")
    if processor_factory is None:
        weights_dir = Path(os.environ.get(WEIGHTS_DIR_ENV) or _BACKEND_ROOT / "weights")
        device = os.environ.get(DEVICE_ENV) or None

        def processor_factory() -> ImageProcessor:
            return default_processor_factory(settings, weights_dir, device)

    store = JobStore(
        data_dir,
        processor_factory,
        max_queued_jobs=settings.api.max_queued_jobs,
        job_ttl=timedelta(hours=settings.api.job_ttl_hours),
        max_pending_studies=settings.api.max_pending_studies,
        max_storage_bytes=settings.api.max_storage_bytes,
        min_free_disk_bytes=settings.api.min_free_disk_bytes,
        api_prefix=API_PREFIX,
    )
    state: dict[str, Any] = {"models_ready": False, "models_error": None}

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        warm = store.warm_up()

        def done(fut: Any) -> None:
            exc = fut.exception()
            state["models_ready"] = exc is None
            # Details (file system paths) go to the log only, never to unauthenticated clients.
            state["models_error"] = (
                None if exc is None else "веса моделей не найдены или повреждены, см. журнал backend"
            )
            if exc is not None:
                log.error("models failed to load: %s", exc)

        warm.add_done_callback(done)
        yield
        store.shutdown()

    docs = os.environ.get(DOCS_ENV, "1") != "0"
    app = FastAPI(
        title="DXA QC API",
        version="1.0.0",
        description="Контроль качества денситометрических исследований (DICOM)",
        lifespan=lifespan,
        docs_url=f"{API_PREFIX}/docs" if docs else None,
        redoc_url=None,
        openapi_url=f"{API_PREFIX}/openapi.json" if docs else None,
    )
    app.add_middleware(
        BodySizeLimitMiddleware,
        max_body_bytes=settings.api.max_upload_bytes + _MULTIPART_OVERHEAD,
        message=f"Файл больше {settings.api.max_upload_bytes // (1 << 20)} МБ",
    )
    origins = [o.strip() for o in os.environ.get(CORS_ENV, "").split(",") if o.strip()]
    if origins:
        app.add_middleware(
            CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["*"]
        )
    app.state.store = store

    def client(request: Request) -> str:
        # nginx sets X-Real-IP to the peer address it saw; without the proxy it is the socket peer.
        return request.headers.get("x-real-ip") or (request.client.host if request.client else "-")

    def job_or_404(job_id: str) -> Job:
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Задача не найдена или удалена по сроку хранения")
        return job

    async def save_upload(request: Request, upload: UploadFile, job: Job, allow_zip: bool) -> None:
        limit = settings.api.max_upload_bytes
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > limit + _CHUNK:
            raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, f"Файл больше {limit // (1 << 20)} МБ")
        size = 0
        head = b""
        with job.upload_path.open("wb") as out:
            while chunk := await upload.read(_CHUNK):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(
                        status.HTTP_413_CONTENT_TOO_LARGE, f"Файл больше {limit // (1 << 20)} МБ"
                    )
                if len(head) < 132:
                    head += chunk[: 132 - len(head)]
                out.write(chunk)
        if size == 0:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Пустой файл")
        is_zip = head[:4] in _ZIP_MAGIC
        is_dicom = head[128:132] == b"DICM"
        if not (is_dicom or (allow_zip and is_zip)):
            expected = "ZIP-архив или DICOM-файл" if allow_zip else "DICOM-файл"
            raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, f"Ожидается {expected}")

    async def accept(request: Request, file: UploadFile, kind: JobKind) -> Job:
        try:
            job = store.new_job(kind, file.filename or "upload")
        except QueueFullError as exc:
            raise HTTPException(
                status.HTTP_429_TOO_MANY_REQUESTS,
                "Очередь заполнена, повторите позже",
                headers={"Retry-After": _RETRY_AFTER_S},
            ) from exc
        except StorageFullError as exc:
            log.warning("upload refused: %s", exc)
            raise HTTPException(
                status.HTTP_507_INSUFFICIENT_STORAGE,
                "На сервере недостаточно места, повторите позже",
                headers={"Retry-After": _RETRY_AFTER_S},
            ) from exc
        try:
            await save_upload(request, file, job, allow_zip=kind is JobKind.BATCH)
        except BaseException:
            store.discard(job)
            raise
        audit.info("upload job=%s kind=%s client=%s", job.id, kind.value, client(request))
        return job

    @app.get(f"{API_PREFIX}/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "models_ready": state["models_ready"], "models_error": state["models_error"]}

    @app.post(f"{API_PREFIX}/batch", status_code=status.HTTP_202_ACCEPTED)
    async def submit_batch(request: Request, file: Annotated[UploadFile, File()]) -> dict[str, str]:
        job = await accept(request, file, JobKind.BATCH)
        store.start(job)
        return {"job_id": job.id}

    @app.get(f"{API_PREFIX}/batch/{{job_id}}/status")
    def batch_status(job_id: JobId) -> dict[str, Any]:
        return job_or_404(job_id).status_payload()

    def finished(job_id: str) -> Job:
        job = job_or_404(job_id)
        if job.status is JobStatus.FAILED:
            raise HTTPException(status.HTTP_409_CONFLICT, job.error or "Задача завершилась с ошибкой")
        if job.status is not JobStatus.DONE:
            raise HTTPException(status.HTTP_409_CONFLICT, "Задача ещё выполняется")
        return job

    @app.get(f"{API_PREFIX}/batch/{{job_id}}/results")
    def batch_results(request: Request, job_id: JobId) -> dict[str, Any]:
        job = finished(job_id)
        audit.info("view-results job=%s client=%s", job.id, client(request))
        return {"job_id": job.id, "rows": job.rows}

    @app.get(f"{API_PREFIX}/batch/{{job_id}}/result")
    async def batch_result(request: Request, job_id: JobId) -> FileResponse:
        job = finished(job_id)
        audit.info("download-archive job=%s client=%s", job.id, client(request))
        archive = await run_in_threadpool(store.result_archive, job)
        return FileResponse(archive, media_type="application/zip", filename=f"dxa_qc_{job.id[:8]}.zip")

    @app.get(f"{API_PREFIX}/batch/{{job_id}}/images/{{index}}/{{kind}}.png")
    def batch_image(job_id: JobId, index: RowIndex, kind: str) -> FileResponse:
        if kind not in IMAGE_KINDS:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Неизвестный тип изображения")
        path = job_or_404(job_id).image_path(index, kind)
        if not path.is_file():
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Изображение не найдено")
        # Patient images: never stored by browsers or intermediaries.
        return FileResponse(path, media_type="image/png", headers={"Cache-Control": "no-store"})

    @app.get(f"{API_PREFIX}/batch/{{job_id}}/dicom/{{index}}/{{kind}}.dcm")
    def batch_dicom(request: Request, job_id: JobId, index: RowIndex, kind: str) -> FileResponse:
        if kind not in DICOM_KINDS:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Неизвестный тип DICOM-объекта")
        job = job_or_404(job_id)
        path = job.dicom_path(index, kind)
        if not path.is_file():
            raise HTTPException(status.HTTP_404_NOT_FOUND, "DICOM-объект не найден")
        audit.info("download-dicom job=%s index=%d kind=%s client=%s", job.id, index, kind, client(request))
        return FileResponse(
            path,
            media_type="application/dicom",
            filename=f"dxa_qc_{job.id[:8]}_{index:05d}_{kind}.dcm",
            headers={"Cache-Control": "no-store"},
        )

    @app.post(f"{API_PREFIX}/batch/{{job_id}}/rows/{{index}}/review")
    def review_row(request: Request, job_id: JobId, index: RowIndex, body: ReviewRequest) -> dict[str, Any]:
        job = finished(job_id)
        try:
            row = store.review(
                job,
                index,
                decision=body.decision,
                quality_class=body.quality_class,
                violations=body.violations,
                comment=body.comment,
            )
        except KeyError as exc:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Строка не найдена") from exc
        except ReviewError as exc:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
        audit.info(
            "review job=%s index=%d decision=%s client=%s", job.id, index, body.decision, client(request)
        )
        return row

    @app.post(f"{API_PREFIX}/study")
    async def analyze_study(request: Request, file: Annotated[UploadFile, File()]) -> dict[str, Any]:
        job = await accept(request, file, JobKind.STUDY)
        future = store.start(job)
        try:
            await run_in_threadpool(future.result, settings.api.study_timeout_s)
        except TimeoutError as exc:
            raise HTTPException(status.HTTP_504_GATEWAY_TIMEOUT, "Превышено время обработки") from exc
        if job.status is not JobStatus.DONE or not job.rows:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT, job.error or "Файл не удалось обработать"
            )
        return {**job.rows[0], "job_id": job.id}

    return app
