from __future__ import annotations

import contextlib
import io
import logging
import time
import zipfile
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from dxa_qc.api.app import create_app
from dxa_qc.api.jobs import JobKind
from dxa_qc.config import Settings
from dxa_qc.domain import Violation
from dxa_qc.pipeline import ImageProcessor, QualityAssessment
from tests.factories import make_dicom
from tests.unit.test_pipeline import FakeRegionModel, FixedAssessor
from tests.unit.test_quality_geometry import synthetic_spine

SPINE_PIXELS = (synthetic_spine(4.0, h=200, w=120) * 255).astype(np.uint8)
BAD = QualityAssessment(
    1,
    0.9,
    frozenset({Violation.SPINE_AXIS}),
    criterion_probs={Violation.SPINE_AXIS: 0.8, Violation.POSITIONING: 0.1, Violation.FOREIGN_OBJECTS: 0.2},
    findings={"axis_tilt_deg": 4.0},
)


def dicom_bytes(tmp_path: Path, name: str = "x.dcm") -> bytes:
    return make_dicom(tmp_path / "src" / name, SPINE_PIXELS).read_bytes()


def zip_bytes(tmp_path: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("study/a.dcm", dicom_bytes(tmp_path, "a.dcm"))
        zf.writestr("study/broken.dcm", b"not a dicom")
    return buf.getvalue()


def make_client(tmp_path: Path, settings: Settings, **overrides) -> TestClient:
    api = settings.api.model_copy(update=overrides)
    s = settings.model_copy(update={"api": api})
    app = create_app(
        settings=s,
        data_dir=tmp_path / "jobs",
        processor_factory=lambda: ImageProcessor(s, FakeRegionModel(), FixedAssessor(BAD)),
    )
    return TestClient(app)


def wait_done(client: TestClient, job_id: str, timeout: float = 20.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/api/v1/batch/{job_id}/status").json()
        if body["status"] in ("done", "failed"):
            return body
        time.sleep(0.05)
    raise AssertionError("job did not finish")


@pytest.fixture
def client(tmp_path, settings):
    with make_client(tmp_path, settings) as c:
        yield c


def test_health_reports_models_loaded(client):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not client.get("/api/v1/health").json()["models_ready"]:
        time.sleep(0.05)
    assert client.get("/api/v1/health").json() == {"status": "ok", "models_ready": True, "models_error": None}


def test_batch_lifecycle(client, tmp_path):
    r = client.post("/api/v1/batch", files={"file": ("batch.zip", zip_bytes(tmp_path), "application/zip")})
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    status = wait_done(client, job_id)
    assert status["status"] == "done" and status["progress"] == 100.0
    assert status["processed"] == status["total"] == 2

    rows = client.get(f"/api/v1/batch/{job_id}/results").json()["rows"]
    ok, bad = rows
    assert ok["processing_status"] == "Success"
    assert ok["violation_type"] == Violation.SPINE_AXIS.value
    assert "Наклон оси позвоночника: 4,0°" in ok["explanation"]
    assert {c["violation"] for c in ok["criteria"]} == {v.value for v in BAD.criterion_probs}
    assert bad["processing_status"] == "Failure"
    assert "DICOM" in bad["error_reason"]

    for url in (ok["preview_url"], ok["overlay_url"]):
        img = client.get(url)
        assert img.status_code == 200 and img.content.startswith(b"\x89PNG")

    archive = client.get(f"/api/v1/batch/{job_id}/result")
    assert archive.headers["content-type"] == "application/zip"
    names = set(zipfile.ZipFile(io.BytesIO(archive.content)).namelist())
    assert {"results.xlsx", "results.csv", "results_errors.csv", "results_details.csv"} <= names
    assert any(n.startswith("visualization/") for n in names)


def test_study_endpoint_returns_row_synchronously(client, tmp_path):
    r = client.post("/api/v1/study", files={"file": ("one.dcm", dicom_bytes(tmp_path), "application/dicom")})
    assert r.status_code == 200
    body = r.json()
    assert body["quality_class"] == 1 and body["preview_url"].endswith("/preview.png")
    assert client.get(body["overlay_url"]).status_code == 200


def test_study_rejects_zip_and_garbage(client, tmp_path):
    assert client.post("/api/v1/study", files={"file": ("a.zip", zip_bytes(tmp_path))}).status_code == 415
    assert client.post("/api/v1/batch", files={"file": ("a.txt", b"hello world")}).status_code == 415
    assert client.post("/api/v1/batch", files={"file": ("e.zip", b"")}).status_code == 400


def test_upload_size_limit(tmp_path, settings):
    with make_client(tmp_path, settings, max_upload_bytes=1000) as c:
        r = c.post("/api/v1/batch", files={"file": ("big.zip", b"PK\x03\x04" + b"0" * 5000)})
        assert r.status_code == 413
        assert list((tmp_path / "jobs").iterdir()) == []  # rejected upload leaves nothing behind


def test_unsafe_archive_fails_job_not_server(client):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("../../evil.dcm", b"x")
    job_id = client.post("/api/v1/batch", files={"file": ("evil.zip", buf.getvalue())}).json()["job_id"]
    status = wait_done(client, job_id)
    assert status["status"] == "failed" and "небезопасный" in status["error"]
    assert client.get(f"/api/v1/batch/{job_id}/results").status_code == 409


def test_invalid_ids_and_paths_are_rejected(client):
    assert client.get("/api/v1/batch/../../etc/status").status_code == 404
    assert client.get("/api/v1/batch/not-a-uuid/status").status_code == 422
    assert client.get(f"/api/v1/batch/{'0' * 32}/status").status_code == 404
    assert client.get(f"/api/v1/batch/{'0' * 32}/images/0/secret.png").status_code == 404


def test_queue_limit(tmp_path, settings, monkeypatch):
    with make_client(tmp_path, settings, max_queued_jobs=1) as c:
        store = c.app.state.store
        monkeypatch.setattr(store, "start", lambda job: None)  # keep the first job queued
        first = c.post("/api/v1/batch", files={"file": ("a.zip", zip_bytes(tmp_path))})
        assert first.status_code == 202
        second = c.post("/api/v1/batch", files={"file": ("b.zip", zip_bytes(tmp_path))})
        assert second.status_code == 429
        assert second.headers["retry-after"].isdigit()


def test_expired_jobs_are_deleted(client, tmp_path):
    from datetime import timedelta

    job_id = client.post("/api/v1/batch", files={"file": ("a.zip", zip_bytes(tmp_path))}).json()["job_id"]
    wait_done(client, job_id)
    store = client.app.state.store
    store.get(job_id).updated_at -= timedelta(days=30)
    store.cleanup()
    assert client.get(f"/api/v1/batch/{job_id}/status").status_code == 404
    assert not (tmp_path / "jobs" / job_id).exists()


def test_upload_name_cannot_escape_job_dir(client, tmp_path):
    r = client.post("/api/v1/batch", files={"file": ("../../../evil.zip", zip_bytes(tmp_path))})
    job = client.app.state.store.get(r.json()["job_id"])
    assert job.upload_path.parent.parent == job.directory
    wait_done(client, job.id)


# ---- denial-of-service guards -------------------------------------------------------------------


def _multipart_chunks(payload_size: int, boundary: str = "XBOUNDARY"):
    """Multipart body streamed in chunks: httpx sends it with Transfer-Encoding: chunked,
    i.e. without Content-Length - the case an endpoint-level size check cannot catch."""
    yield (
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="big.zip"\r\n'
        "Content-Type: application/zip\r\n\r\n"
    ).encode()
    yield b"PK\x03\x04"
    sent = 4
    while sent < payload_size:
        chunk = b"0" * min(64 * 1024, payload_size - sent)
        sent += len(chunk)
        yield chunk
    yield f"\r\n--{boundary}--\r\n".encode()


def test_chunked_upload_over_limit_is_cut_off(tmp_path, settings):
    with make_client(tmp_path, settings, max_upload_bytes=100_000) as c:
        r = c.post(
            "/api/v1/batch",
            content=_multipart_chunks(2_000_000),
            headers={"content-type": "multipart/form-data; boundary=XBOUNDARY"},
        )
        assert r.status_code == 413
        assert "Файл больше" in r.json()["detail"]
        assert c.app.state.store.storage_used() == 0


def test_declared_or_malformed_content_length_rejected_early(tmp_path, settings):
    with make_client(tmp_path, settings, max_upload_bytes=100_000) as c:
        headers = {"content-type": "multipart/form-data; boundary=X"}
        huge = c.post("/api/v1/batch", content=b"x", headers={**headers, "content-length": str(10**12)})
        assert huge.status_code == 413


def test_upload_within_limit_still_works_with_chunked_body(tmp_path, settings):
    with make_client(tmp_path, settings) as c:
        body = zip_bytes(tmp_path)

        def chunks():
            yield b'--B\r\nContent-Disposition: form-data; name="file"; filename="a.zip"\r\n\r\n'
            yield body
            yield b"\r\n--B--\r\n"

        headers = {"content-type": "multipart/form-data; boundary=B"}
        r = c.post("/api/v1/batch", content=chunks(), headers=headers)
        assert r.status_code == 202
        assert wait_done(c, r.json()["job_id"])["status"] == "done"


def test_pending_study_limit(tmp_path, settings, monkeypatch):
    """/study must not accept unbounded concurrent requests: each one holds a server thread."""
    with make_client(tmp_path, settings, max_pending_studies=1) as c:
        store = c.app.state.store
        blocked = store.new_job(JobKind.STUDY, "held.dcm")  # an in-flight study
        r = c.post("/api/v1/study", files={"file": ("one.dcm", dicom_bytes(tmp_path))})
        assert r.status_code == 429
        store.discard(blocked)
        assert c.post("/api/v1/study", files={"file": ("one.dcm", dicom_bytes(tmp_path))}).status_code == 200


def test_storage_quota_and_free_disk(tmp_path, settings, monkeypatch):
    import shutil
    from collections import namedtuple

    with make_client(tmp_path, settings, max_storage_bytes=1) as c:
        job_id = c.post("/api/v1/batch", files={"file": ("a.zip", zip_bytes(tmp_path))}).json()["job_id"]
        wait_done(c, job_id)
        r = c.post("/api/v1/batch", files={"file": ("b.zip", zip_bytes(tmp_path))})
        assert r.status_code == 507
        assert r.headers["retry-after"].isdigit()

    Usage = namedtuple("Usage", "total used free")
    with make_client(tmp_path / "2", settings, min_free_disk_bytes=10**9) as c:
        monkeypatch.setattr(shutil, "disk_usage", lambda _: Usage(10**10, 10**10, 10**6))
        assert c.post("/api/v1/batch", files={"file": ("a.zip", zip_bytes(tmp_path))}).status_code == 507


def test_health_does_not_leak_internal_paths(tmp_path, settings):
    def broken_factory():
        raise FileNotFoundError("/secret/path/weights/region_classifier.npz")

    app = create_app(settings=settings, data_dir=tmp_path / "jobs", processor_factory=broken_factory)
    with TestClient(app) as c:
        deadline = time.monotonic() + 5
        body = c.get("/api/v1/health").json()
        while body["models_error"] is None and time.monotonic() < deadline:
            time.sleep(0.05)
            body = c.get("/api/v1/health").json()
        assert body["models_ready"] is False
        assert body["models_error"] and "/secret" not in body["models_error"]


def test_docs_can_be_disabled(tmp_path, settings, monkeypatch):
    from dxa_qc.api.app import DOCS_ENV

    monkeypatch.setenv(DOCS_ENV, "0")
    with make_client(tmp_path, settings) as c:
        assert c.get("/api/v1/docs").status_code == 404
        assert c.get("/api/v1/openapi.json").status_code == 404


def test_real_server_aborts_oversized_stream_early(tmp_path, settings):
    """End-to-end over a socket: the 413 arrives while the client is still sending, and the
    server has written far less than the attacker sent (TestClient buffers bodies, so it
    cannot show this)."""
    import socket
    import threading

    import uvicorn

    api = settings.api.model_copy(update={"max_upload_bytes": 200_000})
    s = settings.model_copy(update={"api": api})
    app = create_app(
        settings=s,
        data_dir=tmp_path / "jobs",
        processor_factory=lambda: ImageProcessor(s, FakeRegionModel(), FixedAssessor(BAD)),
    )
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning", lifespan="on"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.05)

        client = socket.create_connection(("127.0.0.1", port), timeout=10)
        client.sendall(
            b"POST /api/v1/batch HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n"
            b"Content-Type: multipart/form-data; boundary=B\r\n\r\n"
        )
        head = b'--B\r\nContent-Disposition: form-data; name="file"; filename="a.zip"\r\n\r\nPK\x03\x04'
        client.sendall(f"{len(head):x}\r\n".encode() + head + b"\r\n")
        sent, response = 0, b""
        chunk = b"0" * 65536
        client.setblocking(False)
        while sent < 50_000_000:
            try:
                client.sendall(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                sent += len(chunk)
            except (BlockingIOError, ConnectionError, OSError):
                pass
            try:
                response += client.recv(4096)
            except BlockingIOError:
                continue
            except OSError:
                break
            if response:
                break
        if not response:  # server closed while we were sending: the response is already buffered
            client.setblocking(True)
            client.settimeout(5)
            with contextlib.suppress(OSError):
                response = client.recv(4096)
        client.close()
        # The guarantee is that the server stops reading; the 413 itself may be lost when the
        # peer resets the socket with unread data (Windows does), so it is checked only if seen.
        assert not response or response.startswith(b"HTTP/1.1 413"), response[:100]
        assert sent < 50_000_000, "server kept reading the whole oversized body"
        from dxa_qc.api.jobs import directory_size

        assert directory_size(tmp_path / "jobs") < 1_000_000
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def test_access_to_patient_data_is_audited_and_not_cached(client, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="dxa_qc.audit")
    job_id = client.post(
        "/api/v1/batch", files={"file": ("batch.zip", zip_bytes(tmp_path))}, headers={"X-Real-IP": "10.1.2.3"}
    ).json()["job_id"]
    wait_done(client, job_id)
    rows = client.get(f"/api/v1/batch/{job_id}/results").json()["rows"]
    client.get(f"/api/v1/batch/{job_id}/result")
    img = client.get(rows[0]["preview_url"])
    assert img.headers["cache-control"] == "no-store"
    events = [r.getMessage() for r in caplog.records if r.name == "dxa_qc.audit"]
    assert f"upload job={job_id} kind=batch client=10.1.2.3" in events
    assert any(e.startswith(f"view-results job={job_id}") for e in events)
    assert any(e.startswith(f"download-archive job={job_id}") for e in events)


def test_error_report_does_not_leak_server_paths(client, tmp_path):
    job_id = client.post("/api/v1/batch", files={"file": ("batch.zip", zip_bytes(tmp_path))}).json()["job_id"]
    wait_done(client, job_id)
    archive = zipfile.ZipFile(io.BytesIO(client.get(f"/api/v1/batch/{job_id}/result").content))
    errors = archive.read("results_errors.csv").decode("utf-8")
    data_dir = client.app.state.store.data_dir
    assert "not a valid DICOM" in errors
    assert str(data_dir) not in errors and data_dir.as_posix() not in errors and job_id not in errors


def _done_batch(client, tmp_path) -> tuple[str, list[dict]]:
    r = client.post("/api/v1/batch", files={"file": ("batch.zip", zip_bytes(tmp_path), "application/zip")})
    job_id = r.json()["job_id"]
    assert wait_done(client, job_id)["status"] == "done"
    return job_id, client.get(f"/api/v1/batch/{job_id}/results").json()["rows"]


def test_rows_carry_projection_corrections_and_dicom_series(client, tmp_path):
    import pydicom

    job_id, (ok, bad) = _done_batch(client, tmp_path)
    assert ok["projection"] == "Прямая (AP)"
    assert [c["violation"] for c in ok["corrections"]] == [Violation.SPINE_AXIS.value]
    assert ok["review"] is None
    assert bad["corrections"] == [] and "sr_url" not in bad

    sr = client.get(ok["sr_url"])
    assert sr.status_code == 200 and sr.headers["content-type"] == "application/dicom"
    assert sr.headers["cache-control"] == "no-store"
    ds = pydicom.dcmread(io.BytesIO(sr.content))
    assert ds.Modality == "SR" and ds.VerificationFlag == "UNVERIFIED"
    sc = pydicom.dcmread(io.BytesIO(client.get(ok["sc_url"]).content))
    assert sc.StudyInstanceUID == ds.StudyInstanceUID and sc.SamplesPerPixel == 3

    names = set(zipfile.ZipFile(io.BytesIO(client.get(f"/api/v1/batch/{job_id}/result").content)).namelist())
    assert {"dicom/00000_sr.dcm", "dicom/00000_sc.dcm"} <= names
    assert client.get(f"/api/v1/batch/{job_id}/dicom/0/xx.dcm").status_code == 404
    assert client.get(f"/api/v1/batch/{job_id}/dicom/1/sr.dcm").status_code == 404


def test_specialist_review_verifies_sr_and_updates_archive(client, tmp_path):
    import csv

    import pydicom

    job_id, (ok, _) = _done_batch(client, tmp_path)
    old_uid = pydicom.dcmread(io.BytesIO(client.get(ok["sr_url"]).content)).SOPInstanceUID
    client.get(f"/api/v1/batch/{job_id}/result")  # cache an archive built before the review

    url = f"/api/v1/batch/{job_id}/rows/0/review"
    r = client.post(
        url,
        json={
            "decision": "corrected",
            "quality_class": 1,
            "violations": [Violation.FOREIGN_OBJECTS.value, Violation.SPINE_AXIS.value],
            "comment": "=cmd| артефакт",
        },
    )
    assert r.status_code == 200, r.text
    review = r.json()["review"]
    assert review["decision"] == "corrected"
    # Canonical criteria order regardless of the order sent.
    assert review["violation_type"] == f"{Violation.SPINE_AXIS.value};{Violation.FOREIGN_OBJECTS.value}"
    assert client.get(f"/api/v1/batch/{job_id}/results").json()["rows"][0]["review"] == review

    sr = pydicom.dcmread(io.BytesIO(client.get(ok["sr_url"]).content))
    assert sr.VerificationFlag == "VERIFIED" and sr.SOPInstanceUID != old_uid

    zf = zipfile.ZipFile(io.BytesIO(client.get(f"/api/v1/batch/{job_id}/result").content))
    rows = list(csv.DictReader(io.StringIO(zf.read("results_reviewed.csv").decode("utf-8-sig"))))
    assert rows[0]["review_decision"] == "corrected" and rows[0]["review_quality_class"] == "1"
    assert rows[0]["review_comment"].startswith("'=")  # formula injection neutralized
    assert rows[1]["review_decision"] == ""  # not reviewed (failed row)

    confirmed = client.post(url, json={"decision": "confirmed"}).json()["review"]
    assert confirmed["quality_class"] == 1 and confirmed["violation_type"] == Violation.SPINE_AXIS.value


@pytest.mark.parametrize(
    ("body", "code"),
    [
        ({"decision": "corrected", "violations": []}, 422),  # no class
        ({"decision": "corrected", "quality_class": 1, "violations": ["Некорректная область интереса"]}, 422),
        ({"decision": "corrected", "quality_class": 0, "violations": [Violation.SPINE_AXIS.value]}, 422),
        ({"decision": "maybe"}, 422),
        ({"decision": "confirmed", "comment": "x" * 1001}, 422),
        ({"decision": "confirmed", "extra": 1}, 422),
    ],
)
def test_review_rejects_inconsistent_decisions(client, tmp_path, body, code):
    job_id, _ = _done_batch(client, tmp_path)
    assert client.post(f"/api/v1/batch/{job_id}/rows/0/review", json=body).status_code == code


def test_review_of_failed_or_missing_row(client, tmp_path):
    job_id, _ = _done_batch(client, tmp_path)
    failed = client.post(f"/api/v1/batch/{job_id}/rows/1/review", json={"decision": "confirmed"})
    assert failed.status_code == 422 and "ошибкой обработки" in failed.json()["detail"]
    assert (
        client.post(f"/api/v1/batch/{job_id}/rows/7/review", json={"decision": "confirmed"}).status_code
        == 404
    )
    assert (
        client.post(f"/api/v1/batch/{'0' * 32}/rows/0/review", json={"decision": "confirmed"}).status_code
        == 404
    )


def test_review_accepts_only_json_content_type(client, tmp_path):
    # CSRF: a cross-site form or text/plain POST needs no CORS preflight and carries basic-auth
    # credentials automatically; the review must only be accepted as application/json.
    job_id, _ = _done_batch(client, tmp_path)
    url = f"/api/v1/batch/{job_id}/rows/0/review"
    for content_type in ("text/plain", "application/x-www-form-urlencoded"):
        r = client.post(url, content='{"decision": "confirmed"}', headers={"Content-Type": content_type})
        assert r.status_code == 422
    assert client.get(f"/api/v1/batch/{job_id}/results").json()["rows"][0]["review"] is None
