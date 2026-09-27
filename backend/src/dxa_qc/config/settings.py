"""Typed, validated runtime configuration loaded from ``configs/*.yaml``.

Fails fast on a malformed config: a silently wrong threshold would corrupt every
prediction without any visible error.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

# Overridable so the container can mount configs read-only at any path.
CONFIG_DIR_ENV = "DXA_QC_CONFIG_DIR"
_DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"


class ConfigError(RuntimeError):
    """Configuration is missing or invalid."""


class _Strict(BaseModel):
    # Unknown keys are rejected so a typo in YAML doesn't silently fall back to a default.
    model_config = ConfigDict(extra="forbid", frozen=True)


class PixelSpacingConfig(_Strict):
    row_mm: float = Field(gt=0, le=10)
    column_mm: float = Field(gt=0, le=10)


class SpineThresholds(_Strict):
    max_axis_tilt_deg: float = Field(gt=0, lt=90)


class HipThresholds(_Strict):
    roi_margin_vertical_cm: float = Field(gt=0)
    roi_margin_lateral_cm: float = Field(gt=0)


class RegionConfig(_Strict):
    min_confidence: float = Field(ge=0.0, lt=1.0)


class ReportConfig(_Strict):
    violation_separator: str = Field(min_length=1, max_length=3)


class IngestLimits(_Strict):
    max_file_bytes: int = Field(gt=0)
    max_archive_members: int = Field(gt=0)
    max_archive_uncompressed_bytes: int = Field(gt=0)
    max_compression_ratio: float = Field(gt=1)
    # DXA frames are ~300 px; anything far larger is either not DXA or an attempt to burn CPU/RAM
    # (geometry filters are O(pixels) on one worker thread).
    max_image_side: int = Field(default=4096, ge=64, le=16384)
    # Plausible physical field extent per side; also bounds the isotropic resampling, so a forged
    # PixelSpacing tag cannot blow a 4k image up to 80k pixels.
    min_physical_mm: float = Field(default=10.0, gt=0)
    max_physical_mm: float = Field(default=2500.0, gt=0)
    # Extraction is refused if it would leave less than this much free disk space.
    min_free_disk_bytes: int = Field(default=2 * 1024**3, ge=0)


class ApiConfig(_Strict):
    max_upload_bytes: int = Field(gt=0)
    max_queued_jobs: int = Field(gt=0, le=1000)
    # /study waits synchronously; unbounded, a request flood would pin every server thread.
    max_pending_studies: int = Field(default=4, gt=0, le=100)
    job_ttl_hours: float = Field(gt=0)
    study_timeout_s: float = Field(gt=0)
    # Total disk used by stored jobs (uploads, reports, previews); new uploads get HTTP 507 beyond it.
    max_storage_bytes: int = Field(default=20 * 1024**3, gt=0)
    # Uploads are refused while the data volume has less free space than this.
    min_free_disk_bytes: int = Field(default=5 * 1024**3, ge=0)
    # uvicorn --limit-concurrency: connections beyond this get HTTP 503 instead of exhausting the server.
    max_connections: int = Field(default=64, gt=0, le=10000)


class Settings(_Strict):
    pixel_spacing: PixelSpacingConfig
    spine: SpineThresholds
    hip: HipThresholds
    region: RegionConfig
    report: ReportConfig
    ingest: IngestLimits
    api: ApiConfig


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    try:
        # safe_load: config files must never be able to instantiate arbitrary Python objects.
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"Top level of {path} must be a mapping")
    return data


def load_settings(config_dir: Path | None = None) -> Settings:
    """Load and validate settings from ``config_dir`` (env var / repo default otherwise)."""
    directory = Path(config_dir or os.environ.get(CONFIG_DIR_ENV) or _DEFAULT_CONFIG_DIR)
    thresholds = _read_yaml(directory / "thresholds.yaml")
    merged = {
        **thresholds,
        **_read_yaml(directory / "service.yaml"),
        "pixel_spacing": _read_yaml(directory / "pixel_spacing.yaml"),
    }
    try:
        return Settings.model_validate(merged)
    except ValidationError as exc:
        raise ConfigError(f"Invalid configuration in {directory}:\n{exc}") from exc
