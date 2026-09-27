"""Typed training configuration (``configs/training.yaml``); unknown keys are errors."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from dxa_qc.config.settings import CONFIG_DIR_ENV
from dxa_qc.domain.taxonomy import AnatomicalRegion
from dxa_qc.models.quality.embeddings import BackboneSpec, Crop

DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "training.yaml"


def default_config_path() -> Path:
    """``$DXA_QC_CONFIG_DIR/training.yaml`` (installed package, e.g. Docker), else the repo's configs."""
    directory = os.environ.get(CONFIG_DIR_ENV)
    return Path(directory) / "training.yaml" if directory else DEFAULT_CONFIG


class TrainingConfigError(ValueError):
    pass


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CVConfig(_Strict):
    n_splits: int = Field(5, ge=2, le=20)
    repetitions: int = Field(20, ge=1, le=200)
    bootstrap: int = Field(1000, ge=0, le=100_000)


class HeadsConfig(_Strict):
    geometry_c: tuple[float, ...] = (0.1,)
    embedding_c: tuple[float, ...] = (0.01,)
    inner_splits: int = Field(3, ge=2, le=10)

    @field_validator("geometry_c", "embedding_c")
    @classmethod
    def _positive(cls, v: tuple[float, ...]) -> tuple[float, ...]:
        if not v or any(c <= 0 for c in v):
            raise ValueError("C values must be a non-empty list of positive numbers")
        return v


class EmbeddingConfig(_Strict):
    batch_size: int = Field(16, ge=1, le=1024)
    device: str = "auto"
    cache_dir: Path | None = Path("var/embedding_cache")


class BackboneConfig(_Strict):
    key: str = Field(pattern=r"^[a-z0-9_]{1,64}$")
    timm: str = Field(pattern=r"^[A-Za-z0-9_.\-]{1,128}$")  # also a file name: no path characters
    input_size: int = Field(ge=32, le=1024)
    crop: tuple[float, float, float, float] | None = None

    @model_validator(mode="after")
    def _valid_crop(self) -> BackboneConfig:
        self.spec()  # Crop validates its fractions: fail at load time, not mid-training
        return self

    def spec(self) -> BackboneSpec:
        crop = Crop(*self.crop) if self.crop else Crop()
        return BackboneSpec(self.key, self.timm, self.input_size, crop)


class RegionConfig(_Strict):
    backbones: tuple[BackboneConfig, ...] = Field(min_length=0, max_length=8)


class TrainingConfig(_Strict):
    seed: int = 20260924
    cv: CVConfig = CVConfig()
    heads: HeadsConfig = HeadsConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    regions: dict[str, RegionConfig]

    @field_validator("regions")
    @classmethod
    def _known_regions(cls, v: dict[str, RegionConfig]) -> dict[str, RegionConfig]:
        names = {r.name.lower() for r in AnatomicalRegion}
        if set(v) != names:
            raise ValueError(f"regions must be exactly {sorted(names)}")
        for region in v.values():
            keys = [b.key for b in region.backbones]
            if len(keys) != len(set(keys)):
                raise ValueError("backbone keys must be unique within a region")
        return v

    def backbones(self, region: AnatomicalRegion) -> tuple[BackboneSpec, ...]:
        return tuple(b.spec() for b in self.regions[region.name.lower()].backbones)

    def all_backbones(self) -> list[BackboneSpec]:
        return [s for r in AnatomicalRegion for s in self.backbones(r)]


def load_training_config(path: Path | None = None) -> TrainingConfig:
    path = Path(path or default_config_path())
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return TrainingConfig.model_validate(data)
    except FileNotFoundError as exc:
        raise TrainingConfigError(f"training config not found: {path}") from exc
    except (yaml.YAMLError, ValidationError, TypeError) as exc:
        raise TrainingConfigError(f"invalid training config {path}:\n{exc}") from exc
