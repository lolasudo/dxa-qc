"""Frozen ImageNet/DINOv2 backbones used as feature extractors.

Why frozen + linear heads, not fine-tuning: with 99 spine / 150 hip labeled images, grouped CV
showed fine-tuned ResNet18 / ConvNeXt-T below frozen features + logistic heads (hip positioning
AUC 0.61-0.65 vs 0.69), and linear heads are far less prone to memorising label noise.

Weights are read only from local ``.safetensors`` files (no pickle, no network at inference -
offline deployment). ``export_backbones`` downloads them once, at build time.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class BackboneError(RuntimeError):
    """Backbone weights are missing/corrupted or the ML stack is not installed."""


@dataclass(frozen=True, slots=True)
class Crop:
    """Fractions of the canonical isotropic image: rows [top, bottom), columns [left, right)."""

    top: float = 0.0
    bottom: float = 1.0
    left: float = 0.0
    right: float = 1.0

    def __post_init__(self) -> None:
        if not (0 <= self.top < self.bottom <= 1 and 0 <= self.left < self.right <= 1):
            raise ValueError(f"invalid crop {self}")

    def apply(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape
        out = image[
            int(self.top * h) : max(int(self.bottom * h), int(self.top * h) + 1),
            int(self.left * w) : max(int(self.right * w), int(self.left * w) + 1),
        ]
        return out


FULL = Crop()


@dataclass(frozen=True, slots=True)
class BackboneSpec:
    key: str  # stable id used in model files
    timm_name: str
    input_size: int
    crop: Crop = FULL

    @property
    def weights_file(self) -> str:
        return f"{self.timm_name}.safetensors"

    def to_dict(self) -> dict[str, object]:
        c = self.crop
        return {
            "key": self.key,
            "timm": self.timm_name,
            "input_size": self.input_size,
            "crop": [c.top, c.bottom, c.left, c.right],
        }

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> BackboneSpec:
        """Inverse of :meth:`to_dict`; the timm name becomes a file name, so it is validated."""
        import re

        timm_name = str(d["timm"])
        if not re.fullmatch(r"[A-Za-z0-9_.\-]{1,128}", timm_name) or ".." in timm_name:
            raise ValueError(f"invalid backbone name {timm_name!r}")
        crop = d.get("crop") or [0.0, 1.0, 0.0, 1.0]
        return cls(str(d["key"]), timm_name, int(d["input_size"]), Crop(*(float(v) for v in crop)))  # type: ignore[arg-type]


# Chosen by grouped CV (see backend/README.md, "Модели качества"); a spec change requires retraining.
EFFNET_B3 = BackboneSpec("effb3_full", "efficientnet_b3.ra2_in1k", 288)
DINOV2_S = BackboneSpec("dinov2s_full", "vit_small_patch14_dinov2.lvd142m", 448)
CONVNEXT_S = BackboneSpec("convnexts_full", "convnext_small.fb_in22k_ft_in1k", 224)
# Proximal shaft + neck band, where the lesser trochanter projects.
DINOV2_S_TROCHANTER = BackboneSpec(
    "dinov2s_trochanter", "vit_small_patch14_dinov2.lvd142m", 448, Crop(0.35, 0.8, 0.15, 0.75)
)
ALL_SPECS: dict[str, BackboneSpec] = {
    s.key: s for s in (EFFNET_B3, DINOV2_S, CONVNEXT_S, DINOV2_S_TROCHANTER)
}


def square_input(image: np.ndarray, crop: Crop, size: int) -> np.ndarray:
    """Crop, zero-pad to a centred square (keeps physical aspect), resize to ``size``."""
    part = crop.apply(image).astype(np.float32)
    h, w = part.shape
    side = max(h, w)
    canvas = np.zeros((side, side), np.float32)
    y0, x0 = (side - h) // 2, (side - w) // 2
    canvas[y0 : y0 + h, x0 : x0 + w] = part
    interpolation = cv2.INTER_AREA if side >= size else cv2.INTER_LINEAR
    return cv2.resize(canvas, (size, size), interpolation=interpolation)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pin_numerics(torch: Any) -> None:
    """Reproducible embeddings, close to the CPU ones the thresholds were fitted on.

    On Ampere and newer GPUs PyTorch runs convolutions in TF32 and lets cuDNN pick
    non-deterministic kernels by default (~2e-3 relative drift from CPU); with both off it is ~2e-6.
    """
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("highest")


class EmbeddingExtractor:
    """Lazily builds each backbone once; thread-unsafe by design (one extractor per worker)."""

    def __init__(self, weights_dir: Path, device: str | None = None) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - depends on the installed extras
            raise BackboneError("quality models need the ML extras: pip install -e .[ml]") from exc
        self._torch = torch
        self.weights_dir = Path(weights_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._models: dict[str, tuple[object, object, object]] = {}
        pin_numerics(torch)

    def _model(self, timm_name: str, input_size: int) -> tuple[object, object, object]:
        cache_key = f"{timm_name}@{input_size}"
        if cache_key in self._models:
            return self._models[cache_key]
        import timm
        from safetensors.torch import load_file

        path = self.weights_dir / f"{timm_name}.safetensors"
        if not path.is_file():
            raise BackboneError(
                f"backbone weights not found: {path} (run python -m dxa_qc.training.backbones)"
            )
        try:
            model = timm.create_model(
                timm_name, pretrained=False, num_classes=0, **_create_kwargs(timm_name, input_size)
            )
            state = load_file(str(path), device="cpu")
            model.load_state_dict(state, strict=True)
        except Exception as exc:  # safetensors/timm raise their own unrelated error types
            raise BackboneError(f"cannot load backbone {timm_name} from {path}: {exc}") from exc
        model.eval().to(self.device)
        cfg = model.pretrained_cfg
        torch = self._torch
        mean = torch.tensor(cfg["mean"], dtype=torch.float32).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor(cfg["std"], dtype=torch.float32).view(1, 3, 1, 1).to(self.device)
        self._models[cache_key] = (model, mean, std)
        return self._models[cache_key]

    def preload(self, spec: BackboneSpec) -> None:
        """Build the backbone now (startup) instead of on the first image."""
        self._model(spec.timm_name, spec.input_size)

    def embed(self, images: Sequence[np.ndarray], spec: BackboneSpec, batch_size: int = 16) -> np.ndarray:
        """``images``: canonical isotropic images in [0, 1]. Returns ``(n, dim)`` float64."""
        torch = self._torch
        model, mean, std = self._model(spec.timm_name, spec.input_size)
        out: list[np.ndarray] = []
        with torch.inference_mode():
            for i in range(0, len(images), batch_size):
                batch = np.stack(
                    [square_input(im, spec.crop, spec.input_size) for im in images[i : i + batch_size]]
                )
                x = torch.from_numpy(batch)[:, None].expand(-1, 3, -1, -1).to(self.device)
                out.append(model((x - mean) / std).float().cpu().numpy())  # type: ignore[operator]
        result = np.concatenate(out).astype(np.float64)
        if not np.isfinite(result).all():
            raise BackboneError(f"non-finite embedding from {spec.key}")
        return result


def _create_kwargs(timm_name: str, input_size: int) -> dict[str, int]:
    # ViTs bake the input size into the positional embedding; the stored weights are exported
    # already resampled to it, so export and load must agree on this.
    return {"img_size": input_size} if "dinov2" in timm_name else {}


def export_backbones(weights_dir: Path, specs: Sequence[BackboneSpec]) -> dict[str, str]:
    """Download pretrained weights once (needs network) and store as safetensors; returns sha256s."""
    import timm
    from safetensors.torch import save_file

    weights_dir = Path(weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    sizes: dict[str, int] = {}
    for spec in specs:
        if sizes.setdefault(spec.timm_name, spec.input_size) != spec.input_size:
            raise ValueError(f"{spec.timm_name} is used with different input sizes; weights file is per name")
    hashes: dict[str, str] = {}
    for timm_name, size in sorted(sizes.items()):
        path = weights_dir / f"{timm_name}.safetensors"
        if not path.is_file():
            model = timm.create_model(
                timm_name, pretrained=True, num_classes=0, **_create_kwargs(timm_name, size)
            )
            state = {k: v.contiguous() for k, v in model.state_dict().items()}
            save_file(state, str(path))
        hashes[timm_name] = file_sha256(path)
    return hashes
