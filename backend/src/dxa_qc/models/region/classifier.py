"""Anatomical region + hip side classifier (spine / right hip / left hip), pixels only.

Why not a CNN: on the verified annotations HOG + logistic regression
reaches 100% in study-grouped 5-fold CV, runs in milliseconds on CPU and needs no GPU stack.

Weights are stored as ``.npz`` (``allow_pickle=False``) + JSON metadata, never pickle/joblib:
loading a pickle executes arbitrary code, and weight files travel between machines.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np

from dxa_qc.domain.taxonomy import AnatomicalRegion, HipSide
from dxa_qc.models.region.features import HogParams, hog_features

FORMAT_VERSION = 1


class RegionClass(StrEnum):
    SPINE = "spine"
    HIP_RIGHT = "hip_right"
    HIP_LEFT = "hip_left"

    @property
    def region(self) -> AnatomicalRegion:
        return AnatomicalRegion.SPINE if self is RegionClass.SPINE else AnatomicalRegion.HIP

    @property
    def side(self) -> HipSide | None:
        return {RegionClass.HIP_RIGHT: HipSide.RIGHT, RegionClass.HIP_LEFT: HipSide.LEFT}.get(self)

    @property
    def mirrored(self) -> RegionClass:
        """Class of the horizontally flipped image: a flipped left hip looks exactly like a right hip."""
        return {RegionClass.HIP_RIGHT: RegionClass.HIP_LEFT, RegionClass.HIP_LEFT: RegionClass.HIP_RIGHT}.get(
            self, self
        )


class ModelFormatError(ValueError):
    """Weight file is missing, corrupted or incompatible."""


@dataclass(frozen=True, slots=True)
class RegionPrediction:
    label: RegionClass
    confidence: float
    probabilities: dict[RegionClass, float]

    @property
    def region(self) -> AnatomicalRegion:
        return self.label.region

    @property
    def side(self) -> HipSide | None:
        return self.label.side


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class RegionClassifier:
    """Standardize -> multinomial linear model -> softmax, with explicit, inspectable parameters."""

    def __init__(
        self,
        classes: Sequence[RegionClass],
        mean: np.ndarray,
        scale: np.ndarray,
        coef: np.ndarray,
        intercept: np.ndarray,
        hog_params: HogParams,
    ) -> None:
        n_features = mean.shape[0]
        if not (
            scale.shape == (n_features,)
            and coef.shape == (len(classes), n_features)
            and intercept.shape == (len(classes),)
        ):
            raise ModelFormatError(
                f"inconsistent shapes: mean {mean.shape}, scale {scale.shape}, "
                f"coef {coef.shape}, intercept {intercept.shape}, classes {len(classes)}"
            )
        if np.any(scale <= 0) or not all(np.isfinite(a).all() for a in (mean, scale, coef, intercept)):
            raise ModelFormatError("weights contain non-finite values or non-positive scales")
        self.classes = tuple(classes)
        self.mean, self.scale, self.coef, self.intercept = mean, scale, coef, intercept
        self.hog_params = hog_params

    # ---- training -------------------------------------------------------------------------

    @classmethod
    def fit(
        cls,
        images: Sequence[np.ndarray],
        labels: Sequence[RegionClass],
        *,
        c: float = 0.1,
        hog_params: HogParams | None = None,
        mirror_augment: bool = True,
        seed: int = 0,
    ) -> RegionClassifier:
        # Imported lazily: scikit-learn is a training-only dependency, not needed for inference.
        from sklearn.linear_model import LogisticRegression

        if len(images) != len(labels) or not images:
            raise ValueError("images and labels must be non-empty and of equal length")
        params = hog_params or HogParams()
        x = [hog_features(im, params) for im in images]
        y = [RegionClass(lbl) for lbl in labels]
        if mirror_augment:
            x += [hog_features(im[:, ::-1], params) for im in images]
            y += [lbl.mirrored for lbl in y]
        x_arr = np.stack(x)

        mean = x_arr.mean(axis=0)
        scale = x_arr.std(axis=0)
        # Constant features (e.g. always-empty HOG cells) would divide by zero.
        scale[scale < 1e-12] = 1.0
        model = LogisticRegression(C=c, max_iter=5000, random_state=seed)
        model.fit((x_arr - mean) / scale, [v.value for v in y])

        classes = [RegionClass(v) for v in model.classes_]
        coef, intercept = model.coef_, model.intercept_
        if len(classes) == 2:
            # sklearn stores a single row for binary problems; expand to the multinomial form.
            coef = np.vstack([-coef[0] / 2, coef[0] / 2])
            intercept = np.array([-intercept[0] / 2, intercept[0] / 2])
        return cls(classes, mean, scale, coef.astype(np.float64), intercept.astype(np.float64), params)

    # ---- inference ------------------------------------------------------------------------

    def predict_proba(self, pixels: np.ndarray) -> np.ndarray:
        x = (hog_features(pixels, self.hog_params) - self.mean) / self.scale
        if x.shape[0] != self.mean.shape[0]:
            raise ModelFormatError("feature size does not match the model")
        return _softmax(self.coef @ x + self.intercept)

    def predict(self, pixels: np.ndarray) -> RegionPrediction:
        proba = self.predict_proba(pixels)
        best = int(np.argmax(proba))
        return RegionPrediction(
            label=self.classes[best],
            confidence=float(proba[best]),
            probabilities={c: float(p) for c, p in zip(self.classes, proba, strict=True)},
        )

    # ---- persistence ----------------------------------------------------------------------

    def save(self, path: Path, metadata: dict[str, object] | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "format_version": FORMAT_VERSION,
            "classes": [c.value for c in self.classes],
            "hog_params": self.hog_params.to_dict(),
            **(metadata or {}),
        }
        np.savez(
            path,
            mean=self.mean,
            scale=self.scale,
            coef=self.coef,
            intercept=self.intercept,
            meta=np.array(json.dumps(meta, ensure_ascii=False)),
        )

    @classmethod
    def load(cls, path: Path) -> RegionClassifier:
        path = Path(path)
        try:
            with np.load(path, allow_pickle=False) as data:
                arrays = {
                    k: np.asarray(data[k], dtype=np.float64) for k in ("mean", "scale", "coef", "intercept")
                }
                meta = json.loads(str(data["meta"]))
        except FileNotFoundError as exc:
            raise ModelFormatError(f"weights not found: {path}") from exc
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise ModelFormatError(f"cannot load weights from {path}: {exc}") from exc
        if meta.get("format_version") != FORMAT_VERSION:
            raise ModelFormatError(f"unsupported weights format version: {meta.get('format_version')!r}")
        try:
            classes = [RegionClass(c) for c in meta["classes"]]
            hog_params = HogParams(**meta["hog_params"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelFormatError(f"invalid weights metadata: {exc}") from exc
        return cls(classes, hog_params=hog_params, **arrays)
