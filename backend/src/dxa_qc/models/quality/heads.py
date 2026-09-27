"""Linear heads and Platt calibration with explicit, inspectable parameters (no pickle)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_EPS = 1e-6


def logit(p: np.ndarray | float) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def sigmoid(z: np.ndarray | float) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    return np.where(z >= 0, 1 / (1 + np.exp(-np.abs(z))), np.exp(-np.abs(z)) / (1 + np.exp(-np.abs(z))))


def _l2(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


@dataclass(frozen=True, slots=True)
class LinearHead:
    """``[L2-normalize] -> standardize -> w*z + b``; output is a logit (class-balanced fit)."""

    source: str  # "geometry" or a backbone key
    feature_names: tuple[str, ...]  # geometry features used; empty for embeddings
    l2_normalize: bool
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float

    def __post_init__(self) -> None:
        d = self.mean.shape[0]
        if self.mean.ndim != 1 or self.scale.shape != (d,) or self.coef.shape != (d,):
            raise ValueError(
                f"inconsistent head shapes: {self.mean.shape}, {self.scale.shape}, {self.coef.shape}"
            )
        if self.feature_names and len(self.feature_names) != d:
            raise ValueError("feature_names length does not match the head dimension")
        if np.any(self.scale <= 0) or not all(
            np.isfinite(a).all() for a in (self.mean, self.scale, self.coef)
        ):
            raise ValueError("head parameters must be finite with positive scales")
        if not np.isfinite(self.intercept):
            raise ValueError("head intercept must be finite")

    @property
    def dim(self) -> int:
        return int(self.mean.shape[0])

    def _prepare(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        if x.shape[1] != self.dim:
            raise ValueError(f"head {self.source!r} expects {self.dim} features, got {x.shape[1]}")
        if self.l2_normalize:
            x = _l2(x)
        return (x - self.mean) / self.scale

    def decision(self, x: np.ndarray) -> np.ndarray:
        return self._prepare(x) @ self.coef + self.intercept

    @classmethod
    def fit(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        *,
        source: str,
        c: float,
        l2_normalize: bool,
        feature_names: tuple[str, ...] = (),
        seed: int = 0,
    ) -> LinearHead:
        from sklearn.linear_model import LogisticRegression  # training-only dependency

        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y).astype(int)
        if x.ndim != 2 or len(x) != len(y):
            raise ValueError("x must be 2-D with one row per label")
        if set(np.unique(y)) != {0, 1}:
            raise ValueError("both classes must be present to fit a head")
        if l2_normalize:
            x = _l2(x)
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale[scale < 1e-12] = 1.0
        model = LogisticRegression(C=c, class_weight="balanced", max_iter=10000, random_state=seed)
        model.fit((x - mean) / scale, y)
        return cls(
            source,
            tuple(feature_names),
            l2_normalize,
            mean,
            scale,
            model.coef_[0].astype(np.float64),
            float(model.intercept_[0]),
        )


@dataclass(frozen=True, slots=True)
class PlattCalibrator:
    """``p = sigmoid(a * score + b)``; fitted on out-of-fold scores, not on training fits."""

    a: float = 1.0
    b: float = 0.0

    def __post_init__(self) -> None:
        if not (np.isfinite(self.a) and np.isfinite(self.b)) or self.a <= 0:
            raise ValueError(f"invalid calibrator a={self.a}, b={self.b}: slope must be positive")

    def __call__(self, score: np.ndarray | float) -> np.ndarray:
        return sigmoid(self.a * np.asarray(score, dtype=np.float64) + self.b)

    @classmethod
    def fit(cls, scores: np.ndarray, y: np.ndarray) -> PlattCalibrator:
        from sklearn.linear_model import LogisticRegression

        scores = np.asarray(scores, dtype=np.float64).reshape(-1, 1)
        # Weak L2 keeps the slope finite on perfectly separable folds.
        model = LogisticRegression(C=100.0, max_iter=10000).fit(scores, np.asarray(y).astype(int))
        a = float(model.coef_[0, 0])
        if a <= 0:  # score carries no/negative signal: fall back to the prior, keep monotonicity
            prior = float(np.clip(np.mean(y), _EPS, 1 - _EPS))
            return cls(1e-6, float(logit(prior)))
        return cls(a, float(model.intercept_[0]))
