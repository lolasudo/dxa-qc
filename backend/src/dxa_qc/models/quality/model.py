"""Per-region quality model: criterion ensembles + calibrated overall probability.

Decision rule (fitted by ``dxa_qc.training.quality``; all decisions are made on raw scores):

* criterion score ``s_c`` = mean of member-head logits (equal weights: learned stacking weights
  were worse in nested CV - too few positives to estimate them); violated iff ``s_c >= t_c``;
* quality score ``q = (s_overall + max_c s_c) / 2`` - the expert's overall verdict is its own
  target («Итог» is not the OR of the criteria, e.g. scoliosis), blended with the
  strongest criterion signal (better in CV than either alone);
* ``quality_class = 1`` iff any criterion is violated or ``q >= t_quality``, so a reported
  violation never contradicts the class;
* Platt calibrators turn scores into the reported probabilities (``quality_prob`` and the
  per-criterion probabilities). They are monotone, so they never change a decision; keeping
  decisions on raw scores means a degenerate calibrator (few positives) cannot flatten them.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from dxa_qc.domain.taxonomy import ALLOWED_VIOLATIONS, AnatomicalRegion, Violation
from dxa_qc.models.quality.heads import LinearHead, PlattCalibrator

FORMAT_VERSION = 2  # 2: thresholds on raw scores
OVERALL = "overall"
GEOMETRY = "geometry"


class QualityModelFormatError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class TargetEnsemble:
    heads: tuple[LinearHead, ...]
    calibrator: PlattCalibrator = field(default_factory=PlattCalibrator)
    threshold: float = 0.0  # on the raw score

    def __post_init__(self) -> None:
        if not self.heads:
            raise ValueError("an ensemble needs at least one head")
        if not np.isfinite(self.threshold):
            raise ValueError(f"threshold must be finite, got {self.threshold}")

    def member_logits(self, inputs: ModelInputs) -> np.ndarray:
        """``(n, n_heads)`` raw head logits."""
        return np.column_stack([h.decision(inputs.for_head(h)) for h in self.heads])

    def score(self, inputs: ModelInputs) -> np.ndarray:
        """Mean member logit (uncalibrated)."""
        return self.member_logits(inputs).mean(axis=1)


@dataclass(frozen=True, slots=True)
class ModelInputs:
    """Batch of ``n`` images: geometry as named columns ``(n,)``, embeddings as ``(n, dim)`` per backbone."""

    geometry: Mapping[str, np.ndarray]
    embeddings: Mapping[str, np.ndarray]

    def for_head(self, head: LinearHead) -> np.ndarray:
        if head.source == GEOMETRY:
            missing = [n for n in head.feature_names if n not in self.geometry]
            if missing:
                raise KeyError(f"missing geometry features {missing}")
            return np.column_stack(
                [np.asarray(self.geometry[n], dtype=np.float64) for n in head.feature_names]
            )
        if head.source not in self.embeddings:
            raise KeyError(f"missing embeddings {head.source!r}")
        return self.embeddings[head.source]


@dataclass(frozen=True, slots=True)
class QualityPrediction:
    quality_prob: float
    quality_class: int
    criterion_probs: dict[Violation, float]
    violations: frozenset[Violation]


def quality_score(overall_score: np.ndarray, criterion_scores: np.ndarray) -> np.ndarray:
    """``criterion_scores``: shape ``(n, n_criteria)``."""
    return 0.5 * (np.asarray(overall_score) + np.max(criterion_scores, axis=1))


@dataclass(frozen=True, slots=True)
class RegionQualityModel:
    region: AnatomicalRegion
    criteria: dict[Violation, TargetEnsemble]
    overall: TargetEnsemble
    quality_calibrator: PlattCalibrator = field(default_factory=PlattCalibrator)
    quality_threshold: float = 0.0  # on the raw quality score
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        allowed = ALLOWED_VIOLATIONS[self.region]
        if set(self.criteria) != set(allowed):
            raise ValueError(f"criteria {sorted(self.criteria)} do not match region {self.region.value!r}")
        if not np.isfinite(self.quality_threshold):
            raise ValueError("quality_threshold must be finite")

    @property
    def ordered_criteria(self) -> tuple[Violation, ...]:
        return ALLOWED_VIOLATIONS[self.region]

    @property
    def backbones(self) -> tuple[str, ...]:
        """Embedding sources (backbone keys) this model needs, in a stable order."""
        ensembles = [*self.criteria.values(), self.overall]
        return tuple(sorted({h.source for e in ensembles for h in e.heads if h.source != GEOMETRY}))

    def geometry_features(self) -> tuple[str, ...]:
        names: dict[str, None] = {}
        for e in [*self.criteria.values(), self.overall]:
            for h in e.heads:
                if h.source == GEOMETRY:
                    names.update(dict.fromkeys(h.feature_names))
        return tuple(names)

    def criterion_scores(self, inputs: ModelInputs) -> np.ndarray:
        return np.column_stack([self.criteria[c].score(inputs) for c in self.ordered_criteria])

    def predict(self, inputs: ModelInputs) -> list[QualityPrediction]:
        scores = self.criterion_scores(inputs)
        q_score = quality_score(self.overall.score(inputs), scores)
        q_prob = self.quality_calibrator(q_score)
        result = []
        for i in range(scores.shape[0]):
            violated = frozenset(
                c for j, c in enumerate(self.ordered_criteria) if scores[i, j] >= self.criteria[c].threshold
            )
            probs = {
                c: float(self.criteria[c].calibrator(scores[i, j]))
                for j, c in enumerate(self.ordered_criteria)
            }
            cls = int(bool(violated) or q_score[i] >= self.quality_threshold)
            result.append(QualityPrediction(float(q_prob[i]), cls, probs, violated))
        return result

    # ---- persistence ----------------------------------------------------------------------

    def _named_ensembles(self) -> list[tuple[str, TargetEnsemble]]:
        return [(c.name, self.criteria[c]) for c in self.ordered_criteria] + [(OVERALL, self.overall)]

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {}
        targets = []
        for name, ens in self._named_ensembles():
            heads = []
            for j, h in enumerate(ens.heads):
                prefix = f"{name}.{j}"
                arrays[f"{prefix}.mean"], arrays[f"{prefix}.scale"], arrays[f"{prefix}.coef"] = (
                    h.mean,
                    h.scale,
                    h.coef,
                )
                heads.append(
                    {
                        "source": h.source,
                        "features": list(h.feature_names),
                        "l2_normalize": h.l2_normalize,
                        "intercept": h.intercept,
                    }
                )
            targets.append(
                {
                    "name": name,
                    "heads": heads,
                    "calibrator": [ens.calibrator.a, ens.calibrator.b],
                    "threshold": ens.threshold,
                }
            )
        meta = {
            "format_version": FORMAT_VERSION,
            "region": self.region.name,
            "targets": targets,
            "quality_calibrator": [self.quality_calibrator.a, self.quality_calibrator.b],
            "quality_threshold": self.quality_threshold,
            "metadata": self.metadata,
        }
        np.savez(path, meta=np.array(json.dumps(meta, ensure_ascii=False)), **arrays)

    @classmethod
    def load(cls, path: Path) -> RegionQualityModel:
        path = Path(path)
        try:
            with np.load(path, allow_pickle=False) as data:
                meta = json.loads(str(data["meta"]))
                arrays = {k: np.asarray(data[k], dtype=np.float64) for k in data.files if k != "meta"}
        except FileNotFoundError as exc:
            raise QualityModelFormatError(f"quality model not found: {path}") from exc
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise QualityModelFormatError(f"cannot load quality model {path}: {exc}") from exc
        if meta.get("format_version") != FORMAT_VERSION:
            raise QualityModelFormatError(f"unsupported quality model format: {meta.get('format_version')!r}")
        try:
            region = AnatomicalRegion[meta["region"]]
            ensembles: dict[str, TargetEnsemble] = {}
            for t in meta["targets"]:
                heads = tuple(
                    LinearHead(
                        h["source"],
                        tuple(h["features"]),
                        bool(h["l2_normalize"]),
                        arrays[f"{t['name']}.{j}.mean"],
                        arrays[f"{t['name']}.{j}.scale"],
                        arrays[f"{t['name']}.{j}.coef"],
                        float(h["intercept"]),
                    )
                    for j, h in enumerate(t["heads"])
                )
                ensembles[t["name"]] = TargetEnsemble(
                    heads, PlattCalibrator(*t["calibrator"]), float(t["threshold"])
                )
            overall = ensembles.pop(OVERALL)
            return cls(
                region=region,
                criteria={Violation[k]: v for k, v in ensembles.items()},
                overall=overall,
                quality_calibrator=PlattCalibrator(*meta["quality_calibrator"]),
                quality_threshold=float(meta["quality_threshold"]),
                metadata=dict(meta.get("metadata", {})),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise QualityModelFormatError(f"invalid quality model {path}: {exc}") from exc
