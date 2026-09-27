"""Train and evaluate the quality models (both regions).

Usage::

    python -m dxa_qc.training.backbones --weights-dir weights/backbones   # once, needs network
    python -m dxa_qc.training.quality --studies ../Датасет/Исследования \
        --labels ../Датасет/разметка.xlsx --annotations data/region_annotations.csv --weights-dir weights

Protocol:

* one canonical image per (study, region); CV groups = study, so no study is split across folds;
* repeated StratifiedGroupKFold; within every repetition the calibrators and thresholds used on
  a fold are fitted on the *other* folds' out-of-fold scores only, so reported metrics never see
  the labels they are evaluated on (no optimistic threshold tuning);
* the model design (feature groups, backbones, regularization) is fixed a priori in
  :data:`DESIGNS`; it was chosen in exploratory CV documented in the README;
* the shipped model is refitted on all data, with calibrators/thresholds from the pooled OOF scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from dxa_qc.data.manifest import ManifestRow, load_manifest
from dxa_qc.domain.taxonomy import ALLOWED_VIOLATIONS, AnatomicalRegion, HipSide, Violation
from dxa_qc.models.quality.embeddings import (
    CONVNEXT_S,
    DINOV2_S,
    DINOV2_S_TROCHANTER,
    EFFNET_B3,
    BackboneSpec,
)
from dxa_qc.models.quality.geometry import hip as hip_geo
from dxa_qc.models.quality.geometry import spine as spine_geo
from dxa_qc.models.quality.heads import LinearHead, PlattCalibrator
from dxa_qc.models.quality.model import (
    GEOMETRY,
    OVERALL,
    ModelInputs,
    RegionQualityModel,
    TargetEnsemble,
    quality_score,
)
from dxa_qc.models.region.classifier import RegionClass
from dxa_qc.training.config import TrainingConfig, load_training_config
from dxa_qc.training.metrics import best_score_threshold, binary_report, macro_f1

log = logging.getLogger(__name__)

SEED = 20260924
C_GEOMETRY = 0.1
C_EMBEDDING = 0.01


@dataclass(frozen=True, slots=True)
class Design:
    backbones: tuple[BackboneSpec, ...]
    geometry_groups: dict[Violation, tuple[str, ...]]
    # Candidate L2 strengths: one value = fixed; several = chosen by nested grouped CV AUC.
    geometry_c: tuple[float, ...] = (C_GEOMETRY,)
    embedding_c: tuple[float, ...] = (C_EMBEDDING,)
    inner_splits: int = 3

    def geometry_for(self, target: str) -> tuple[str, ...]:
        if target == OVERALL:
            union: dict[str, None] = {}
            for group in self.geometry_groups.values():
                union.update(dict.fromkeys(group))
            return tuple(union)
        return self.geometry_groups[Violation[target]]


GEOMETRY_GROUPS: dict[AnatomicalRegion, dict[Violation, tuple[str, ...]]] = {
    AnatomicalRegion.SPINE: {
        Violation.POSITIONING: spine_geo.COVERAGE_FEATURES,
        Violation.SPINE_AXIS: spine_geo.AXIS_FEATURES,
        Violation.FOREIGN_OBJECTS: spine_geo.FOREIGN_FEATURES,
    },
    AnatomicalRegion.HIP: {
        Violation.POSITIONING: hip_geo.POSITIONING_FEATURES,
        Violation.HIP_ROI: hip_geo.ROI_FEATURES,
    },
}


def design_from_config(region: AnatomicalRegion, config: TrainingConfig) -> Design:
    return Design(
        backbones=config.backbones(region),
        geometry_groups=GEOMETRY_GROUPS[region],
        geometry_c=config.heads.geometry_c,
        embedding_c=config.heads.embedding_c,
        inner_splits=config.heads.inner_splits,
    )


# Defaults of configs/training.yaml, kept in code for tests and programmatic use.
DESIGNS: dict[AnatomicalRegion, Design] = {
    AnatomicalRegion.SPINE: Design(
        (EFFNET_B3, DINOV2_S),
        {
            Violation.POSITIONING: spine_geo.COVERAGE_FEATURES,
            Violation.SPINE_AXIS: spine_geo.AXIS_FEATURES,
            Violation.FOREIGN_OBJECTS: spine_geo.FOREIGN_FEATURES,
        },
    ),
    AnatomicalRegion.HIP: Design(
        (CONVNEXT_S, DINOV2_S_TROCHANTER),
        {Violation.POSITIONING: hip_geo.POSITIONING_FEATURES, Violation.HIP_ROI: hip_geo.ROI_FEATURES},
    ),
}


@dataclass(frozen=True, slots=True)
class RegionData:
    """Everything the learning code needs for one region; ``labels``: target name -> 0/1 array."""

    region: AnatomicalRegion
    groups: np.ndarray
    inputs: ModelInputs
    labels: dict[str, np.ndarray]
    comments: np.ndarray | None = None  # expert comments, for error analysis only (never a feature)

    @property
    def n(self) -> int:
        return len(self.groups)

    def subset(self, idx: np.ndarray) -> RegionData:
        return RegionData(
            self.region,
            self.groups[idx],
            ModelInputs(
                {k: v[idx] for k, v in self.inputs.geometry.items()},
                {k: v[idx] for k, v in self.inputs.embeddings.items()},
            ),
            {k: v[idx] for k, v in self.labels.items()},
            None if self.comments is None else self.comments[idx],
        )


def targets(region: AnatomicalRegion) -> list[str]:
    return [v.name for v in ALLOWED_VIOLATIONS[region]] + [OVERALL]


def fit_ensemble(data: RegionData, target: str, design: Design, seed: int = SEED) -> TargetEnsemble:
    """Equal-weight ensemble of the geometry head and one head per backbone.

    Uncalibrated (identity calibrator, threshold 0): calibration and thresholds are fitted on
    out-of-fold scores later. Learned stacking weights were tried and were worse in nested CV.
    """
    y = data.labels[target]
    geo_names = design.geometry_for(target)
    x_geo = np.column_stack([data.inputs.geometry[n] for n in geo_names])
    members: list[tuple[str, np.ndarray, bool, tuple[str, ...], tuple[float, ...]]] = [
        (GEOMETRY, x_geo, False, geo_names, design.geometry_c)
    ]
    members += [
        (b.key, data.inputs.embeddings[b.key], True, (), design.embedding_c) for b in design.backbones
    ]
    heads = []
    for source, x, l2, names, grid in members:
        c = select_c(x, y, data.groups, grid, l2_normalize=l2, n_splits=design.inner_splits, seed=seed)
        heads.append(
            LinearHead.fit(x, y, source=source, c=c, l2_normalize=l2, feature_names=names, seed=seed)
        )
    return TargetEnsemble(tuple(heads))


def select_c(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    grid: tuple[float, ...],
    *,
    l2_normalize: bool,
    n_splits: int,
    seed: int,
) -> float:
    """Pick C by grouped inner CV AUC on the given (training) rows only: no outer-fold leakage.

    With a single candidate nothing is fitted. Ties go to the smaller C (stronger regularization).
    """
    if len(grid) == 1:
        return grid[0]
    from sklearn.model_selection import StratifiedGroupKFold

    from dxa_qc.training.metrics import roc_auc

    n_splits = int(min(n_splits, y.sum(), (1 - y).sum()))
    if n_splits < 2:
        return min(grid)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(cv.split(x, y, groups))
    best_c, best_auc = min(grid), -1.0
    for c in sorted(grid):
        scores = np.zeros(len(y))
        try:
            for train, test in folds:
                head = LinearHead.fit(
                    x[train], y[train], source="inner", c=c, l2_normalize=l2_normalize, seed=seed
                )
                scores[test] = head.decision(x[test])
        except ValueError:  # an inner fold without positives
            continue
        auc = roc_auc(y, scores, y)
        if auc > best_auc + 1e-9:
            best_c, best_auc = c, auc
    return best_c


def _folds(data: RegionData, n_splits: int, seed: int) -> list[np.ndarray]:
    from sklearn.model_selection import StratifiedGroupKFold

    # Stratify on the overall verdict: every criterion positive is also an overall positive in
    # almost all rows, and finer strata are too small for 5 folds.
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [test for _, test in cv.split(np.zeros(data.n), data.labels[OVERALL], data.groups)]


def oof_scores(
    data: RegionData, design: Design, folds: Sequence[np.ndarray], seed: int = SEED
) -> dict[str, np.ndarray]:
    """Out-of-fold ensemble scores (mean logits) for every target."""
    scores = {t: np.full(data.n, np.nan) for t in targets(data.region)}
    for test in folds:
        train = np.setdiff1d(np.arange(data.n), test)
        train_data, test_data = data.subset(train), data.subset(test)
        for t in scores:
            scores[t][test] = fit_ensemble(train_data, t, design, seed).score(test_data.inputs)
    if any(np.isnan(s).any() for s in scores.values()):
        raise RuntimeError("folds do not cover every sample")
    return scores


@dataclass(frozen=True, slots=True)
class Decision:
    """Thresholds on raw scores (decisions) and calibrators (reported probabilities)."""

    thresholds: dict[str, float]
    quality_threshold: float
    calibrators: dict[str, PlattCalibrator]
    quality_calibrator: PlattCalibrator


def _criteria(region: AnatomicalRegion) -> list[str]:
    return [v.name for v in ALLOWED_VIOLATIONS[region]]


def _quality_score(region: AnatomicalRegion, scores: dict[str, np.ndarray]) -> np.ndarray:
    return quality_score(scores[OVERALL], np.column_stack([scores[c] for c in _criteria(region)]))


def fit_decision(
    region: AnatomicalRegion, scores: dict[str, np.ndarray], labels: dict[str, np.ndarray]
) -> Decision:
    crit = _criteria(region)
    thresholds = {c: best_score_threshold(labels[c], scores[c]) for c in crit}
    q = _quality_score(region, scores)
    any_flag = np.column_stack([scores[c] >= thresholds[c] for c in crit]).any(axis=1)
    # Images already flagged are class 1 regardless of the cut: pin them above every other score
    # (finite, so the stored threshold stays finite), then tune the cut for the rest by F1.
    pinned = np.where(any_flag, q.max() + 1.0, q)
    q_thr = best_score_threshold(labels[OVERALL], pinned)
    calibrators = {t: PlattCalibrator.fit(scores[t], labels[t]) for t in scores}
    return Decision(thresholds, q_thr, calibrators, PlattCalibrator.fit(q, labels[OVERALL]))


def apply_decision(
    region: AnatomicalRegion, scores: dict[str, np.ndarray], d: Decision
) -> dict[str, np.ndarray]:
    crit = _criteria(region)
    flags = np.column_stack([scores[c] >= d.thresholds[c] for c in crit])
    q = _quality_score(region, scores)
    return {
        **{f"s_{t}": s for t, s in scores.items()},
        **{f"v_{c}": flags[:, j].astype(int) for j, c in enumerate(crit)},
        "quality_score": q,
        "quality_prob": d.quality_calibrator(q),
        "quality_class": (flags.any(axis=1) | (q >= d.quality_threshold)).astype(int),
    }


def crossfit_predictions(
    data: RegionData, scores: dict[str, np.ndarray], folds: Sequence[np.ndarray]
) -> dict[str, np.ndarray]:
    """Cross-fitted decisions: fold k is decided by thresholds/calibrators fitted on the other folds.

    Ranking metrics are computed on the pooled raw OOF scores, which the cross-fitting leaves as is.
    """
    out: dict[str, np.ndarray] = {}
    for test in folds:
        train = np.setdiff1d(np.arange(data.n), test)
        decision = fit_decision(
            data.region,
            {t: s[train] for t, s in scores.items()},
            {t: y[train] for t, y in data.labels.items()},
        )
        part = apply_decision(data.region, {t: s[test] for t, s in scores.items()}, decision)
        for k, v in part.items():
            out.setdefault(k, np.zeros(data.n, dtype=v.dtype))[test] = v
    return out


def evaluate(
    data: RegionData,
    design: Design,
    *,
    reps: int,
    n_splits: int = 5,
    n_boot: int = 1000,
    seed: int = SEED,
) -> tuple[dict[str, object], dict[str, np.ndarray], list[dict[str, np.ndarray]]]:
    """Returns (report, mean OOF scores over reps, crossfit predictions per rep)."""
    all_scores: list[dict[str, np.ndarray]] = []
    crossfit: list[dict[str, np.ndarray]] = []
    for r in range(reps):
        folds = _folds(data, n_splits, seed + r)
        scores = oof_scores(data, design, folds, seed)
        all_scores.append(scores)
        crossfit.append(crossfit_predictions(data, scores, folds))
        log.info("%s: repetition %d/%d done", data.region.name, r + 1, reps)
    mean_scores = {t: np.mean([s[t] for s in all_scores], axis=0) for t in all_scores[0]}
    report = region_report(data, crossfit, n_boot=n_boot)
    return report, mean_scores, crossfit


def region_report(
    data: RegionData, crossfit: list[dict[str, np.ndarray]], *, n_boot: int
) -> dict[str, object]:
    crit = _criteria(data.region)
    report: dict[str, object] = {
        "quality_class": binary_report(
            data.labels[OVERALL],
            [h["quality_score"] for h in crossfit],
            [h["quality_class"] for h in crossfit],
            data.groups,
            n_boot=n_boot,
        ),
    }
    for c in crit:
        report[c] = binary_report(
            data.labels[c],
            [h[f"s_{c}"] for h in crossfit],
            [h[f"v_{c}"] for h in crossfit],
            data.groups,
            n_boot=n_boot,
        )
    # Calibration of the reported probability (cross-fitted Platt): Brier vs. the prevalence-only
    # forecast, which is what an uninformative but calibrated model would score.
    y_q = data.labels[OVERALL]
    report["quality_prob_brier"] = {
        "value": round(float(np.mean([np.mean((h["quality_prob"] - y_q) ** 2) for h in crossfit])), 4),
        "prevalence_baseline": round(float(np.mean((y_q.mean() - y_q) ** 2)), 4),
    }
    y_multi = np.column_stack([data.labels[c] for c in crit])
    report["violation_macro_f1"] = round(
        float(np.mean([macro_f1(y_multi, np.column_stack([h[f"v_{c}"] for c in crit])) for h in crossfit])), 4
    )
    return report


def fit_final(
    data: RegionData,
    design: Design,
    mean_scores: dict[str, np.ndarray],
    metadata: dict[str, object],
    seed: int = SEED,
) -> RegionQualityModel:
    decision = fit_decision(data.region, mean_scores, data.labels)
    ensembles = {
        t: replace(
            fit_ensemble(data, t, design, seed),
            calibrator=decision.calibrators[t],
            threshold=decision.thresholds.get(t, 0.0),
        )
        for t in targets(data.region)
    }
    overall = ensembles.pop(OVERALL)
    return RegionQualityModel(
        region=data.region,
        criteria={Violation[k]: v for k, v in ensembles.items()},
        overall=overall,
        quality_calibrator=decision.quality_calibrator,
        quality_threshold=decision.quality_threshold,
        metadata=metadata,
    )


# ---- dataset -> RegionData (needs the real data and the ML extras) -------------------------------


PREPROCESSING_VERSION = 1  # bump when prepare_image changes: invalidates the embedding cache


class EmbeddingCache:
    """On-disk cache ``<dir>/<spec hash>/<pixel digest>.npy`` (numpy format, never pickle).

    Frozen-backbone embeddings dominate training time on large datasets; with the cache,
    re-training with other heads/C/CV settings touches the GPU only for new images.
    """

    def __init__(self, root: Path | None) -> None:
        self.root = Path(root) if root else None

    def _dir(self, spec: BackboneSpec) -> Path:
        key = json.dumps({"spec": spec.to_dict(), "prep": PREPROCESSING_VERSION}, sort_keys=True)
        if self.root is None:
            raise RuntimeError("embedding cache is disabled")
        return self.root / hashlib.sha256(key.encode()).hexdigest()[:16]

    def get(self, spec: BackboneSpec, digest: str) -> np.ndarray | None:
        if self.root is None:
            return None
        path = self._dir(spec) / f"{digest}.npy"
        try:
            return np.load(path, allow_pickle=False) if path.is_file() else None
        except (OSError, ValueError):
            return None  # corrupted entry: recompute

    def put(self, spec: BackboneSpec, digest: str, vector: np.ndarray) -> None:
        if self.root is None:
            return
        directory = self._dir(spec)
        directory.mkdir(parents=True, exist_ok=True)
        tmp = directory / f"{digest}.tmp.npy"
        np.save(tmp, vector, allow_pickle=False)
        tmp.replace(directory / f"{digest}.npy")


def resolve_region(
    row: ManifestRow, pixels: np.ndarray, region_model: Any
) -> tuple[AnatomicalRegion, HipSide | None]:
    """Manifest region/side, or the region classifier's answer where the manifest leaves them empty."""
    if row.region is AnatomicalRegion.SPINE:
        return AnatomicalRegion.SPINE, None
    if row.region is AnatomicalRegion.HIP and row.side is not None:
        return AnatomicalRegion.HIP, row.side
    if region_model is None:
        raise ValueError(f"line {row.line}: region/side is empty and no region classifier is available")
    pred = region_model.predict(pixels)
    if row.region is AnatomicalRegion.HIP:  # region given, only the side is detected
        left = pred.probabilities.get(RegionClass.HIP_LEFT, 0.0)
        right = pred.probabilities.get(RegionClass.HIP_RIGHT, 0.0)
        return AnatomicalRegion.HIP, HipSide.LEFT if left >= right else HipSide.RIGHT
    return pred.region, pred.side


def load_manifest_data(
    rows: list[ManifestRow],
    weights_dir: Path,
    config: TrainingConfig,
    *,
    region_model: Any = None,
) -> dict[AnatomicalRegion, RegionData]:
    """Stream labeled manifest rows in batches: memory stays O(batch), not O(dataset).

    Byte-identical images inside a study are counted once. Unreadable files are skipped with a
    warning (a large dataset always has a few), never silently relabeled.
    """
    from dxa_qc.config import load_settings
    from dxa_qc.dicom_io import DicomReadError, pixel_digest, read_dicom
    from dxa_qc.models.quality.assessor import BACKBONES_SUBDIR, prepare_image
    from dxa_qc.models.quality.embeddings import EmbeddingExtractor

    settings = load_settings()
    limits = settings.ingest
    device = None if config.embedding.device == "auto" else config.embedding.device
    extractor = EmbeddingExtractor(weights_dir / BACKBONES_SUBDIR, device)
    cache = EmbeddingCache(config.embedding.cache_dir)
    designs = {r: design_from_config(r, config) for r in AnatomicalRegion}
    acc: dict[AnatomicalRegion, dict[str, list[Any]]] = {
        r: {"groups": [], "geometry": [], "rows": [], **{b.key: [] for b in designs[r].backbones}}
        for r in AnatomicalRegion
    }
    seen: set[tuple[str, str]] = set()
    labeled = [r for r in rows if r.labeled]
    batch_size = config.embedding.batch_size
    skipped = 0
    for start in range(0, len(labeled), batch_size):
        pending: dict[AnatomicalRegion, list[tuple[ManifestRow, str, Any]]] = {
            r: [] for r in AnatomicalRegion
        }
        for row in labeled[start : start + batch_size]:
            try:
                digest = pixel_digest(row.path)
                if (row.study_id, digest) in seen:
                    continue
                image = read_dicom(
                    row.path,
                    settings.pixel_spacing,
                    limits.max_file_bytes,
                    max_side=limits.max_image_side,
                    physical_range_mm=(limits.min_physical_mm, limits.max_physical_mm),
                )
                region, side = resolve_region(row, image.pixels, region_model)
                if row.region is not None and region is not row.region:
                    raise ValueError(f"line {row.line}: manifest region disagrees with the classifier")
                prepared = prepare_image(image, region, side)
            except (DicomReadError, ValueError, OSError) as exc:
                skipped += 1
                log.warning("skipping %s: %s", row.path, exc)
                continue
            seen.add((row.study_id, digest))
            pending[region].append((row, digest, prepared))
        for region, items in pending.items():
            if not items:
                continue
            a = acc[region]
            for spec in designs[region].backbones:
                vectors: list[np.ndarray | None] = [cache.get(spec, d) for _, d, _ in items]
                missing = [i for i, v in enumerate(vectors) if v is None]
                if missing:
                    computed = extractor.embed(
                        [items[i][2].image for i in missing], spec, batch_size=batch_size
                    )
                    for i, vec in zip(missing, computed, strict=True):
                        vectors[i] = vec
                        cache.put(spec, items[i][1], vec)
                a[spec.key].extend(vectors)
            for row, _, prepared in items:
                a["groups"].append(row.study_id)
                a["geometry"].append(prepared.geometry.as_dict())
                a["rows"].append(row)
        log.info("prepared %d/%d labeled images", min(start + batch_size, len(labeled)), len(labeled))
    if skipped:
        log.warning("%d image(s) skipped (see warnings above)", skipped)

    result = {}
    for region, a in acc.items():
        if not a["groups"]:
            raise ValueError(f"no labeled {region.name.lower()} images in the manifest")
        geometry = {n: np.array([g[n] for g in a["geometry"]]) for n in a["geometry"][0]}
        embeddings = {b.key: np.stack(a[b.key]) for b in designs[region].backbones}
        labels = {
            v.name: np.array([int(v in r.violations) for r in a["rows"]]) for v in ALLOWED_VIOLATIONS[region]
        }
        labels[OVERALL] = np.array([r.overall for r in a["rows"]])
        for name, y in labels.items():
            if len(np.unique(y)) < 2:
                raise ValueError(f"{region.name}: target {name} has a single class, it cannot be trained")
        result[region] = RegionData(
            region,
            np.array(a["groups"]),
            ModelInputs(geometry, embeddings),
            labels,
            np.array([r.comment for r in a["rows"]], dtype=object),
        )
        log.info(
            "%s: %d images, %d studies, %d positives overall",
            region.name,
            len(a["groups"]),
            len(set(a["groups"])),
            labels[OVERALL].sum(),
        )
    return result


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = parser.add_argument_group("dataset: a manifest, or the organizer's format")
    src.add_argument("--manifest", type=Path, help="training manifest CSV (see dxa_qc.data.manifest)")
    src.add_argument("--images-root", type=Path, help="root for manifest paths (default: the manifest's dir)")
    src.add_argument("--studies", type=Path, help="organizer format: studies folder")
    src.add_argument("--labels", type=Path, help="organizer format: разметка.xlsx")
    src.add_argument("--annotations", type=Path, help="organizer format: region_annotations.csv")
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="training config (default: configs/training.yaml)")
    parser.add_argument("--reps", type=int, help="override cv.repetitions")
    parser.add_argument("--n-boot", type=int, help="override cv.bootstrap")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from dxa_qc.models.quality.assessor import BACKBONES_SUBDIR, MODEL_FILES
    from dxa_qc.models.quality.embeddings import file_sha256
    from dxa_qc.models.region import ModelFormatError, RegionClassifier

    config = load_training_config(args.config)
    reps = args.reps or config.cv.repetitions
    n_boot = config.cv.bootstrap if args.n_boot is None else args.n_boot

    if args.manifest:
        rows = load_manifest(args.manifest, args.images_root)
        dataset_sha = _sha256(args.manifest)
    elif args.studies and args.labels and args.annotations:
        import tempfile

        from dxa_qc.data.manifest import write_manifest
        from dxa_qc.training.make_manifest import build_rows

        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.csv"
            write_manifest(build_rows(args.studies, args.labels, args.annotations, args.studies), manifest)
            rows = load_manifest(manifest, images_root=args.studies)
            dataset_sha = _sha256(manifest)
    else:
        parser.error("give --manifest, or all of --studies/--labels/--annotations")

    try:  # needed only when the manifest leaves region/side empty
        region_model = RegionClassifier.load(args.weights_dir / "region_classifier.npz")
    except ModelFormatError:
        region_model = None

    from dxa_qc.training.analysis import ALL_REGIONS, RegionPredictions, error_analysis, overall_report

    data = load_manifest_data(rows, args.weights_dir, config, region_model=region_model)
    reports: dict[str, object] = {}
    predictions: list[RegionPredictions] = []
    for region, region_data in data.items():
        design = design_from_config(region, config)
        report, mean_scores, crossfit = evaluate(
            region_data, design, reps=reps, n_splits=config.cv.n_splits, n_boot=n_boot, seed=config.seed
        )
        comments = region_data.comments
        part = RegionPredictions(
            region.name,
            region_data.groups,
            region_data.labels,
            crossfit,
            comments if comments is not None else np.full(region_data.n, "", dtype=object),
        )
        predictions.append(part)
        # Error analysis names studies and quotes expert notes: metrics file only, not the model file.
        reports[region.name] = {**report, "error_analysis": error_analysis(part)}
        metadata = {
            "seed": config.seed,
            "cv": f"{reps}x StratifiedGroupKFold({config.cv.n_splits}), groups=study",
            "n_images": region_data.n,
            "n_studies": len(set(region_data.groups.tolist())),
            "dataset_sha256": dataset_sha,
            "heads": {"geometry_c": list(design.geometry_c), "embedding_c": list(design.embedding_c)},
            "backbones": {
                b.key: {
                    **b.to_dict(),
                    "sha256": file_sha256(args.weights_dir / BACKBONES_SUBDIR / b.weights_file),
                }
                for b in design.backbones
            },
            "cv_report": report,
        }
        model = fit_final(region_data, design, mean_scores, metadata, seed=config.seed)
        model.save(args.weights_dir / MODEL_FILES[region])
        log.info(
            "%s: saved %s; CV: %s", region.name, MODEL_FILES[region], json.dumps(report["quality_class"])
        )
    reports[ALL_REGIONS] = overall_report(predictions, n_boot=n_boot)
    log.info("all regions: %s", json.dumps(reports[ALL_REGIONS]["quality_class"]))
    out = args.weights_dir / "quality_metrics.json"
    out.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("metrics written to %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
