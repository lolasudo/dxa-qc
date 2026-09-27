"""Evaluation beyond the per-region report: metrics over both regions together and
error analysis by the clinical notes of the expert annotation.

Inputs are the cross-fitted predictions of :func:`dxa_qc.training.quality.evaluate`,
so nothing here sees a label its prediction was fitted on.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from dxa_qc.data.labels import ClinicalTag, extract_tags
from dxa_qc.domain.taxonomy import ALLOWED_VIOLATIONS, Violation
from dxa_qc.models.quality.model import OVERALL
from dxa_qc.training.metrics import binary_report, f1

ALL_REGIONS = "ALL_REGIONS"  # key of the pooled report in quality_metrics.json
NO_TAG = "no_note"


@dataclass(frozen=True, slots=True)
class RegionPredictions:
    """One region's labels, groups, comments and the cross-fitted predictions of every CV repetition."""

    region_name: str
    groups: np.ndarray  # study ids (the same study appears in both regions)
    labels: dict[str, np.ndarray]  # target name (Violation.name or OVERALL) -> 0/1
    crossfit: Sequence[dict[str, np.ndarray]]
    comments: np.ndarray  # expert comment per image ("" if none)


def overall_report(parts: Sequence[RegionPredictions], *, n_boot: int, seed: int = 0) -> dict[str, object]:
    """quality_class metrics over all images of both regions, ranked by the reported ``quality_prob``.

    Raw scores of the two regions' models live on different scales; the cross-fitted calibrated
    probability is comparable across them and is the value written to the report. The bootstrap
    resamples studies, which keeps a study's spine and hips together.
    """
    reps = len(parts[0].crossfit)
    if any(len(p.crossfit) != reps for p in parts):
        raise ValueError("all regions must have the same number of CV repetitions")
    y = np.concatenate([p.labels[OVERALL] for p in parts])
    groups = np.concatenate([p.groups for p in parts])
    probs = [np.concatenate([p.crossfit[r]["quality_prob"] for p in parts]) for r in range(reps)]
    preds = [np.concatenate([p.crossfit[r]["quality_class"] for p in parts]) for r in range(reps)]
    report: dict[str, object] = {
        "quality_class": binary_report(y, probs, preds, groups, n_boot=n_boot, seed=seed)
    }
    # Macro-F1 over the four violation types; "incorrect positioning" pools both regions.
    per_type: dict[str, float] = {}
    for v in dict.fromkeys(v for vs in ALLOWED_VIOLATIONS.values() for v in vs):
        having = [p for p in parts if v.name in p.labels]
        yv = np.concatenate([p.labels[v.name] for p in having])
        scores = [
            f1(yv, yv, np.concatenate([p.crossfit[r][f"v_{v.name}"] for p in having])) for r in range(reps)
        ]
        per_type[v.value] = round(float(np.mean(scores)), 4)
    report["violation_f1"] = per_type
    report["violation_macro_f1"] = round(float(np.mean(list(per_type.values()))), 4)
    return report


def clinical_tags(comment: str) -> frozenset[ClinicalTag]:
    """Tags of a manifest comment: free text of the expert, or the tag names that
    ``make_manifest`` writes for the organizer's workbook (``scoliosis;fracture``)."""
    names = {t.strip() for t in comment.split(";")}
    return extract_tags(comment) | {t for t in ClinicalTag if t.value in names}


def _rates(y: np.ndarray, pred_reps: np.ndarray) -> dict[str, object]:
    """Error rates over images x repetitions (``pred_reps``: reps x n)."""
    pos, neg = y == 1, y == 0
    return {
        "n": len(y),
        "n_positive": int(pos.sum()),
        "sensitivity": round(float(pred_reps[:, pos].mean()), 4) if pos.any() else None,
        "specificity": round(float(1 - pred_reps[:, neg].mean()), 4) if neg.any() else None,
    }


def error_analysis(part: RegionPredictions, *, top: int = 10) -> dict[str, object]:
    """quality_class errors stratified by clinical notes, plus the most consistently misclassified
    images (by study) - the cases to show and discuss, not to tune on."""
    y = part.labels[OVERALL]
    pred = np.stack([h["quality_class"] for h in part.crossfit])  # reps x n
    prob = np.mean([h["quality_prob"] for h in part.crossfit], axis=0)
    tags = [clinical_tags(str(c)) for c in part.comments]
    strata: dict[str, object] = {}
    for name in [*sorted({t.value for ts in tags for t in ts}), NO_TAG]:
        mask = np.array([(name in {t.value for t in ts}) if name != NO_TAG else not ts for ts in tags])
        if mask.any():
            strata[name] = _rates(y[mask], pred[:, mask])
    error_rate = (pred != y).mean(axis=0)
    names = [v for v in Violation if v.name in part.labels]
    worst = [int(i) for i in np.argsort(-error_rate, kind="mergesort") if error_rate[i] >= 0.5][:top]
    return {
        "by_clinical_note": strata,
        "consistent_errors": [
            {
                "study": str(part.groups[i]),
                "label": int(y[i]),
                "error_rate": round(float(error_rate[i]), 3),
                "mean_quality_prob": round(float(prob[i]), 3),
                "violations": [v.value for v in names if part.labels[v.name][i]],
                "comment": str(part.comments[i]),
            }
            for i in worst
        ],
        "n_consistent_errors": int((error_rate >= 0.5).sum()),
    }
