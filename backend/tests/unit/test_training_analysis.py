from __future__ import annotations

import numpy as np
import pytest

from dxa_qc.models.quality.model import OVERALL
from dxa_qc.training.analysis import NO_TAG, RegionPredictions, error_analysis, overall_report


def _part(name: str, groups, y, prob, pred, criteria: dict[str, tuple], comments, reps: int = 2):
    labels = {OVERALL: np.array(y), **{k: np.array(v[0]) for k, v in criteria.items()}}
    crossfit = [
        {
            "quality_prob": np.array(prob, float),
            "quality_class": np.array(pred),
            **{f"v_{k}": np.array(v[1]) for k, v in criteria.items()},
        }
        for _ in range(reps)
    ]
    return RegionPredictions(name, np.array(groups), labels, crossfit, np.array(comments, dtype=object))


SPINE = _part(
    "SPINE",
    ["s1", "s2", "s3", "s4"],
    [1, 0, 1, 0],
    [0.9, 0.2, 0.4, 0.3],
    [1, 0, 0, 0],
    {
        "POSITIONING": ([1, 0, 0, 0], [1, 0, 0, 0]),
        "SPINE_AXIS": ([0, 0, 1, 0], [0, 0, 0, 0]),
        "FOREIGN_OBJECTS": ([0, 0, 0, 0], [0, 0, 0, 0]),
    },
    ["", "сколиоз", "Сколиоз 2 ст.", ""],
)
HIP = _part(
    "HIP",
    ["s1", "s1", "s2", "s3"],
    [0, 1, 0, 1],
    [0.1, 0.8, 0.35, 0.7],
    [0, 1, 1, 1],
    {"POSITIONING": ([0, 1, 0, 1], [0, 1, 1, 1]), "HIP_ROI": ([0, 0, 0, 1], [0, 0, 0, 1])},
    ["", "", "endoprosthesis", ""],  # tag names, as make_manifest writes them
)


def test_overall_pools_both_regions_on_the_reported_probability() -> None:
    report = overall_report([SPINE, HIP], n_boot=50)
    q = report["quality_class"]
    assert q["n"] == 8 and q["n_positive"] == 4
    # AUC over the pooled quality_prob of both regions, recomputed by pair counting.
    y = np.array([1, 0, 1, 0, 0, 1, 0, 1])
    p = np.array([0.9, 0.2, 0.4, 0.3, 0.1, 0.8, 0.35, 0.7])
    pairs = [(i, j) for i in np.flatnonzero(y) for j in np.flatnonzero(1 - y)]
    assert q["roc_auc"]["value"] == pytest.approx(np.mean([p[i] > p[j] for i, j in pairs]), abs=1e-4)
    assert q["sensitivity"]["value"] == 0.75 and q["specificity"]["value"] == 0.75


def test_overall_violation_f1_pools_positioning_across_regions() -> None:
    report = overall_report([SPINE, HIP], n_boot=0)
    f1 = report["violation_f1"]
    # Positioning: spine TP1; hip TP2 FP1 -> tp 3, pred 4, true 3 -> 6/7.
    assert f1["Некорректная укладка"] == pytest.approx(6 / 7, abs=1e-4)
    assert f1["Не выравнена ось позвоночника"] == 0.0
    assert f1["Некорректная область интереса"] == 1.0
    assert report["violation_macro_f1"] == pytest.approx(np.mean(list(f1.values())), abs=1e-4)


def test_overall_requires_equal_repetitions() -> None:
    other = RegionPredictions("HIP", HIP.groups, HIP.labels, HIP.crossfit[:1], HIP.comments)
    with pytest.raises(ValueError, match="repetitions"):
        overall_report([SPINE, other], n_boot=0)


def test_error_analysis_by_clinical_note_and_consistent_errors() -> None:
    spine = error_analysis(SPINE)
    strata = spine["by_clinical_note"]
    assert strata["scoliosis"] == {"n": 2, "n_positive": 1, "sensitivity": 0.0, "specificity": 1.0}
    assert strata[NO_TAG]["n"] == 2 and strata[NO_TAG]["sensitivity"] == 1.0
    [err] = spine["consistent_errors"]
    assert err["study"] == "s3" and err["label"] == 1 and err["error_rate"] == 1.0
    assert err["violations"] == ["Не выравнена ось позвоночника"] and err["comment"] == "Сколиоз 2 ст."

    hip = error_analysis(HIP)
    assert hip["by_clinical_note"]["endoprosthesis"]["specificity"] == 0.0
    assert hip["n_consistent_errors"] == 1


def test_clinical_tags_from_free_text_and_tag_names() -> None:
    from dxa_qc.data.labels import ClinicalTag
    from dxa_qc.training.analysis import clinical_tags

    assert clinical_tags("Сколиоз, L6") == {ClinicalTag.SCOLIOSIS, ClinicalTag.TRANSITIONAL_VERTEBRA}
    assert clinical_tags("fracture;scoliosis") == {ClinicalTag.FRACTURE, ClinicalTag.SCOLIOSIS}
    assert clinical_tags("") == frozenset()
