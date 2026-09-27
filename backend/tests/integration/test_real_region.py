from __future__ import annotations

import csv
import shutil
import uuid

import pytest

from dxa_qc.data import ClinicalTag, load_labels
from dxa_qc.dicom_io import read_dicom
from dxa_qc.domain import HipSide
from dxa_qc.models.region import RegionClass, RegionClassifier
from dxa_qc.training.region import cross_validate, load_annotated_images
from tests.conftest import BACKEND_ROOT, LABELS_XLSX, STUDIES_DIR, TEST_SAMPLE_DIR, require

pytestmark = pytest.mark.dataset

WEIGHTS = BACKEND_ROOT / "weights" / "region_classifier.npz"
ANNOTATIONS = BACKEND_ROOT / "data" / "region_annotations.csv"
# Suffixes exist only in the organizer's sample; used as ground truth here, never by the code.
EXPECTED = {"ПОП": RegionClass.SPINE, "ППОБ": RegionClass.HIP_RIGHT, "ЛПОБ": RegionClass.HIP_LEFT}


@pytest.fixture(scope="module")
def model() -> RegionClassifier:
    return RegionClassifier.load(require(WEIGHTS))


def test_organizer_samples_classified_from_pixels_only(model, settings, tmp_path):
    for f in sorted(require(TEST_SAMPLE_DIR).glob("*.dcm")):
        # Closed-test files have no region suffix: rename to a random name first.
        anon = shutil.copy(f, tmp_path / f"{uuid.uuid4().hex}.dcm")
        img = read_dicom(anon, settings.pixel_spacing, settings.ingest.max_file_bytes)
        pred = model.predict(img.pixels)
        assert pred.label is EXPECTED[f.stem.split("_")[-1]], f.name
        assert pred.confidence > 0.9


def test_annotations_consistent_with_expert_labels():
    labels = load_labels(require(LABELS_XLSX))
    by_study: dict[str, set[str]] = {}
    with require(ANNOTATIONS).open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            by_study.setdefault(row["study_id"], set()).add(row["side"] or "spine")
    assert set(by_study) == set(labels)
    for study, found in by_study.items():
        lab = labels[study]
        assert ("spine" in found) == (lab.spine is not None), study
        for side in HipSide:
            if side in lab.hips:
                assert side.value in found, f"{study}: labeled {side} hip has no image"
            elif side.value in found:
                # Only endoprosthesis hips are imaged but left unannotated by the experts.
                assert ClinicalTag.ENDOPROSTHESIS in lab.tags, f"{study}: unlabeled {side} hip image"


def test_cross_validation_is_perfect_on_verified_annotations():
    items = load_annotated_images(require(ANNOTATIONS), require(STUDIES_DIR))
    report = cross_validate(items)
    assert report["n_studies"] == 100
    assert report["accuracy"] == 1.0, report["errors"]
