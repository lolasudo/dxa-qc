from __future__ import annotations

import pytest

from dxa_qc.data import ClinicalTag, load_labels
from dxa_qc.domain import HipSide, Violation
from tests.conftest import LABELS_XLSX, STUDIES_DIR, require

pytestmark = pytest.mark.dataset


@pytest.fixture(scope="module")
def labels():
    return load_labels(require(LABELS_XLSX))


def test_counts_match_eda(labels):
    # Figures established during EDA; a mismatch means the file or the parser changed.
    assert len(labels) == 100
    spine = [s.spine for s in labels.values() if s.spine]
    assert len(spine) == 99
    assert sum(r.overall for r in spine) == 32
    assert sum(Violation.FOREIGN_OBJECTS in r.violations for r in spine) == 17
    assert sum(HipSide.RIGHT in s.hips for s in labels.values()) == 72
    assert sum(HipSide.LEFT in s.hips for s in labels.values()) == 78


def test_every_labeled_study_has_a_folder(labels):
    folders = {p.name for p in require(STUDIES_DIR).iterdir() if p.is_dir()}
    assert set(labels) == folders


def test_clinical_tags_found(labels):
    tags = [t for s in labels.values() for t in s.tags]
    assert tags.count(ClinicalTag.ENDOPROSTHESIS) == 2
    assert tags.count(ClinicalTag.FRACTURE) == 1
    assert tags.count(ClinicalTag.TRANSITIONAL_VERTEBRA) == 1
    assert tags.count(ClinicalTag.SCOLIOSIS) >= 14
