from __future__ import annotations

from collections import Counter

import pytest

from dxa_qc.data import build_index, load_labels, load_region_annotations
from tests.conftest import BACKEND_ROOT, LABELS_XLSX, STUDIES_DIR, require

pytestmark = pytest.mark.dataset


@pytest.fixture(scope="module")
def index():
    return build_index(
        require(STUDIES_DIR),
        load_region_annotations(BACKEND_ROOT / "data" / "region_annotations.csv"),
        load_labels(require(LABELS_XLSX)),
    )


def test_every_file_indexed(index):
    assert len(index) == 499
    assert sum(not i.is_duplicate for i in index) == 252


def test_one_canonical_image_per_region_per_study(index):
    # Makes the study-level label -> image mapping unambiguous.
    counts = Counter((i.study_id, i.region_class) for i in index if not i.is_duplicate)
    assert set(counts.values()) == {1}


def test_labeled_canonical_images_match_expert_counts(index):
    canonical = [i for i in index if not i.is_duplicate and i.label is not None]
    # 99 spines + 72 right + 78 left hips annotated by the experts.
    assert len(canonical) == 99 + 72 + 78
    unlabeled = [i for i in index if not i.is_duplicate and i.label is None]
    assert len(unlabeled) == 3  # endoprosthesis hips
