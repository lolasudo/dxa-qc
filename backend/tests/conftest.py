from __future__ import annotations

from pathlib import Path

import pytest

from dxa_qc.config import Settings, load_settings

BACKEND_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = BACKEND_ROOT.parent / "Датасет"
STUDIES_DIR = DATASET_ROOT / "Исследования"
TEST_SAMPLE_DIR = DATASET_ROOT / "Для теста"
LABELS_XLSX = DATASET_ROOT / "разметка.xlsx"


@pytest.fixture(scope="session")
def settings() -> Settings:
    return load_settings(BACKEND_ROOT / "configs")


def require(path: Path) -> Path:
    """Skip (not fail) dataset-dependent tests on machines without the dataset, e.g. CI."""
    if not path.exists():
        pytest.skip(f"dataset path not available: {path}")
    return path
