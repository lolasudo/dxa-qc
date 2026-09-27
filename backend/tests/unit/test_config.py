from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from dxa_qc.config import ConfigError, load_settings
from dxa_qc.config.settings import CONFIG_DIR_ENV
from tests.conftest import BACKEND_ROOT


@pytest.fixture
def config_copy(tmp_path: Path) -> Path:
    shutil.copytree(BACKEND_ROOT / "configs", tmp_path / "configs")
    return tmp_path / "configs"


def test_repo_configs_match_spec(settings):
    # Values fixed by the specification; a change here must be intentional.
    assert settings.pixel_spacing.row_mm == 1.05
    assert settings.pixel_spacing.column_mm == 0.6
    assert settings.spine.max_axis_tilt_deg == 5.0
    assert settings.hip.roi_margin_vertical_cm == 3.0
    assert settings.hip.roi_margin_lateral_cm == 2.0
    assert settings.report.violation_separator == ";"


def test_env_var_selects_config_dir(config_copy, monkeypatch):
    (config_copy / "pixel_spacing.yaml").write_text("row_mm: 2.0\ncolumn_mm: 1.0\n", encoding="utf-8")
    monkeypatch.setenv(CONFIG_DIR_ENV, str(config_copy))
    assert load_settings().pixel_spacing.row_mm == 2.0


def test_missing_file_raises(config_copy):
    (config_copy / "thresholds.yaml").unlink()
    with pytest.raises(ConfigError, match="not found"):
        load_settings(config_copy)


def test_unknown_key_is_rejected(config_copy):
    path = config_copy / "pixel_spacing.yaml"
    path.write_text(path.read_text(encoding="utf-8") + "colum_mm: 0.6\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="colum_mm"):
        load_settings(config_copy)


def test_out_of_range_value_is_rejected(config_copy):
    (config_copy / "pixel_spacing.yaml").write_text("row_mm: -1\ncolumn_mm: 0.6\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_settings(config_copy)


def test_yaml_object_tags_are_not_executed(config_copy):
    # yaml.load with an unsafe loader would instantiate this; safe_load must refuse.
    (config_copy / "pixel_spacing.yaml").write_text("!!python/object/apply:os.system ['echo x']\n")
    with pytest.raises(ConfigError, match="Invalid YAML"):
        load_settings(config_copy)


def test_non_mapping_top_level_rejected(config_copy):
    (config_copy / "pixel_spacing.yaml").write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        load_settings(config_copy)
