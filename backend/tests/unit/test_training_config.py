from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dxa_qc.domain.taxonomy import AnatomicalRegion
from dxa_qc.models.quality.embeddings import Crop
from dxa_qc.training.config import DEFAULT_CONFIG, TrainingConfigError, load_training_config
from dxa_qc.training.quality import DESIGNS, design_from_config


def write(tmp_path: Path, data: object) -> Path:
    path = tmp_path / "training.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def base() -> dict:
    return yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))


def test_default_config_reproduces_shipped_designs() -> None:
    config = load_training_config()
    for region in AnatomicalRegion:
        design = design_from_config(region, config)
        assert design.backbones == DESIGNS[region].backbones
        assert design.geometry_c == DESIGNS[region].geometry_c
        assert design.embedding_c == DESIGNS[region].embedding_c


def test_crop_and_defaults(tmp_path: Path) -> None:
    data = base()
    data["regions"]["spine"]["backbones"] = [{"key": "r18", "timm": "resnet18.a1_in1k", "input_size": 224}]
    del data["cv"]
    config = load_training_config(write(tmp_path, data))
    assert config.cv.n_splits == 5 and config.cv.repetitions == 20
    (spec,) = config.backbones(AnatomicalRegion.SPINE)
    assert spec.crop == Crop() and spec.weights_file == "resnet18.a1_in1k.safetensors"
    hip = {s.key: s for s in config.backbones(AnatomicalRegion.HIP)}
    assert hip["dinov2s_trochanter"].crop == Crop(0.35, 0.8, 0.15, 0.75)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(unknown=1),  # typos must not be silently ignored
        lambda d: d["cv"].update(n_splits=1),
        lambda d: d["heads"].update(geometry_c=[]),
        lambda d: d["heads"].update(embedding_c=[0.1, -1]),
        lambda d: d["regions"].pop("hip"),
        lambda d: d["regions"].update(knee={"backbones": []}),
        lambda d: d["regions"]["spine"]["backbones"][0].update(timm="../../evil"),
        lambda d: d["regions"]["spine"]["backbones"][0].update(key="Bad Key"),
        lambda d: d["regions"]["spine"]["backbones"][0].update(input_size=4096),
        lambda d: d["regions"]["spine"]["backbones"][0].update(crop=[0.5, 0.4, 0, 1]),
        lambda d: d["regions"]["spine"]["backbones"].append(dict(d["regions"]["spine"]["backbones"][0])),
    ],
)
def test_invalid_configs_are_rejected(tmp_path: Path, mutate) -> None:
    data = base()
    mutate(data)
    with pytest.raises(TrainingConfigError):
        load_training_config(write(tmp_path, data))


def test_missing_and_malformed_files(tmp_path: Path) -> None:
    with pytest.raises(TrainingConfigError, match="not found"):
        load_training_config(tmp_path / "nope.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("seed: [unclosed", encoding="utf-8")
    with pytest.raises(TrainingConfigError, match="invalid"):
        load_training_config(bad)
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(TrainingConfigError):
        load_training_config(empty)


def test_config_dir_env_is_respected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = base()
    data["seed"] = 7
    write(tmp_path, data)
    monkeypatch.setenv("DXA_QC_CONFIG_DIR", str(tmp_path))
    assert load_training_config().seed == 7
