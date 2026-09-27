from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dxa_qc.models.quality.embeddings import (
    ALL_SPECS,
    EFFNET_B3,
    BackboneError,
    Crop,
    EmbeddingExtractor,
    square_input,
)

BACKBONES = Path(__file__).resolve().parents[2] / "weights" / "backbones"
torch = pytest.importorskip("torch")


def test_crop_validation_and_apply() -> None:
    with pytest.raises(ValueError):
        Crop(0.5, 0.4)
    img = np.arange(100, dtype=np.float32).reshape(10, 10)
    part = Crop(0.2, 0.5, 0.0, 0.3).apply(img)
    assert part.shape == (3, 3)
    assert part[0, 0] == 20


def test_square_input_keeps_aspect_by_padding() -> None:
    tall = np.ones((200, 100), np.float32)
    out = square_input(tall, Crop(), 64)
    assert out.shape == (64, 64)
    # 100/200 of the width is image, the rest is zero padding split evenly
    assert out[:, 16:48].min() > 0.99
    assert out[:, :14].max() == 0 and out[:, 50:].max() == 0


def test_specs_have_unique_keys_and_one_size_per_weights_file() -> None:
    sizes: dict[str, int] = {}
    for spec in ALL_SPECS.values():
        assert sizes.setdefault(spec.timm_name, spec.input_size) == spec.input_size


def test_missing_weights_raise_backbone_error(tmp_path: Path) -> None:
    extractor = EmbeddingExtractor(tmp_path, device="cpu")
    with pytest.raises(BackboneError, match="not found"):
        extractor.embed([np.zeros((50, 50), np.float32)], EFFNET_B3)


def test_corrupted_weights_raise_backbone_error(tmp_path: Path) -> None:
    (tmp_path / EFFNET_B3.weights_file).write_bytes(b"garbage")
    with pytest.raises(BackboneError):
        EmbeddingExtractor(tmp_path, device="cpu").embed([np.zeros((50, 50), np.float32)], EFFNET_B3)


@pytest.mark.skipif(
    not (BACKBONES / EFFNET_B3.weights_file).is_file(), reason="backbone weights not exported"
)
def test_real_backbone_is_deterministic_and_device_independent() -> None:
    rng = np.random.default_rng(0)
    images = [rng.random((300, 180)).astype(np.float32) for _ in range(3)]
    cpu = EmbeddingExtractor(BACKBONES, device="cpu")
    a = cpu.embed(images, EFFNET_B3)
    b = cpu.embed(images, EFFNET_B3, batch_size=1)  # batching must not change results
    assert a.shape == (3, 1536)
    assert np.allclose(a, b, atol=1e-4)
    if torch.cuda.is_available():
        extractor = EmbeddingExtractor(BACKBONES, device="cuda")
        gpu = extractor.embed(images, EFFNET_B3)
        assert np.array_equal(gpu, extractor.embed(images, EFFNET_B3))  # bit-identical rerun
        # TF32 off: ~1e-6 relative to CPU (with the TF32 default it is ~2e-3).
        assert np.abs(a - gpu).max() / np.abs(a).max() < 1e-4


def test_extractor_pins_gpu_numerics(tmp_path: Path) -> None:
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    EmbeddingExtractor(tmp_path, device="cpu")
    assert not torch.backends.cudnn.allow_tf32 and not torch.backends.cuda.matmul.allow_tf32
    assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark
