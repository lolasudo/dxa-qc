from __future__ import annotations

import numpy as np
import pytest

from dxa_qc.domain import AnatomicalRegion, HipSide
from dxa_qc.models.region import HogParams, ModelFormatError, RegionClass, RegionClassifier, hog_features


def synthetic(label: RegionClass, rng: np.random.Generator, h: int = 120, w: int = 110) -> np.ndarray:
    """Cartoon of the real anatomy: vertebral column in the middle vs. femoral shaft + pelvis on one side."""
    img = rng.normal(0.08, 0.03, (h, w))
    jitter = int(rng.integers(-5, 6))
    if label is RegionClass.SPINE:
        c = w // 2 + jitter
        img[:, c - 12 : c + 12] = 0.7
        img[::14, c - 12 : c + 12] = 0.3  # intervertebral gaps
    else:
        # Right hip in radiological convention: shaft on the image left, pelvis on the image right.
        shaft = int(w * 0.3) + jitter
        img[h // 3 :, shaft - 10 : shaft + 10] = 0.8
        img[: h // 3, int(w * 0.55) : int(w * 0.95)] = 0.6
        if label is RegionClass.HIP_LEFT:
            img = img[:, ::-1]
    # Real studies vary in size; the model must cope with that.
    return np.clip(img, 0, 1).astype(np.float32)


def dataset(n_per_class: int, seed: int, classes=tuple(RegionClass)):
    rng = np.random.default_rng(seed)
    labels = [c for c in classes for _ in range(n_per_class)]
    return [synthetic(c, rng) for c in labels], labels


@pytest.fixture(scope="module")
def trained() -> RegionClassifier:
    images, labels = dataset(15, seed=1)
    return RegionClassifier.fit(images, labels, seed=0)


def test_class_properties():
    assert RegionClass.SPINE.region is AnatomicalRegion.SPINE and RegionClass.SPINE.side is None
    assert RegionClass.HIP_LEFT.side is HipSide.LEFT
    assert RegionClass.HIP_LEFT.mirrored is RegionClass.HIP_RIGHT
    assert RegionClass.SPINE.mirrored is RegionClass.SPINE


def test_features_fixed_length_regardless_of_input_size():
    p = HogParams()
    a = hog_features(np.zeros((300, 280), np.float32), p)
    b = hog_features(np.zeros((207, 300), np.float32), p)
    assert a.shape == b.shape


def test_features_reject_non_2d():
    with pytest.raises(ValueError):
        hog_features(np.zeros((10, 10, 3)), HogParams())


def test_generalizes_to_unseen_samples(trained):
    images, labels = dataset(10, seed=99)
    preds = [trained.predict(im) for im in images]
    assert [p.label for p in preds] == labels
    assert all(abs(sum(p.probabilities.values()) - 1) < 1e-9 for p in preds)
    assert all(p.confidence == max(p.probabilities.values()) for p in preds)


def test_mirror_augmentation_teaches_unseen_side():
    # Only left hips (and spines) in training: right hips must be learned from the mirrored copies.
    images, labels = dataset(15, seed=2, classes=(RegionClass.SPINE, RegionClass.HIP_LEFT))
    model = RegionClassifier.fit(images, labels, seed=0, mirror_augment=True)
    test_imgs, _ = dataset(5, seed=3, classes=(RegionClass.HIP_RIGHT,))
    assert all(model.predict(im).label is RegionClass.HIP_RIGHT for im in test_imgs)


def test_fit_is_deterministic():
    images, labels = dataset(8, seed=4)
    a = RegionClassifier.fit(images, labels, seed=0)
    b = RegionClassifier.fit(images, labels, seed=0)
    np.testing.assert_array_equal(a.coef, b.coef)


def test_fit_validates_inputs():
    with pytest.raises(ValueError):
        RegionClassifier.fit([], [])


def test_save_load_roundtrip(trained, tmp_path):
    path = tmp_path / "region.npz"
    trained.save(path, metadata={"cv_accuracy": 1.0})
    loaded = RegionClassifier.load(path)
    img = synthetic(RegionClass.HIP_LEFT, np.random.default_rng(5))
    np.testing.assert_allclose(loaded.predict_proba(img), trained.predict_proba(img))
    assert loaded.classes == trained.classes


def test_load_refuses_pickled_payload(tmp_path):
    path = tmp_path / "evil.npz"
    np.savez(
        path,
        mean=np.array([object()], dtype=object),
        scale=np.ones(1),
        coef=np.ones((1, 1)),
        intercept=np.ones(1),
        meta=np.array("{}"),
    )
    with pytest.raises(ModelFormatError):
        RegionClassifier.load(path)


def test_load_rejects_wrong_version(trained, tmp_path):
    path = tmp_path / "m.npz"
    trained.save(path, metadata={"format_version": 999})
    with pytest.raises(ModelFormatError, match="version"):
        RegionClassifier.load(path)


def test_load_missing_and_corrupted(tmp_path):
    with pytest.raises(ModelFormatError, match="not found"):
        RegionClassifier.load(tmp_path / "none.npz")
    bad = tmp_path / "bad.npz"
    bad.write_bytes(b"garbage")
    with pytest.raises(ModelFormatError):
        RegionClassifier.load(bad)


def test_inconsistent_shapes_rejected():
    with pytest.raises(ModelFormatError, match="inconsistent"):
        RegionClassifier(
            list(RegionClass), np.zeros(4), np.ones(4), np.zeros((3, 5)), np.zeros(3), HogParams()
        )


def test_non_positive_scale_rejected():
    with pytest.raises(ModelFormatError, match="non-positive"):
        RegionClassifier(
            list(RegionClass), np.zeros(4), np.zeros(4), np.zeros((3, 4)), np.zeros(3), HogParams()
        )
