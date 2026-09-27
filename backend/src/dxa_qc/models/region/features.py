"""Pixel-only features for the region/side classifier.

HOG on a fixed-size downsample captures the coarse bone layout (a central vertebral column vs.
an off-centre femoral shaft with the pelvis on one side), which is all this task needs.
Parameters are part of the model contract and are stored alongside the weights.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np
from skimage.feature import hog


@dataclass(frozen=True, slots=True)
class HogParams:
    size: int = 64
    orientations: int = 9
    pixels_per_cell: int = 8
    cells_per_block: int = 2

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def hog_features(pixels: np.ndarray, params: HogParams) -> np.ndarray:
    """``pixels`` is a 2-D float image in [0, 1]; aspect ratio is deliberately not preserved -
    image sizes vary per study and the classifier must not learn from them."""
    if pixels.ndim != 2:
        raise ValueError(f"expected a 2-D image, got shape {pixels.shape}")
    image = cv2.resize(pixels.astype(np.float32), (params.size, params.size), interpolation=cv2.INTER_AREA)
    return hog(
        image,
        orientations=params.orientations,
        pixels_per_cell=(params.pixels_per_cell, params.pixels_per_cell),
        cells_per_block=(params.cells_per_block, params.cells_per_block),
        feature_vector=True,
    ).astype(np.float64)
