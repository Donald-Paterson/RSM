from pathlib import Path

import cv2
import numpy as np


def load_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path))

    if image is None:
        raise ValueError(f"Failed to load image: {path}")

    return image


def ensure_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image

    normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    return normalized.astype(np.uint8)
