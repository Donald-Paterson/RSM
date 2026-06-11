from pathlib import Path

import cv2
import numpy as np

from app.vision.image_utils import ensure_uint8


def preprocess_roi(
    roi: np.ndarray,
    save: bool = False,
    save_path: str | Path | None = None,
) -> np.ndarray:
    """Create a balanced monochrome ROI that preserves visible scratches."""
    if roi.ndim == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = ensure_uint8(roi)

    clahe = cv2.createCLAHE(
        clipLimit=1.5,
        tileGridSize=(12, 12),
    )
    contrast = clahe.apply(gray)

    base = np.clip(
        contrast.astype(np.float32) * 0.82 - 8.0,
        0,
        255,
    )

    defect_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (15, 15),
    )

    dark_scratches = cv2.morphologyEx(
        contrast,
        cv2.MORPH_BLACKHAT,
        defect_kernel,
    ).astype(np.float32)

    bright_scratches = cv2.morphologyEx(
        contrast,
        cv2.MORPH_TOPHAT,
        defect_kernel,
    ).astype(np.float32)

    enhanced = base + 0.85 * bright_scratches - 0.85 * dark_scratches
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

    blurred = cv2.GaussianBlur(
        enhanced,
        (0, 0),
        sigmaX=1.0,
    )

    final = cv2.addWeighted(
        enhanced,
        1.35,
        blurred,
        -0.35,
        0,
    )
    final = ensure_uint8(final)

    if save and save_path is not None:
        destination = Path(save_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(destination), final)

    return final
