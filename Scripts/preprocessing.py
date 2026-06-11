import os

import cv2
import numpy as np

from config import PREPROCESS_FOLDER
from image_utils import ensure_uint8


def preprocess_roi(
    roi,
    save=False,
    save_name=None
):
    """
    Create a balanced monochrome ROI that preserves visible scratches.

    Bright scratches are enhanced and dark scratches are darkened without
    saturating the metallic surface or changing the existing pipeline API.
    """
    if roi.ndim == 3:
        gray = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2GRAY
        )
    else:
        gray = ensure_uint8(roi)

    # Gentle local contrast enhancement prevents bright metal saturation.
    clahe = cv2.createCLAHE(
        clipLimit=1.5,
        tileGridSize=(12, 12)
    )

    contrast = clahe.apply(gray)

    # Slightly darken the base image to retain surface texture and headroom.
    base = np.clip(
        contrast.astype(np.float32) * 0.82 - 8.0,
        0,
        255
    )

    # Extract scratch-like details across a larger local neighbourhood.
    defect_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (15, 15)
    )

    dark_scratches = cv2.morphologyEx(
        contrast,
        cv2.MORPH_BLACKHAT,
        defect_kernel
    ).astype(np.float32)

    bright_scratches = cv2.morphologyEx(
        contrast,
        cv2.MORPH_TOPHAT,
        defect_kernel
    ).astype(np.float32)

    # Preserve polarity: dark scratches become darker and bright scratches
    # become brighter instead of making the whole surface brighter.
    enhanced = (
        base
        + 0.85 * bright_scratches
        - 0.85 * dark_scratches
    )

    enhanced = np.clip(
        enhanced,
        0,
        255
    ).astype(np.uint8)

    # Gentle unsharp masking reveals fine lines without washing out the ROI.
    blurred = cv2.GaussianBlur(
        enhanced,
        (0, 0),
        sigmaX=1.0
    )

    final = cv2.addWeighted(
        enhanced,
        1.35,
        blurred,
        -0.35,
        0
    )

    final = ensure_uint8(final)

    if save and save_name:
        os.makedirs(
            PREPROCESS_FOLDER,
            exist_ok=True
        )

        save_path = os.path.join(
            PREPROCESS_FOLDER,
            save_name
        )

        cv2.imwrite(
            save_path,
            final
        )

    return final
