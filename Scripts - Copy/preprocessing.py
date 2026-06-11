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

    gray = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY
    )

    enhanced = cv2.convertScaleAbs(
        gray,
        alpha=3.0,
        beta=30
    )

    kernel = np.array([

        [-1,-1,-1],
        [-1, 9,-1],
        [-1,-1,-1]

    ])

    sharpened = cv2.filter2D(
        enhanced,
        -1,
        kernel
    )

    final = ensure_uint8(
        sharpened
    )

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