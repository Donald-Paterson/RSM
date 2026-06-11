import cv2
import numpy as np


def ensure_uint8(img):

    if img.dtype != np.uint8:

        img = cv2.normalize(
            img,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        )

        img = img.astype(np.uint8)

    return img