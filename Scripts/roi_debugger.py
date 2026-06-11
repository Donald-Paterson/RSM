import os
import cv2
import numpy as np

from config import DEBUG_FOLDER


def save_roi_debug(
    roi_ref,
    roi_test,
    roi_index,
    image_name,
    diff_map=None
):

    image_base = os.path.splitext(
        image_name
    )[0]

    roi_folder = os.path.join(
        DEBUG_FOLDER,
        image_base
    )

    os.makedirs(
        roi_folder,
        exist_ok=True
    )

    # ==========================================
    # FILE PATHS
    # ==========================================

    golden_path = os.path.join(
        roi_folder,
        f"roi_{roi_index}_golden.png"
    )

    test_path = os.path.join(
        roi_folder,
        f"roi_{roi_index}_test.png"
    )

    diff_path = os.path.join(
        roi_folder,
        f"roi_{roi_index}_diff.png"
    )

    # ==========================================
    # SAVE GOLDEN ROI
    # ==========================================

    cv2.imwrite(
        golden_path,
        roi_ref
    )

    # ==========================================
    # SAVE TEST ROI
    # ==========================================

    cv2.imwrite(
        test_path,
        roi_test
    )

    # ==========================================
    # SAVE DIFF MAP
    # ==========================================

    if diff_map is not None:

        if len(diff_map.shape) == 2:

            diff_vis = diff_map

        else:

            diff_vis = cv2.cvtColor(
                diff_map,
                cv2.COLOR_BGR2GRAY
            )

        cv2.imwrite(
            diff_path,
            diff_vis
        )

    else:

        diff = cv2.absdiff(
            roi_ref,
            roi_test
        )

        cv2.imwrite(
            diff_path,
            diff
        )

    return (
        golden_path,
        test_path,
        diff_path
    )