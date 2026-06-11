import os
import cv2
import numpy as np

from utils import load_image

from alignment import align_images

from roi_extractor import (

    load_annotations,

    extract_polygon_rois,

    divide_into_quadrants
)

from preprocessing import preprocess_roi

from similarity import *

from decision_engine import *

from config import *

from roi_debugger import save_roi_debug


# =========================================================
# SHIFT-TOLERANT SSIM
# =========================================================

def best_shift_match(ref, test):

    best_score = -1

    for dx in range(-2, 3):

        for dy in range(-2, 3):

            shifted = np.roll(

                test,

                shift=(dy, dx),

                axis=(0,1)
            )

            score = masked_ssim(

                ref,
                shifted
            )

            if score > best_score:

                best_score = score

    return best_score


# =========================================================
# MAIN VALIDATION
# =========================================================

def validate(image_path):

    # =====================================================
    # LOAD IMAGES
    # =====================================================

    ref = load_image(
        REFERENCE_IMAGE
    )

    test = load_image(
        image_path
    )

    # =====================================================
    # ALIGNMENT
    # =====================================================

    if ENABLE_ALIGNMENT:

        aligned = align_images(
            ref,
            test
        )

    else:

        aligned = test

    # =====================================================
    # LOAD ROIs
    # =====================================================

    if ANNOTATION_TYPE == "XML":

        components = load_annotations(
            REFERENCE_LABEL
        )

        ref_rois = extract_polygon_rois(
            ref,
            components
        )

        test_rois = extract_polygon_rois(
            aligned,
            components
        )

    else:

        # =====================================================
        # LOAD XML COMPONENTS
        # =====================================================

        components = load_annotations(
            REFERENCE_LABEL
        )

        # =====================================================
        # EXTRACT POLYGON ROIS
        # =====================================================

        ref_rois = extract_polygon_rois(

            ref,
            components
        )

        test_rois = extract_polygon_rois(

            aligned,
            components
        )

    # =====================================================
    # RESULTS
    # =====================================================

    results = []

    image_name = os.path.basename(
        image_path
    )

    # =====================================================
    # ROI LOOP
    # =====================================================

    for i, (ref_data, test_data) in enumerate(

    zip(ref_rois, test_rois)
    ):

        roi_ref = ref_data["roi"]

        roi_test = test_data["roi"]

        label = ref_data["label"]

        bbox = ref_data["bbox"]

        ref_quads = divide_into_quadrants(
            roi_ref
        )

        test_quads = divide_into_quadrants(
            roi_test
        )

        ...

        # =================================================
        # SHAPE FIX
        # =================================================

        if roi_ref.shape != roi_test.shape:

            roi_test = cv2.resize(

                roi_test,

                (
                    roi_ref.shape[1],
                    roi_ref.shape[0]
                )
            )

        # =================================================
        # QUADRANT MODE
        # =================================================

        if ENABLE_QUADRANTS:

            ref_parts = divide_into_quadrants(
                roi_ref
            )

            test_parts = divide_into_quadrants(
                roi_test
            )

        else:

            ref_parts = [roi_ref]

            test_parts = [roi_test]

        # =================================================
        # QUADRANT LOOP
        # =================================================

        for q_idx, (rq, tq) in enumerate(

            zip(ref_parts, test_parts)
        ):

            # =============================================
            # EMPTY CHECK
            # =============================================

            if rq.size == 0 or tq.size == 0:

                continue

            # =============================================
            # SHAPE FIX
            # =============================================

            if rq.shape != tq.shape:

                tq = cv2.resize(

                    tq,

                    (
                        rq.shape[1],
                        rq.shape[0]
                    )
                )

            # =============================================
            # PREPROCESSING
            # =============================================

            if ENABLE_PREPROCESSING:

                rq = preprocess_roi(

                    rq,

                    save=SAVE_PREPROCESSED,

                    save_name=f"{image_name}_roi{i}_Q{q_idx+1}_golden.png"
                )

                tq = preprocess_roi(

                    tq,

                    save=SAVE_PREPROCESSED,

                    save_name=f"{image_name}_roi{i}_Q{q_idx+1}_test.png"
                )

            # =============================================
            # STRUCTURE MODE
            # =============================================

            if RSM_MODE == "STRUCTURE":

                s = best_shift_match(
                    rq,
                    tq
                )

                e = edge_difference(
                    rq,
                    tq
                )

                m = mse_score(
                    rq,
                    tq
                )

                h = histogram_similarity(
                    rq,
                    tq
                )

                g = gradient_difference(
                    rq,
                    tq
                )

                similarity, ratio, diff_map = \
                    local_difference_score(
                        rq,
                        tq
                    )

                score = compute_score(

                    s,
                    e,
                    m,
                    h,
                    g,
                    i
                )

                # =====================================================
                # FINAL DECISION
                # =====================================================

                if RSM_MODE == "STRUCTURE":

                    verdict = decide_structure(

                        score,
                        s,
                        e,
                        ratio
                    )

                else:

                    verdict = decide_defect(

                        ratio,
                        similarity
                    )

            # =============================================
            # DEFECT MODE
            # =============================================

            else:

                similarity, ratio, diff_map = \
                    local_difference_score(
                        rq,
                        tq
                    )

                s = similarity

                e = ratio

                m = 0

                h = 0

                g = 0

                score = similarity

                if ratio > MAX_DEFECT_RATIO:

                    verdict = "FAIL"

                else:

                    verdict = "PASS"

            # =============================================
            # DEBUG SAVE
            # =============================================

            golden_img, test_img, diff_img = \
                save_roi_debug(

                    rq,
                    tq,

                    f"{i}_Q{q_idx+1}",

                    image_name
                )

            # =============================================
            # RESULT APPEND
            # =============================================

            results.append({

                "Image Name": image_name,

                "ROI Index": i,

                "Quadrant": q_idx + 1,

                "Class Label": label,

                "SSIM": round(s, 4),

                "Edge Difference": round(e, 4),

                "MSE": round(m, 4),

                "Histogram Score": round(h, 4),

                "Gradient Difference": round(g, 4),

                "Local Difference": round(ratio, 4),

                "Final Score": round(score, 4),

                "Prediction": verdict,

                "Golden ROI": golden_img,

                "Test ROI": test_img,

                "Diff Map": diff_img
            })

    return results

