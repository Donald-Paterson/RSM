import cv2
import numpy as np

from skimage.metrics import structural_similarity as ssim

from image_utils import ensure_uint8

from config import (

    DIFF_THRESHOLD,
    MIN_BLOB_AREA
)


# =========================================================
# BASIC SSIM
# =========================================================

def ssim_score(a, b):

    a = ensure_uint8(a)
    b = ensure_uint8(b)

    score, _ = ssim(

        a,
        b,

        full=True
    )

    return score


# =========================================================
# EDGE-MASKED SSIM
# FOCUSES ON STRUCTURAL AREAS
# =========================================================

def masked_ssim(a, b):

    a = ensure_uint8(a)
    b = ensure_uint8(b)

    edges = cv2.Canny(

        a,

        50,
        150
    )

    mask = edges > 0

    score, diff = ssim(

        a,
        b,

        full=True
    )

    # =====================================================
    # NO EDGE CASE
    # =====================================================

    if mask.sum() == 0:

        return score

    masked_diff = diff[mask]

    return float(
        masked_diff.mean()
    )


# =========================================================
# EDGE DIFFERENCE
# =========================================================

def edge_difference(a, b):

    a = ensure_uint8(a)
    b = ensure_uint8(b)

    e1 = cv2.Canny(a, 50, 150)

    e2 = cv2.Canny(b, 50, 150)

    diff = np.sum(
        e1 != e2
    )

    return diff / e1.size


# =========================================================
# MSE
# =========================================================

def mse_score(a, b):

    return np.mean(

        (a.astype("float") - b.astype("float")) ** 2
    )


# =========================================================
# HISTOGRAM SIMILARITY
# =========================================================

def histogram_similarity(a, b):

    h1 = cv2.calcHist(

        [a],
        [0],

        None,

        [256],

        [0,256]
    )

    h2 = cv2.calcHist(

        [b],
        [0],

        None,

        [256],

        [0,256]
    )

    cv2.normalize(h1, h1)

    cv2.normalize(h2, h2)

    return cv2.compareHist(

        h1,
        h2,

        cv2.HISTCMP_CORREL
    )


# =========================================================
# GRADIENT DIFFERENCE
# =========================================================

def gradient_difference(a, b):

    gx1 = cv2.Sobel(
        a,
        cv2.CV_64F,
        1,
        0
    )

    gy1 = cv2.Sobel(
        a,
        cv2.CV_64F,
        0,
        1
    )

    gx2 = cv2.Sobel(
        b,
        cv2.CV_64F,
        1,
        0
    )

    gy2 = cv2.Sobel(
        b,
        cv2.CV_64F,
        0,
        1
    )

    g1 = np.sqrt(
        gx1**2 + gy1**2
    )

    g2 = np.sqrt(
        gx2**2 + gy2**2
    )

    return np.mean(
        np.abs(g1 - g2)
    )


# =========================================================
# LOCAL DIFFERENCE ANALYSIS
# FOR DEFECT MODE
# =========================================================

def local_difference_score(a, b):

    a = ensure_uint8(a)
    b = ensure_uint8(b)

    # =====================================================
    # ABSOLUTE DIFFERENCE
    # =====================================================

    diff = cv2.absdiff(
        a,
        b
    )

    # =====================================================
    # THRESHOLD
    # =====================================================

    _, thresh = cv2.threshold(

        diff,

        DIFF_THRESHOLD,

        255,

        cv2.THRESH_BINARY
    )

    thresh = ensure_uint8(
        thresh
    )

    # =====================================================
    # MORPH CLEANUP
    # =====================================================

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3,3)
    )

    # Opening removes thin scratches, so preserve them with closing and
    # a single dilation pass that connects fragmented hairline pixels.
    thresh = cv2.morphologyEx(

        thresh,

        cv2.MORPH_CLOSE,

        kernel,

        iterations=1
    )

    thresh = cv2.dilate(

        thresh,

        kernel,

        iterations=1
    )

    # =====================================================
    # CONTOURS
    # =====================================================

    contours, _ = cv2.findContours(

        thresh,

        cv2.RETR_EXTERNAL,

        cv2.CHAIN_APPROX_SIMPLE
    )

    total_area = 0

    valid_contours = []

    # =====================================================
    # SMALL BLOB FILTERING
    # =====================================================

    for c in contours:

        area = cv2.contourArea(c)

        x, y, w, h = cv2.boundingRect(c)

        short_side = max(1, min(w, h))

        elongation = max(w, h) / short_side

        # Keep regular defects by area and preserve long, thin scratches.
        if area < MIN_BLOB_AREA and elongation < 3.0:

            continue

        total_area += area

        valid_contours.append(c)

    # =====================================================
    # VISUALIZATION MAP
    # =====================================================

    diff_map = cv2.cvtColor(

        thresh,

        cv2.COLOR_GRAY2BGR
    )

    cv2.drawContours(

        diff_map,

        valid_contours,

        -1,

        (255,255,255),

        2
    )

    # Draw a bounding box around every detected scratch or surface defect.
    for c in valid_contours:

        x, y, w, h = cv2.boundingRect(c)

        cv2.rectangle(

            diff_map,

            (x, y),

            (x + w - 1, y + h - 1),

            (255,255,255),

            1
        )

    # =====================================================
    # DEFECT RATIO
    # =====================================================

    image_area = a.shape[0] * a.shape[1]

    ratio = total_area / image_area

    similarity = 1 - ratio

    return (

        similarity,
        ratio,
        diff_map
    )
