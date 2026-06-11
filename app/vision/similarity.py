import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from app.backend.config import settings
from app.vision.image_utils import ensure_uint8


def ssim_score(a: np.ndarray, b: np.ndarray) -> float:
    a = ensure_uint8(a)
    b = ensure_uint8(b)
    score, _ = ssim(a, b, full=True)
    return float(score)


def masked_ssim(a: np.ndarray, b: np.ndarray) -> float:
    a = ensure_uint8(a)
    b = ensure_uint8(b)

    edges = cv2.Canny(a, 50, 150)
    mask = edges > 0

    score, diff = ssim(a, b, full=True)
    if mask.sum() == 0:
        return float(score)

    return float(diff[mask].mean())


def best_shift_match(reference: np.ndarray, test: np.ndarray) -> float:
    best_score = -1.0

    for dx in range(-2, 3):
        for dy in range(-2, 3):
            shifted = np.roll(test, shift=(dy, dx), axis=(0, 1))
            score = masked_ssim(reference, shifted)
            best_score = max(best_score, score)

    return best_score


def edge_difference(a: np.ndarray, b: np.ndarray) -> float:
    a = ensure_uint8(a)
    b = ensure_uint8(b)

    edges_a = cv2.Canny(a, 50, 150)
    edges_b = cv2.Canny(b, 50, 150)

    return float(np.sum(edges_a != edges_b) / edges_a.size)


def mse_score(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a.astype("float") - b.astype("float")) ** 2))


def histogram_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = ensure_uint8(a)
    b = ensure_uint8(b)

    hist_a = cv2.calcHist([a], [0], None, [256], [0, 256])
    hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])

    cv2.normalize(hist_a, hist_a)
    cv2.normalize(hist_b, hist_b)

    return float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL))


def gradient_difference(a: np.ndarray, b: np.ndarray) -> float:
    grad_x_a = cv2.Sobel(a, cv2.CV_64F, 1, 0)
    grad_y_a = cv2.Sobel(a, cv2.CV_64F, 0, 1)
    grad_x_b = cv2.Sobel(b, cv2.CV_64F, 1, 0)
    grad_y_b = cv2.Sobel(b, cv2.CV_64F, 0, 1)

    grad_a = np.sqrt(grad_x_a**2 + grad_y_a**2)
    grad_b = np.sqrt(grad_x_b**2 + grad_y_b**2)

    return float(np.mean(np.abs(grad_a - grad_b)))


def local_difference_analysis(
    a: np.ndarray,
    b: np.ndarray,
) -> tuple[float, float, np.ndarray, list[tuple[int, int, int, int]]]:
    """Return the established local-difference metrics plus valid contour boxes."""
    a = ensure_uint8(a)
    b = ensure_uint8(b)

    diff = cv2.absdiff(a, b)
    _, threshold = cv2.threshold(diff, settings.diff_threshold, 255, cv2.THRESH_BINARY)
    threshold = ensure_uint8(threshold)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # Opening removes thin scratches. Closing and one dilation pass preserve
    # fragmented hairline defects for contour and bounding-box detection.
    threshold = cv2.morphologyEx(
        threshold,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1,
    )
    threshold = cv2.dilate(
        threshold,
        kernel,
        iterations=1,
    )

    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    total_area = 0.0
    valid_contours = []
    bounding_boxes = []

    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, width, height = cv2.boundingRect(contour)
        short_side = max(1, min(width, height))
        elongation = max(width, height) / short_side

        # Keep regular defects by area and preserve long, thin scratches.
        if area < settings.min_blob_area and elongation < 3.0:
            continue

        total_area += area
        valid_contours.append(contour)
        bounding_boxes.append((x, y, width, height))

    diff_map = cv2.cvtColor(threshold, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(diff_map, valid_contours, -1, (255, 255, 255), 2)

    image_area = a.shape[0] * a.shape[1]
    ratio = total_area / image_area if image_area else 1.0
    similarity = 1 - ratio

    return float(similarity), float(ratio), diff_map, bounding_boxes


def local_difference_score(a: np.ndarray, b: np.ndarray) -> tuple[float, float, np.ndarray]:
    similarity, ratio, diff_map, _ = local_difference_analysis(a, b)
    return similarity, ratio, diff_map


