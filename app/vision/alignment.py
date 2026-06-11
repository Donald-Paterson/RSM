import cv2
import numpy as np


def orb_alignment(reference: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, bool]:
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(5000)
    kp_ref, des_ref = orb.detectAndCompute(ref_gray, None)
    kp_test, des_test = orb.detectAndCompute(test_gray, None)

    if des_ref is None or des_test is None:
        return test, False

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des_ref, des_test)

    if len(matches) < 10:
        return test, False

    matches = sorted(matches, key=lambda match: match.distance)[:200]

    points_ref = np.float32([kp_ref[match.queryIdx].pt for match in matches])
    points_test = np.float32([kp_test[match.trainIdx].pt for match in matches])

    homography, _ = cv2.findHomography(points_test, points_ref, cv2.RANSAC, 5)

    if homography is None:
        return test, False

    aligned = cv2.warpPerspective(test, homography, (reference.shape[1], reference.shape[0]))
    return aligned, True


def ecc_alignment(reference: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, bool]:
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype("float32")
    test_gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY).astype("float32")

    warp = np.eye(2, 3, dtype="float32")
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-6)

    try:
        _, warp = cv2.findTransformECC(ref_gray, test_gray, warp, cv2.MOTION_AFFINE, criteria)
        aligned = cv2.warpAffine(
            test,
            warp,
            (reference.shape[1], reference.shape[0]),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        )
        return aligned, True
    except cv2.error:
        return test, False


def align_images(reference: np.ndarray, test: np.ndarray) -> np.ndarray:
    orb_aligned, _ = orb_alignment(reference, test)
    ecc_aligned, _ = ecc_alignment(reference, orb_aligned)
    return ecc_aligned
