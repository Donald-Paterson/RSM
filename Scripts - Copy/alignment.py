import cv2
import numpy as np


def orb_alignment(reference, test):

    ref_gray = cv2.cvtColor(
        reference,
        cv2.COLOR_BGR2GRAY
    )

    test_gray = cv2.cvtColor(
        test,
        cv2.COLOR_BGR2GRAY
    )

    orb = cv2.ORB_create(5000)

    kp1, des1 = orb.detectAndCompute(
        ref_gray,
        None
    )

    kp2, des2 = orb.detectAndCompute(
        test_gray,
        None
    )

    if des1 is None or des2 is None:
        return test, False

    matcher = cv2.BFMatcher(
        cv2.NORM_HAMMING,
        crossCheck=True
    )

    matches = matcher.match(des1, des2)

    if len(matches) < 10:
        return test, False

    matches = sorted(
        matches,
        key=lambda x: x.distance
    )[:200]

    pts_ref = np.float32([
        kp1[m.queryIdx].pt
        for m in matches
    ])

    pts_test = np.float32([
        kp2[m.trainIdx].pt
        for m in matches
    ])

    H, _ = cv2.findHomography(
        pts_test,
        pts_ref,
        cv2.RANSAC,
        5
    )

    if H is None:
        return test, False

    aligned = cv2.warpPerspective(
        test,
        H,
        (
            reference.shape[1],
            reference.shape[0]
        )
    )

    return aligned, True


def ecc_alignment(reference, test):

    ref_gray = cv2.cvtColor(
        reference,
        cv2.COLOR_BGR2GRAY
    ).astype("float32")

    test_gray = cv2.cvtColor(
        test,
        cv2.COLOR_BGR2GRAY
    ).astype("float32")

    warp = np.eye(
        2,
        3,
        dtype="float32"
    )

    criteria = (
        cv2.TERM_CRITERIA_EPS |
        cv2.TERM_CRITERIA_COUNT,
        5000,
        1e-6
    )

    try:

        _, warp = cv2.findTransformECC(
            ref_gray,
            test_gray,
            warp,
            cv2.MOTION_AFFINE,
            criteria
        )

        aligned = cv2.warpAffine(
            test,
            warp,
            (
                reference.shape[1],
                reference.shape[0]
            ),
            flags=cv2.INTER_LINEAR +
                  cv2.WARP_INVERSE_MAP
        )

        return aligned, True

    except cv2.error:

        return test, False


def align_images(reference, test):

    orb_aligned, _ = orb_alignment(
        reference,
        test
    )

    ecc_aligned, _ = ecc_alignment(
        reference,
        orb_aligned
    )

    return ecc_aligned