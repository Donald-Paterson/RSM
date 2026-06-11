from config import *


# =========================================================
# FINAL SCORE COMPUTATION
# =========================================================

def compute_score(

    ssim_v,
    edge_v,
    mse_v,
    hist_v,
    grad_v,
    roi_index
):

    # =====================================================
    # ROI-SPECIFIC WEIGHTS
    # =====================================================

    weights = ROI_METRIC_WEIGHTS.get(

        roi_index,

        {

            "ssim": W_SSIM,

            "edge": W_EDGE,

            "mse": W_MSE,

            "hist": W_HIST,

            "grad": W_GRAD
        }
    )

    # =====================================================
    # NORMALIZATION
    # =====================================================

    # HIGHER = BETTER

    ssim_n = ssim_v

    edge_n = 1 - edge_v

    mse_n = 1 / (1 + mse_v)

    hist_n = hist_v

    grad_n = 1 / (1 + grad_v)

    # =====================================================
    # FINAL WEIGHTED SCORE
    # =====================================================

    score = (

        weights["ssim"] * ssim_n +

        weights["edge"] * edge_n +

        weights["mse"] * mse_n +

        weights["hist"] * hist_n +

        weights["grad"] * grad_n
    )

    return score


# =========================================================
# STRUCTURE MODE DECISION
# =========================================================

def decide_structure(

    score,
    ssim_val,
    edge_val,
    local_difference=0
):

    # =====================================================
    # ABSENCE CHECK
    # =====================================================

    if ssim_val < ABSENCE_SSIM_THRESHOLD:

        return "ABSENT"

    # =====================================================
    # SCRATCH / SURFACE DEFECT CHECK
    # =====================================================

    if local_difference > SCRATCH_DEFECT_RATIO_THRESHOLD:

        return "FAIL"

    # =====================================================
    # STRONG STRUCTURE MATCH
    # =====================================================

    if (

        ssim_val > SSIM_PASS_THRESHOLD and

        edge_val < EDGE_PASS_THRESHOLD
    ):

        return "PASS"

    # =====================================================
    # FINAL SCORE CHECK
    # =====================================================

    if score >= FINAL_PASS_SCORE:

        return "PASS"

    return "FAIL"


# =========================================================
# DEFECT MODE DECISION
# =========================================================

def decide_defect(

    blob_ratio,
    local_similarity
):

    # =====================================================
    # HEAVY DEFECT
    # =====================================================

    if blob_ratio > MAX_DEFECT_RATIO:

        return "FAIL"

    # =====================================================
    # LOCAL STRUCTURE DAMAGE
    # =====================================================

    if local_similarity < LOCAL_SIMILARITY_THRESHOLD:

        return "FAIL"

    return "PASS"

