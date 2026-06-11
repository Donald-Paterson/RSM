from app.backend.config import settings


ROI_METRIC_WEIGHTS = {
    0: {
        "ssim": 0.20,
        "edge": 0.30,
        "mse": 0.15,
        "hist": 0.20,
        "grad": 0.15,
    }
}


def compute_score(
    ssim_value: float,
    edge_value: float,
    mse_value: float,
    hist_value: float,
    grad_value: float,
    roi_index: int,
) -> float:
    weights = ROI_METRIC_WEIGHTS.get(
        roi_index,
        {
            "ssim": settings.metric_weight_ssim,
            "edge": settings.metric_weight_edge,
            "mse": settings.metric_weight_mse,
            "hist": settings.metric_weight_hist,
            "grad": settings.metric_weight_grad,
        },
    )

    score = (
        weights["ssim"] * ssim_value
        + weights["edge"] * (1 - edge_value)
        + weights["mse"] * (1 / (1 + mse_value))
        + weights["hist"] * hist_value
        + weights["grad"] * (1 / (1 + grad_value))
    )

    return float(score)


def decide_structure(
    score: float,
    ssim_value: float,
    edge_value: float,
    local_difference: float = 0.0,
) -> str:
    if ssim_value < settings.absence_ssim_threshold:
        return "ABSENT"

    if local_difference > settings.scratch_defect_ratio_threshold:
        return "FAIL"

    if ssim_value > settings.ssim_pass_threshold and edge_value < settings.edge_pass_threshold:
        return "PASS"

    if score >= settings.final_pass_score:
        return "PASS"

    return "FAIL"


def decide_defect(blob_ratio: float, local_similarity: float) -> str:
    if blob_ratio > settings.max_defect_ratio:
        return "FAIL"

    if local_similarity < settings.local_similarity_threshold:
        return "FAIL"

    return "PASS"


