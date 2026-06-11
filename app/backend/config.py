from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    project_root: Path = Path(__file__).resolve().parents[2]

    storage_root: Path = project_root / "storage" / "positions"
    output_root: Path = project_root / "outputs"
    debug_root: Path = output_root / "debug"
    preprocess_root: Path = output_root / "preprocessed"

    enable_alignment: bool = True
    enable_quadrants: bool = True
    enable_preprocessing: bool = True
    save_preprocessed: bool = True

    rsm_mode: str = "STRUCTURE"

    final_pass_score: float = 0.65
    absence_ssim_threshold: float = 0.40
    ssim_pass_threshold: float = 0.80
    edge_pass_threshold: float = 0.15

    max_defect_ratio: float = 0.01
    local_similarity_threshold: float = 0.98

    diff_threshold: int = 20
    min_blob_area: int = 5
    scratch_defect_ratio_threshold: float = 0.0001

    preprocess_brightness: int = 30
    preprocess_contrast: float = 3.0

    metric_weight_ssim: float = 0.25
    metric_weight_edge: float = 0.25
    metric_weight_mse: float = 0.15
    metric_weight_hist: float = 0.20
    metric_weight_grad: float = 0.15

    model_config = SettingsConfigDict(env_prefix="RSM_")


settings = Settings()

