from typing import List, Optional

from pydantic import BaseModel, Field


class GoldenUploadResponse(BaseModel):
    message: str
    project_id: str
    position: str
    path: str


class DefectBoundingBox(BaseModel):
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    width: int
    height: int
    roi_index: int
    quadrant: int
    class_label: str


class RoiResult(BaseModel):
    image_name: str
    project_id: str
    position: str
    roi_index: int
    quadrant: int
    class_label: str
    ssim: float
    edge_difference: float
    mse: float
    histogram_score: float
    gradient_difference: float
    local_difference: float
    final_score: float
    prediction: str
    defect_count: int = 0
    defect_bounding_boxes: List[DefectBoundingBox] = Field(default_factory=list)


class InspectionResponse(BaseModel):
    project_id: str
    position: str
    test_image_path: str
    aligned_annotated_test_image_path: Optional[str] = None
    annotated_image_base64: Optional[str] = None
    bounding_box_coordinate_system: str = "aligned_test_image_pixels_xyxy_exclusive"
    overall_prediction: str
    total_rois: int
    passed: int
    failed: int
    absent: int
    defect_count: int = 0
    defect_bounding_boxes: List[DefectBoundingBox] = Field(default_factory=list)
    results: List[RoiResult]


