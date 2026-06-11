from dataclasses import dataclass
from pathlib import Path

from app.backend.schemas import DefectBoundingBox, RoiResult


@dataclass
class PipelineOutput:
    results: list[RoiResult]
    defect_bounding_boxes: list[DefectBoundingBox]
    aligned_annotated_test_image_path: Path
