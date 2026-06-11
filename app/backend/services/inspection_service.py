from pathlib import Path

from app.backend.config import settings
from app.backend.schemas import InspectionResponse
from app.backend.services.storage_service import StorageService
from app.vision.pipeline import RSMPipeline


class InspectionService:
    def __init__(self, storage: StorageService) -> None:
        self.storage = storage
        self.pipeline = RSMPipeline()

    def inspect(self, project_id: str, position: str, test_image_path: Path) -> InspectionResponse:
        golden_path, annotation_path = self.storage.get_ready_position_files(project_id, position)

        pipeline_output = self.pipeline.run(
            project_id=project_id,
            position=position,
            golden_image_path=golden_path,
            test_image_path=test_image_path,
            annotation_path=annotation_path,
        )
        results = pipeline_output.results
        artifact_relative_path = pipeline_output.aligned_annotated_test_image_path.relative_to(
            settings.output_root
        ).as_posix()

        passed = sum(item.prediction == "PASS" for item in results)
        failed = sum(item.prediction == "FAIL" for item in results)
        absent = sum(item.prediction == "ABSENT" for item in results)

        return InspectionResponse(
            project_id=project_id,
            position=position,
            test_image_path=str(test_image_path),
            aligned_annotated_test_image_path=str(pipeline_output.aligned_annotated_test_image_path),
            aligned_annotated_test_image_url=f"/artifacts/{artifact_relative_path}",
            overall_prediction="PASS" if failed == 0 and absent == 0 else "FAIL",
            total_rois=len(results),
            passed=passed,
            failed=failed,
            absent=absent,
            defect_count=len(pipeline_output.defect_bounding_boxes),
            defect_bounding_boxes=pipeline_output.defect_bounding_boxes,
            results=results,
        )
