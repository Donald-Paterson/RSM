from pathlib import Path

from app.backend.services.inspection_service import InspectionService
from app.backend.services.storage_service import StorageService


def run_inspection(project_id: str, position: str, test_image_path: str) -> dict:
    """Run one inspection inside a process-pool worker."""
    service = InspectionService(StorageService())
    response = service.inspect(
        project_id,
        position,
        Path(test_image_path),
        include_annotated_image_base64=False,
    )
    response_data = response.model_dump()
    response_data.pop("annotated_image_base64", None)
    return response_data


