from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile

from app.backend.config import settings


class StorageService:
    def __init__(self) -> None:
        settings.storage_root.mkdir(parents=True, exist_ok=True)
        settings.debug_root.mkdir(parents=True, exist_ok=True)

    def position_root(self, project_id: str, position: str) -> Path:
        path = settings.storage_root / self._safe_segment(project_id, "project_id") / self._safe_segment(position, "position")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def data_root(self, project_id: str, position: str) -> Path:
        path = self.position_root(project_id, position) / "data"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def golden_image_path(self, project_id: str, position: str) -> Path:
        path = self.data_root(project_id, position) / "images"
        path.mkdir(parents=True, exist_ok=True)
        return path / "golden.png"

    def annotation_path(self, project_id: str, position: str) -> Path:
        path = self.data_root(project_id, position) / "labels"
        path.mkdir(parents=True, exist_ok=True)
        return path / "annotations.xml"

    def test_image_path(self, project_id: str, position: str) -> Path:
        path = self.position_root(project_id, position) / "tests"
        path.mkdir(parents=True, exist_ok=True)
        return path / f"{uuid4().hex}.png"

    async def save_golden_image(self, project_id: str, position: str, file: UploadFile) -> Path:
        return await self._save_image(file, self.golden_image_path(project_id, position))

    async def save_test_image(self, project_id: str, position: str, file: UploadFile) -> Path:
        return await self._save_image(file, self.test_image_path(project_id, position))

    def get_ready_position_files(self, project_id: str, position: str) -> tuple[Path, Path]:
        golden_path = self.golden_image_path(project_id, position)
        annotation_path = self.annotation_path(project_id, position)

        if not golden_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Golden image not found for this project and position. Upload it before inspection.",
            )

        if not annotation_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Annotation XML not found for this project and position. Complete CVAT annotation and place annotations.xml in data/labels before inspection.",
            )

        return golden_path, annotation_path

    async def _save_image(self, file: UploadFile, path: Path) -> Path:
        data = await file.read()
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        cv2.imwrite(str(path), image)
        return path

    def _safe_segment(self, value: str, field_name: str) -> str:
        normalized = value.strip().replace("\\", "_").replace("/", "_")
        if not normalized or normalized in {".", ".."}:
            raise HTTPException(status_code=400, detail=f"Invalid {field_name}")
        return normalized
