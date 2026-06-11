from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.backend.config import settings
from app.backend.schemas import GoldenUploadResponse, InspectionResponse
from app.backend.services.inspection_service import InspectionService
from app.backend.services.storage_service import StorageService

app = FastAPI(title="RSM Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

storage_service = StorageService()
inspection_service = InspectionService(storage_service)
app.mount("/artifacts", StaticFiles(directory=settings.output_root), name="artifacts")


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/projects/{project_id}/goldens/{position}/image", response_model=GoldenUploadResponse)
async def upload_golden_image(project_id: str, position: str, image: UploadFile = File(...)) -> GoldenUploadResponse:
    path = await storage_service.save_golden_image(project_id, position, image)
    return GoldenUploadResponse(
        message="Golden image saved",
        project_id=project_id,
        position=position,
        path=str(path),
    )


@app.post("/api/v1/projects/{project_id}/inspections/{position}", response_model=InspectionResponse)
async def inspect_image(project_id: str, position: str, image: UploadFile = File(...)) -> InspectionResponse:
    test_path = await storage_service.save_test_image(project_id, position, image)
    return inspection_service.inspect(project_id, position, test_path)
