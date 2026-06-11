import asyncio
import base64
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.backend.config import settings
from app.backend.process_worker import run_inspection
from app.backend.schemas import GoldenUploadResponse, InspectionResponse
from app.backend.services.storage_service import StorageService

storage_service = StorageService()

def encode_image_base64(image_path: str) -> str:
    return base64.b64encode(Path(image_path).read_bytes()).decode("ascii")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.process_pool = ProcessPoolExecutor(
        max_workers=settings.process_pool_workers,
    )
    app.state.inspection_semaphore = asyncio.Semaphore(
        settings.max_concurrent_inspections,
    )

    try:
        yield
    finally:
        app.state.process_pool.shutdown(wait=True, cancel_futures=True)


app = FastAPI(
    title="RSM Backend",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/golden/", response_model=GoldenUploadResponse)
async def upload_golden_image(
    projectid: str = Form(...),
    position: str = Form(...),
    file: UploadFile = File(...),
) -> GoldenUploadResponse:
    path = await storage_service.save_golden_image(projectid, position, file)

    return GoldenUploadResponse(
        message="Golden image saved",
        project_id=projectid,
        position=position,
        path=str(path),
    )


@app.post("/predict/", response_model=InspectionResponse)
async def predict_image(
    request: Request,
    projectid: str = Form(...),
    position: str = Form(...),
    file: UploadFile = File(...),
) -> InspectionResponse:
    # Validate readiness in the API process so expected HTTP errors retain
    # their status codes instead of crossing the process-pool boundary.
    storage_service.get_ready_position_files(projectid, position)

    test_path = await storage_service.save_test_image(projectid, position, file)
    loop = asyncio.get_running_loop()

    async with request.app.state.inspection_semaphore:
        response_data = await loop.run_in_executor(
            request.app.state.process_pool,
            run_inspection,
            projectid,
            position,
            str(test_path),
        )

    annotated_path = response_data["aligned_annotated_test_image_path"]
    response_data["annotated_image_base64"] = await asyncio.to_thread(
        encode_image_base64,
        annotated_path,
    )

    return InspectionResponse.model_validate(response_data)




