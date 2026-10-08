from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
import uvicorn

from app.inference import CrackDetector


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "best.pt"
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"


class PredictionItem(BaseModel):
    class_id: int = Field(..., description="Numeric class identifier")
    class_name: str = Field(..., description="Human-readable class name")
    confidence: float = Field(..., description="Detection confidence")
    bbox: list[float] = Field(..., min_length=4, max_length=4)


class PredictionResponse(BaseModel):
    filename: str
    count: int
    detections: list[PredictionItem]
    image_base64: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.detector = CrackDetector(MODEL_PATH)
    yield


app = FastAPI(
    title="Crack Detection Service",
    version="1.0.0",
    description="FastAPI microservice for crack segmentation and counting using best.pt.",
    lifespan=lifespan,
)

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"model_name": MODEL_PATH.name},
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/predict", response_model=PredictionResponse)
async def predict(request: Request, file: UploadFile = File(...)) -> PredictionResponse:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    detector: CrackDetector = request.app.state.detector

    try:
        result = await run_in_threadpool(detector.predict, content, file.filename or "upload.png")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return PredictionResponse(
        filename=result.filename,
        count=result.count,
        detections=[PredictionItem(**item.to_dict()) for item in result.detections],
        image_base64=result.to_data_url(),
    )


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
