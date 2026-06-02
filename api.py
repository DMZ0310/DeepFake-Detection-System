"""
DeepFake Detection System - FastAPI REST API
=============================================
Run with:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET  /              → Health check
    GET  /health        → Detailed health status
    POST /predict       → Analyze uploaded video
    GET  /docs          → Swagger UI
    GET  /redoc         → ReDoc UI
"""

import sys
import os
import json
import time
import tempfile
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field

from app.models.predictor import DeepFakePredictor
from app.utils.report_generator import DetectionReport
from app.utils.logger import get_logger
from app.config import CONFIDENCE_THRESHOLD, MODEL_PATH, VERSION, SUPPORTED_FORMATS

logger = get_logger(__name__)

# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="DeepFake Shield API",
    description=(
        "AI-powered deepfake video detection REST API. "
        "Upload a video to receive a real/fake verdict with confidence score."
    ),
    version=VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton predictor (loaded on first request)
_predictor: Optional[DeepFakePredictor] = None


def get_predictor() -> DeepFakePredictor:
    global _predictor
    if _predictor is None:
        _predictor = DeepFakePredictor(lightweight=True)
    return _predictor


# ── Response Models ────────────────────────────────────────────────────────────

class VideoMetadata(BaseModel):
    total_frames:   int
    fps:            float
    width:          int
    height:         int
    duration_sec:   float

class DetectionResponse(BaseModel):
    report_id:          str
    verdict:            str = Field(..., description="REAL or FAKE")
    confidence:         float = Field(..., ge=0, le=1)
    fake_score:         float = Field(..., ge=0, le=1)
    is_fake:            bool
    threshold:          float
    frames_analyzed:    int
    faces_detected:     int
    processing_time_sec: float
    model_used:         str
    metadata:           VideoMetadata

class HealthResponse(BaseModel):
    status:      str
    version:     str
    model_loaded: bool
    uptime_sec:  float

_start_time = time.time()


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return {"message": "DeepFake Shield API", "version": VERSION, "docs": "/docs"}


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """System health check."""
    return {
        "status":       "healthy",
        "version":      VERSION,
        "model_loaded": Path(MODEL_PATH).exists(),
        "uptime_sec":   round(time.time() - _start_time, 1),
    }


@app.post(
    "/predict",
    response_model=DetectionResponse,
    tags=["Detection"],
    summary="Analyze video for DeepFake manipulation",
    responses={
        200: {"description": "Detection result"},
        400: {"description": "Invalid file format or size"},
        500: {"description": "Processing error"},
    }
)
async def predict(
    file:      UploadFile = File(..., description="Video file to analyze"),
    threshold: float      = Query(CONFIDENCE_THRESHOLD, ge=0.1, le=0.9,
                                   description="Detection threshold"),
    report:    bool       = Query(False, description="Save PDF report to disk"),
    background_tasks: BackgroundTasks = None,
):
    """
    Upload a video file and receive a deepfake detection verdict.
    
    - **file**: Video file (MP4, AVI, MOV, MKV, WEBM)
    - **threshold**: Fake probability threshold (0.5 default)
    - **report**: If true, saves a PDF report server-side
    
    Returns classification verdict, confidence score, and metadata.
    """
    # Validate format
    suffix = Path(file.filename or "").suffix.lstrip(".").lower()
    if suffix not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{suffix}'. Supported: {SUPPORTED_FORMATS}"
        )

    # Save to temp file
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
            content = await file.read()
            if len(content) > 500 * 1024 * 1024:  # 500 MB limit
                raise HTTPException(status_code=400, detail="File too large (max 500 MB)")
            tmp.write(content)
            tmp_path = tmp.name
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload error: {e}")

    # Run detection
    try:
        predictor = get_predictor()
        predictor.threshold = threshold
        result = predictor.predict(tmp_path)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {e}")
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    # Optional PDF report (async)
    if report and background_tasks:
        def _save_report():
            try:
                rpt = DetectionReport(result, video_filename=file.filename)
                rpt.save_pdf()
            except Exception as e:
                logger.warning(f"Background report save failed: {e}")
        background_tasks.add_task(_save_report)

    # Build response
    meta = result.get("metadata", {})
    report_obj = DetectionReport(result, video_filename=file.filename or "video")

    return DetectionResponse(
        report_id=report_obj.report_id,
        verdict=result["verdict"],
        confidence=result["confidence"],
        fake_score=result["fake_score"],
        is_fake=result["is_fake"],
        threshold=result["threshold"],
        frames_analyzed=result["total_frames"],
        faces_detected=result["frames_with_faces"],
        processing_time_sec=result["processing_time_sec"],
        model_used=result["model_used"],
        metadata=VideoMetadata(
            total_frames=meta.get("total_frames", 0),
            fps=meta.get("fps", 0.0),
            width=meta.get("width", 0),
            height=meta.get("height", 0),
            duration_sec=meta.get("duration_sec", 0.0),
        )
    )


@app.post(
    "/predict/batch",
    tags=["Detection"],
    summary="Analyze multiple videos (returns list of results)",
)
async def predict_batch(
    files:     list[UploadFile] = File(...),
    threshold: float = Query(CONFIDENCE_THRESHOLD, ge=0.1, le=0.9),
):
    """Analyze multiple videos in a single request (sequential processing)."""
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files per batch")

    results = []
    predictor = get_predictor()
    predictor.threshold = threshold

    for f in files:
        suffix = Path(f.filename or "").suffix.lstrip(".").lower()
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
                tmp.write(await f.read())
                tmp_path = tmp.name
            result = predictor.predict(tmp_path)
            results.append({
                "filename": f.filename,
                "verdict":  result["verdict"],
                "confidence": result["confidence"],
                "fake_score": result["fake_score"],
                "error": None,
            })
        except Exception as e:
            results.append({"filename": f.filename, "error": str(e)})
        finally:
            try: os.unlink(tmp_path)
            except: pass

    return {"results": results, "total": len(results)}


# ── Run ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    from app.config import API_HOST, API_PORT
    uvicorn.run("api:app", host=API_HOST, port=API_PORT, reload=True)
