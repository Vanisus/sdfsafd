from __future__ import annotations

import logging
import os

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .inspection_service import InspectionService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Part Inspection API", version="1.0.0")
service = InspectionService()


class LoadReferenceResponse(BaseModel):
    object_id: str
    reference_count: int
    reference_id: int
    message: str


class DefectOut(BaseModel):
    bbox_xyxy: list[int]
    area_px: int
    mean_score: float
    max_score: float
    label: str
    severity: str


class AlignmentOut(BaseModel):
    best_reference_idx: int
    rotation_deg: float
    scale: float
    matches: int
    inliers: int
    inlier_ratio: float
    mask_iou: float


class InspectDetailResponse(BaseModel):
    ok: bool
    message: str
    best_reference_idx: int | None = None
    alignment: AlignmentOut | None = None
    image_score: float | None = None
    defects: list[DefectOut] = []
    annotated_image_b64: str | None = None
    heatmap_b64: str | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/load_reference", response_model=LoadReferenceResponse)
async def load_reference(
    object_id: str = Form(...),
    image: UploadFile = File(...),
) -> LoadReferenceResponse:
    """Загружает эталонное изображение для object_id. Можно вызывать несколько раз."""
    try:
        data = await image.read()
        if not data:
            raise HTTPException(400, "Empty image file")
        result = service.load_reference(object_id, data)
        return LoadReferenceResponse(**result)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        logger.exception("load_reference failed")
        raise HTTPException(500, f"load_reference failed: {exc}")


@app.post("/inspect_detail", response_model=InspectDetailResponse)
async def inspect_detail(
    object_id: str = Form(...),
    image: UploadFile = File(...),
) -> InspectDetailResponse:
    """Проверяет текущее изображение на дефекты, сравнивая с эталонами object_id."""
    try:
        data = await image.read()
        if not data:
            raise HTTPException(400, "Empty image file")
        result = service.inspect_detail(object_id, data)
        return InspectDetailResponse(
            ok=result.ok,
            message=result.message,
            best_reference_idx=result.best_reference_idx,
            alignment=AlignmentOut(**result.alignment) if result.alignment else None,
            image_score=result.image_score,
            defects=[
                DefectOut(
                    bbox_xyxy=list(d.bbox_xyxy),
                    area_px=d.area_px,
                    mean_score=d.mean_score,
                    max_score=d.max_score,
                    label=d.label,
                    severity=d.severity,
                )
                for d in result.defects
            ],
            annotated_image_b64=result.annotated_image_b64,
            heatmap_b64=result.heatmap_b64,
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(404 if "No references" in str(exc) else 400, str(exc))
    except Exception as exc:
        logger.exception("inspect_detail failed")
        raise HTTPException(500, f"inspect_detail failed: {exc}")


@app.delete("/objects/{object_id}")
def delete_object(object_id: str) -> dict:
    if object_id not in service._refs:
        raise HTTPException(404, f"object_id={object_id} not found")
    service._refs.pop(object_id)
    service._detectors.pop(object_id, None)
    return {"object_id": object_id, "message": "deleted"}


def main() -> None:
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("app.api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
