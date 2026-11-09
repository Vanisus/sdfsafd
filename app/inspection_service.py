from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch

from .alignment import AlignmentResult, align_part
from .defect_detection import DefectCandidate, DefectDetector
from .segmentation import crop_with_mask, part_mask, place_on_background

logger = logging.getLogger(__name__)


@dataclass
class ReferenceEntry:
    image_bgr: np.ndarray
    prepared_bgr: np.ndarray
    mask: np.ndarray
    crop_mask: np.ndarray


@dataclass
class InspectionResult:
    ok: bool
    message: str
    best_reference_idx: int | None = None
    alignment: dict | None = None
    image_score: float | None = None
    defects: list[DefectCandidate] = field(default_factory=list)
    annotated_image_b64: str | None = None
    heatmap_b64: str | None = None


class InspectionService:
    """Хранит N эталонов по object_id, находит лучший alignment, запускает детектор."""

    def __init__(
        self,
        pixel_threshold: float = 0.70,
        min_area_px: int = 100,
        alignment_min_mask_iou: float = 0.50,
        alignment_max_candidates: int = 5,
    ) -> None:
        self.pixel_threshold = pixel_threshold
        self.min_area_px = min_area_px
        self.alignment_min_mask_iou = alignment_min_mask_iou
        self.alignment_max_candidates = alignment_max_candidates
        self._refs: dict[str, list[ReferenceEntry]] = {}
        self._detectors: dict[str, DefectDetector] = {}

    # ---------- load_reference ----------
    def load_reference(
        self,
        object_id: str,
        image_bytes: bytes,
    ) -> dict:
        if not image_bytes:
            raise ValueError("Empty image bytes")
        bgr = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Invalid or unsupported image bytes")

        mask = part_mask(bgr)
        crop, crop_mask = crop_with_mask(bgr, mask)
        prepared = place_on_background(crop, crop_mask)

        refs = self._refs.setdefault(object_id, [])
        refs.append(ReferenceEntry(image_bgr=bgr, prepared_bgr=prepared, mask=mask, crop_mask=crop_mask))

        # rebuild detector for this object_id
        detector = DefectDetector(pixel_threshold=self.pixel_threshold, min_area_px=self.min_area_px)
        detector.fit([r.prepared_bgr for r in refs])
        self._detectors[object_id] = detector

        return {
            "object_id": object_id,
            "reference_count": len(refs),
            "reference_id": len(refs) - 1,
            "message": "Reference loaded",
        }

    # ---------- inspect_detail ----------
    def inspect_detail(
        self,
        object_id: str,
        image_bytes: bytes,
    ) -> InspectionResult:
        if object_id not in self._refs or not self._refs[object_id]:
            raise ValueError(f"No references loaded for object_id={object_id}")

        bgr = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Invalid or unsupported image bytes")

        refs = self._refs[object_id]
        candidates: list[tuple[int, AlignmentResult]] = []
        for idx, ref in enumerate(refs[: self.alignment_max_candidates]):
            try:
                alignment = align_part(
                    ref.image_bgr,
                    bgr,
                    reference_mask=ref.mask,
                    min_mask_iou=self.alignment_min_mask_iou,
                )
                candidates.append((idx, alignment))
            except RuntimeError as exc:
                logger.info("Alignment with ref %d failed: %s", idx, exc)
                continue

        if not candidates:
            return InspectionResult(
                ok=False,
                message="Could not align current image to any reference",
                defects=[],
            )

        best_idx, alignment = max(candidates, key=lambda p: p[1].mask_iou)
        ref = refs[best_idx]
        detector = self._detectors[object_id]

        defects, heatmap, image_score = detector.detect(
            reference_bgr=ref.prepared_bgr,
            aligned_bgr=alignment.aligned_image,
            aligned_mask=alignment.aligned_mask,
            reference_mask=alignment.reference_mask,
        )

        annotated = self._draw(alignment.aligned_image, defects)
        annotated_b64 = self._to_b64(annotated)
        heat_b64 = self._to_b64(self._heatmap_overlay(alignment.aligned_image, heatmap))

        alignment_info = {
            "best_reference_idx": best_idx,
            "rotation_deg": alignment.rotation_deg,
            "scale": alignment.scale,
            "matches": alignment.matches,
            "inliers": alignment.inliers,
            "inlier_ratio": round(alignment.inlier_ratio, 4),
            "mask_iou": round(alignment.mask_iou, 4),
        }
        return InspectionResult(
            ok=True,
            message="Inspection complete",
            best_reference_idx=best_idx,
            alignment=alignment_info,
            image_score=image_score,
            defects=defects,
            annotated_image_b64=annotated_b64,
            heatmap_b64=heat_b64,
        )

    # ---------- helpers ----------
    @staticmethod
    def _to_b64(image_bgr: np.ndarray) -> str:
        ok, buf = cv2.imencode(".png", image_bgr)
        if not ok:
            raise RuntimeError("Failed to encode image to PNG")
        return base64.b64encode(buf.tobytes()).decode("ascii")

    @staticmethod
    def _draw(image_bgr: np.ndarray, defects: list[DefectCandidate]) -> np.ndarray:
        canvas = image_bgr.copy()
        for d in defects:
            x0, y0, x1, y1 = d.bbox_xyxy
            color = (0, 0, 255) if d.severity == "high" else (0, 165, 255) if d.severity == "medium" else (0, 200, 0)
            cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 2)
            cv2.putText(
                canvas,
                f"{d.severity} {d.max_score:.2f}",
                (x0, max(0, y0 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
        return canvas

    @staticmethod
    def _heatmap_overlay(image_bgr: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
        norm = np.clip(heatmap, 0, 1)
        colored = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        return cv2.addWeighted(image_bgr, 1 - alpha, colored, alpha, 0)
