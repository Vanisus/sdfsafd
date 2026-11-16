from __future__ import annotations

import base64
import logging
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .alignment import AlignmentResult, align_part
from .defect_detection import DefectCandidate, DefectDetector
from .segmentation import part_mask

logger = logging.getLogger(__name__)


@dataclass
class ReferenceEntry:
    image_bgr: np.ndarray
    mask: np.ndarray


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
    aligned_image_b64: str | None = None
    reference_image_b64: str | None = None
    log: list[dict] = field(default_factory=list)


class InspectionService:
    """Хранит N эталонов по object_id, находит лучший alignment, запускает детектор.

    Лог пайплайна собирается на каждый вызов и возвращается в ответе:
    тайминги стадий, попытки выравнивания по каждому эталону, параметры
    сегментации и сводка детекции.
    """

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

    # ---------- load_reference ----------
    def load_reference(
        self,
        object_id: str,
        image_bytes: bytes,
    ) -> dict:
        t0 = time.perf_counter()
        if not image_bytes:
            raise ValueError("Empty image bytes")
        bgr = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Invalid or unsupported image bytes")

        mask = part_mask(bgr)
        mask_px = int((mask > 0).sum())
        if mask_px < 0.005 * mask.size:
            raise ValueError(
                f"Segmentation failed: part mask is only {mask_px} px. "
                "Проверьте, что деталь занимает заметную часть кадра и контрастна с фоном."
            )

        entry = ReferenceEntry(image_bgr=bgr, mask=mask)

        refs = self._refs.setdefault(object_id, [])
        refs.append(entry)

        # Memory bank строится на canvas эталона в момент инспекции
        # (frame-consistent с выровненным текущим фото), поэтому при загрузке
        # фичи не извлекаются — загрузка эталона быстрая.

        logger.info(
            "load_reference object_id=%s ref#%d mask_px=%d time=%.2fs",
            object_id, len(refs) - 1, mask_px, time.perf_counter() - t0,
        )
        return {
            "object_id": object_id,
            "reference_count": len(refs),
            "reference_id": len(refs) - 1,
            "mask_px": mask_px,
            "message": "Reference loaded",
        }

    # ---------- inspect_detail ----------
    def inspect_detail(
        self,
        object_id: str,
        image_bytes: bytes,
        *,
        debug: bool = False,
    ) -> InspectionResult:
        t_start = time.perf_counter()
        log: list[dict] = []

        if object_id not in self._refs or not self._refs[object_id]:
            raise ValueError(f"No references loaded for object_id={object_id}")

        bgr = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Invalid or unsupported image bytes")

        # --- сегментация ---
        t0 = time.perf_counter()
        try:
            cur_mask = part_mask(bgr)
        except RuntimeError as exc:
            return InspectionResult(
                ok=False,
                message=f"Segmentation failed on current image: {exc}",
                log=log + [{"stage": "segmentation", "error": str(exc)}],
            )
        mask_px = int((cur_mask > 0).sum())
        log.append({
            "stage": "segmentation",
            "time_ms": round((time.perf_counter() - t0) * 1000),
            "mask_px": mask_px,
            "mask_ratio": round(mask_px / cur_mask.size, 4),
        })
        logger.info("inspect object_id=%s mask_px=%d", object_id, mask_px)

        # --- выравнивание по каждому эталону ---
        refs = self._refs[object_id]
        candidates: list[tuple[int, AlignmentResult]] = []
        for idx, ref in enumerate(refs[: self.alignment_max_candidates]):
            t0 = time.perf_counter()
            try:
                alignment = align_part(
                    ref.image_bgr,
                    bgr,
                    reference_mask=ref.mask,
                    min_mask_iou=self.alignment_min_mask_iou,
                )
                candidates.append((idx, alignment))
                log.append({
                    "stage": "alignment",
                    "reference_idx": idx,
                    "time_ms": round((time.perf_counter() - t0) * 1000),
                    "ok": True,
                    "matches": alignment.matches,
                    "inliers": alignment.inliers,
                    "inlier_ratio": round(alignment.inlier_ratio, 4),
                    "mask_iou": round(alignment.mask_iou, 4),
                    "rotation_deg": round(alignment.rotation_deg, 2),
                    "scale": round(alignment.scale, 3),
                })
            except RuntimeError as exc:
                log.append({
                    "stage": "alignment",
                    "reference_idx": idx,
                    "time_ms": round((time.perf_counter() - t0) * 1000),
                    "ok": False,
                    "error": str(exc),
                })
                continue

        if not candidates:
            return InspectionResult(
                ok=False,
                message="Could not align current image to any reference",
                log=log + [{"stage": "detection", "error": "skipped: no alignment"}],
            )

        # --- детекция ---
        best_idx, alignment = max(candidates, key=lambda p: p[1].mask_iou)

        t0 = time.perf_counter()

        # detector и memory bank — в системе координат canvas эталона
        detector = DefectDetector(pixel_threshold=self.pixel_threshold, min_area_px=self.min_area_px)
        ref_patches, _ = detector.extractor.patch_features(alignment.reference_image)
        detector.fit([ref_patches])
        fit_ms = round((time.perf_counter() - t0) * 1000)
        log.append({
            "stage": "fit_memory_bank",
            "time_ms": fit_ms,
            "reference_idx": best_idx,
            "bank_patches": int(ref_patches.shape[0]),
        })
        defects, heatmap, image_score = detector.detect(
            reference_bgr=alignment.reference_image,
            aligned_bgr=alignment.aligned_image,
            aligned_mask=alignment.aligned_mask,
            reference_mask=alignment.reference_mask,
        )
        detect_ms = round((time.perf_counter() - t0) * 1000)

        by_type: dict[str, int] = {}
        for d in defects:
            by_type[d.type] = by_type.get(d.type, 0) + 1
        log.append({
            "stage": "detection",
            "time_ms": detect_ms,
            "best_reference_idx": best_idx,
            "image_score": round(image_score, 3),
            "defect_count": len(defects),
            "defects_by_type": by_type,
        })
        log.append({
            "stage": "total",
            "time_ms": round((time.perf_counter() - t_start) * 1000),
        })

        annotated = self._draw(alignment.aligned_image, defects)
        result = InspectionResult(
            ok=True,
            message="Inspection complete",
            best_reference_idx=best_idx,
            alignment={
                "best_reference_idx": best_idx,
                "rotation_deg": alignment.rotation_deg,
                "scale": alignment.scale,
                "matches": alignment.matches,
                "inliers": alignment.inliers,
                "inlier_ratio": round(alignment.inlier_ratio, 4),
                "mask_iou": round(alignment.mask_iou, 4),
            },
            image_score=image_score,
            defects=defects,
            annotated_image_b64=self._to_b64(annotated),
            heatmap_b64=self._to_b64(self._heatmap_overlay(alignment.aligned_image, heatmap)),
            log=log,
        )
        if debug:
            result.aligned_image_b64 = self._to_b64(alignment.aligned_image)
            result.reference_image_b64 = self._to_b64(alignment.reference_image)
        return result

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
        h_img, w_img = canvas.shape[:2]
        for d in defects:
            x0, y0, x1, y1 = d.bbox_xyxy
            color = (0, 0, 255) if d.severity == "high" else (0, 165, 255) if d.severity == "medium" else (0, 200, 0)
            cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 2)
            label = f"{d.type} {d.max_score:.2f}"
            (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            lx = x0 if x0 + tw + 4 <= w_img else max(0, w_img - tw - 4)
            ly = y0 - 6 if y0 - th - bl >= 0 else y1 + th + bl + 4
            cv2.putText(canvas, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        return canvas

    @staticmethod
    def _heatmap_overlay(image_bgr: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
        norm = np.clip(heatmap, 0, 1)
        colored = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        return cv2.addWeighted(image_bgr, 1 - alpha, colored, alpha, 0)
