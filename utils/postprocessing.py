# postprocessing.py
from __future__ import annotations

import cv2
import numpy as np


def normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """Приводит anomaly map к диапазону [0, 1] по min/max этого конкретного кадра."""
    min_value = float(heatmap.min())
    max_value = float(heatmap.max())
    if max_value - min_value < 1e-6:
        return np.zeros_like(heatmap)
    return (heatmap - min_value) / (max_value - min_value)


def defects_from_heatmap(
    heatmap: np.ndarray,
    part_mask: np.ndarray,
    *,
    pixel_threshold: float = 0.6,
    min_area_px: int = 40,
) -> list[dict]:
    heatmap = heatmap.copy()

    if heatmap.shape[:2] != part_mask.shape[:2]:
        heatmap = cv2.resize(heatmap, (part_mask.shape[1], part_mask.shape[0]), interpolation=cv2.INTER_LINEAR)

    heatmap[part_mask == 0] = 0.0

    binary = (heatmap >= pixel_threshold).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    boxes: list[dict] = []
    for label in range(1, count):
        x, y, width, height, area = stats[label]
        if area < min_area_px:
            continue

        component_scores = heatmap[labels == label]
        boxes.append(
            {
                "bbox_xyxy": [int(x), int(y), int(x + width), int(y + height)],
                "area_px": int(area),
                "mean_score": float(component_scores.mean()),
                "max_score": float(component_scores.max()),
            }
        )

    return sorted(boxes, key=lambda item: item["max_score"], reverse=True)


def draw_boxes(image_bgr: np.ndarray, boxes: list[dict]) -> np.ndarray:
    canvas = image_bgr.copy()
    for box in boxes:
        x0, y0, x1, y1 = box["bbox_xyxy"]
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 0, 255), 2)
        label = f"{box['max_score']:.2f}"
        cv2.putText(canvas, label, (x0, max(0, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return canvas


def heatmap_overlay(image_bgr: np.ndarray, heatmap_normalized: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    colored = cv2.applyColorMap((heatmap_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(image_bgr, 1 - alpha, colored, alpha, 0)
