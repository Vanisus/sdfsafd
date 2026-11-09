from __future__ import annotations

import cv2
import numpy as np


def part_mask(
    image_bgr: np.ndarray,
    *,
    box_ratio: float = 0.48,
    min_area_px: int = 3_000,
    padding_px: int = 35,
    adaptive: bool = True,
) -> np.ndarray:
    """Выделяет маску детали в кадре.

    Пытается найти контур детали адаптивно (Otsu-порог на размытом изображении).
    Если адаптивный путь не даёт стабильный контур (мелкие артефакты, шов, разрыв),
    использует GrabCut как fallback, ограничивая область поиска центральным боксом.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # детерминированность: GrabCut использует глобальный RNG OpenCV
    cv2.setRNGSeed(0)

    h, w = gray.shape
    size = int(min(h, w) * box_ratio)
    x0 = max(0, w // 2 - size // 2)
    y0 = max(0, h // 2 - size // 2)
    size = min(size, w - x0, h - y0)

    if adaptive:
        otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        otsu = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        otsu = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

        center_crop = otsu[y0 : y0 + size, x0 : x0 + size]
        mask = np.zeros_like(gray, dtype=np.uint8)
        mask[y0 : y0 + size, x0 : x0 + size] = center_crop

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) >= min_area_px]
        # валидация: деталь должна занимать заметную часть бокса и не
        # упираться в его границы (иначе это провал сегментации)
        box_area = size * size
        if contours:
            best = max(contours, key=cv2.contourArea)
            bx, by, bw, bh = cv2.boundingRect(best)
            touches_border = (
                bx <= x0 + 2 or by <= y0 + 2
                or bx + bw >= x0 + size - 2 or by + bh >= y0 + size - 2
            )
            if 0.15 * box_area <= cv2.contourArea(best) <= 0.9 * box_area and not touches_border:
                filled = np.zeros_like(gray)
                cv2.drawContours(filled, [best], -1, 255, cv2.FILLED)
                return filled

    # GrabCut fallback
    gc_mask = np.zeros((h, w), dtype=np.uint8)
    bg_model = np.zeros((1, 65), dtype=np.float64)
    fg_model = np.zeros((1, 65), dtype=np.float64)
    cv2.grabCut(
        image_bgr,
        gc_mask,
        (x0, y0, size, size),
        bg_model,
        fg_model,
        7,
        cv2.GC_INIT_WITH_RECT,
    )
    mask = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area_px]
    if not contours:
        raise RuntimeError("Part contour not found")
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, [max(contours, key=cv2.contourArea)], -1, 255, cv2.FILLED)
    return filled


def crop_with_mask(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    padding_px: int = 35,
) -> tuple[np.ndarray, np.ndarray]:
    """Вырезает деталь с маской и паддингом вокруг bbox."""
    points = cv2.findNonZero(mask)
    if points is None:
        raise RuntimeError("Empty part mask")
    x, y, w, h = cv2.boundingRect(points)
    H, W = image_bgr.shape[:2]
    x0, y0 = max(0, x - padding_px), max(0, y - padding_px)
    x1, y1 = min(W, x + w + padding_px), min(H, y + h + padding_px)
    return image_bgr[y0:y1, x0:x1].copy(), mask[y0:y1, x0:x1].copy()


def place_on_background(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    background_bgr: tuple[int, int, int] = (30, 30, 30),
    feather_px: int = 2,
) -> np.ndarray:
    """Помещает деталь на однотонный фон, сглаживая края."""
    if feather_px > 0:
        k = feather_px * 2 + 1
        alpha = cv2.GaussianBlur(mask, (k, k), 0).astype(np.float32) / 255.0
    else:
        alpha = mask.astype(np.float32) / 255.0
    alpha = alpha[..., None]
    bg = np.full_like(image_bgr, background_bgr, dtype=np.uint8)
    return (image_bgr.astype(np.float32) * alpha + bg.astype(np.float32) * (1 - alpha)).astype(np.uint8)
