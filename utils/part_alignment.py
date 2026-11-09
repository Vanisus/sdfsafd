# part_alignment.py
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class AlignmentResult:
    image_bytes: bytes
    mask_bytes: bytes
    rotation_deg: float
    scale: float
    matches: int
    inliers: int
    inlier_ratio: float


def _decode(image_bytes: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid or unsupported image bytes")
    return image


def _encode(image: np.ndarray, extension: str = ".png") -> bytes:
    ok, buffer = cv2.imencode(extension, image)
    if not ok:
        raise RuntimeError(f"Failed to encode image as {extension}")
    return buffer.tobytes()


def _part_mask(image: np.ndarray, box_ratio: float = 0.48, min_area: int = 5_000) -> np.ndarray:
    height, width = image.shape[:2]
    size = int(min(height, width) * box_ratio)
    x = max(0, width // 2 - size // 2)
    y = max(0, height // 2 - size // 2)
    size = min(size, width - x, height - y)

    gc_mask = np.zeros((height, width), dtype=np.uint8)
    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)
    cv2.grabCut(
        image,
        gc_mask,
        (x, y, size, size),
        background_model,
        foreground_model,
        7,
        cv2.GC_INIT_WITH_RECT,
    )

    mask = np.where(
        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if cv2.contourArea(contour) >= min_area]
    if not contours:
        raise RuntimeError("Part contour was not found. Adjust box_ratio or min_area.")

    result = np.zeros_like(mask)
    cv2.drawContours(result, [max(contours, key=cv2.contourArea)], -1, 255, cv2.FILLED)
    return result


def _crop(image: np.ndarray, mask: np.ndarray, padding: int) -> tuple[np.ndarray, np.ndarray]:
    points = cv2.findNonZero(mask)
    if points is None:
        raise RuntimeError("Part mask is empty")

    x, y, width, height = cv2.boundingRect(points)
    image_height, image_width = image.shape[:2]
    x0, y0 = max(0, x - padding), max(0, y - padding)
    x1 = min(image_width, x + width + padding)
    y1 = min(image_height, y + height + padding)
    return image[y0:y1, x0:x1].copy(), mask[y0:y1, x0:x1].copy()


def _place_on_background(
    image: np.ndarray,
    mask: np.ndarray,
    background_bgr: tuple[int, int, int] = (30, 30, 30),
    feather_px: int = 2,
) -> np.ndarray:
    if feather_px > 0:
        kernel_size = feather_px * 2 + 1
        alpha = cv2.GaussianBlur(
            mask,
            (kernel_size, kernel_size),
            0,
        ).astype(np.float32) / 255.0
    else:
        alpha = mask.astype(np.float32) / 255.0

    alpha = alpha[..., None]
    background = np.full_like(image, background_bgr)

    return (
        image.astype(np.float32) * alpha
        + background.astype(np.float32) * (1.0 - alpha)
    ).astype(np.uint8)


def align_part_bytes(
    reference_bytes: bytes,
    current_bytes: bytes,
    *,
    output_ext: str = ".png",
    box_ratio: float = 0.48,
    min_area: int = 5_000,
    padding: int = 35,
    max_features: int = 3_000,
    ratio: float = 0.72,
    ransac_threshold: float = 3.0,
) -> AlignmentResult:
    if not 0.4 <= ratio <= 0.95:
        raise ValueError("ratio must be in range [0.4, 0.95]")

    reference_image = _decode(reference_bytes)
    current_image = _decode(current_bytes)
    reference_mask = _part_mask(reference_image, box_ratio, min_area)
    current_mask = _part_mask(current_image, box_ratio, min_area)
    reference_crop, reference_crop_mask = _crop(reference_image, reference_mask, padding)
    current_crop, current_crop_mask = _crop(current_image, current_mask, padding)

    orb = cv2.ORB_create(
        nfeatures=max_features,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=15,
        fastThreshold=7,
    )
    ref_gray = cv2.cvtColor(reference_crop, cv2.COLOR_BGR2GRAY)
    cur_gray = cv2.cvtColor(current_crop, cv2.COLOR_BGR2GRAY)
    kp_ref, desc_ref = orb.detectAndCompute(ref_gray, reference_crop_mask)
    kp_cur, desc_cur = orb.detectAndCompute(cur_gray, current_crop_mask)
    if desc_ref is None or desc_cur is None:
        raise RuntimeError("Not enough distinctive features on the part")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    pairs = matcher.knnMatch(desc_cur, desc_ref, k=2)
    matches = [
        first
        for pair in pairs
        if len(pair) == 2
        for first, second in [pair]
        if first.distance < ratio * second.distance
    ]
    if len(matches) < 6:
        raise RuntimeError(f"Only {len(matches)} reliable feature matches")

    source = np.float32([kp_cur[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    target = np.float32([kp_ref[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    matrix, inlier_mask = cv2.estimateAffinePartial2D(
        source,
        target,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold,
        maxIters=5_000,
        confidence=0.995,
        refineIters=20,
    )
    if matrix is None or inlier_mask is None:
        raise RuntimeError("Could not estimate alignment transform")

    inliers = int(inlier_mask.ravel().sum())
    inlier_ratio = inliers / len(matches)
    if inliers < 6 or inlier_ratio < 0.20:
        raise RuntimeError(
            f"Unreliable alignment: {inliers}/{len(matches)} RANSAC inliers ({inlier_ratio:.1%})"
        )

    ref_height, ref_width = reference_crop.shape[:2]
    aligned = cv2.warpAffine(
        current_crop,
        matrix,
        (ref_width, ref_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    aligned_mask = cv2.warpAffine(
        current_crop_mask,
        matrix,
        (ref_width, ref_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    reference_prepared = _place_on_background(
        reference_crop,
        reference_crop_mask,
        background_bgr=(30, 30, 30),
    )

    aligned_prepared = _place_on_background(
        aligned,
        aligned_mask,
        background_bgr=(30, 30, 30),
    )

    rotation_deg = float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))
    scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
    return AlignmentResult(
        image_bytes=_encode(aligned_prepared, output_ext),
        mask_bytes=_encode(aligned_mask, ".png"),
        rotation_deg=rotation_deg,
        scale=scale,
        matches=len(matches),
        inliers=inliers,
        inlier_ratio=inlier_ratio,
    )


def align_image_bytes(reference_bytes: bytes, current_bytes: bytes, **kwargs: object) -> bytes:
    return align_part_bytes(reference_bytes, current_bytes, **kwargs).image_bytes


def prepare_reference_bytes(
    reference_bytes: bytes,
    *,
    box_ratio: float = 0.48,
    min_area: int = 5_000,
    padding: int = 35,
    background_bgr: tuple[int, int, int] = (30, 30, 30),
) -> tuple[bytes, bytes]:
    """Готовит эталон (crop на едином фоне) без alignment — для построения memory bank."""
    image = _decode(reference_bytes)
    mask = _part_mask(image, box_ratio, min_area)
    crop, crop_mask = _crop(image, mask, padding)
    prepared = _place_on_background(crop, crop_mask, background_bgr)
    return _encode(prepared, ".png"), _encode(crop_mask, ".png")
