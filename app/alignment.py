from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .segmentation import crop_with_mask, part_mask, place_on_background


@dataclass(frozen=True)
class AlignmentResult:
    aligned_image: np.ndarray
    aligned_mask: np.ndarray
    reference_image: np.ndarray
    reference_mask: np.ndarray
    transform: np.ndarray
    rotation_deg: float
    scale: float
    matches: int
    inliers: int
    inlier_ratio: float
    mask_iou: float


def _match_sift(
    ref_gray: np.ndarray,
    ref_mask: np.ndarray,
    cur_gray: np.ndarray,
    cur_mask: np.ndarray,
    ratio: float,
) -> tuple[list, list, list]:
    """SIFT + FLANN, keypoints фильтруются маской детали (фон даёт ложные совпадения)."""
    sift = cv2.SIFT_create(nfeatures=5000)
    kp_r_all, _ = sift.detectAndCompute(ref_gray, None)
    kp_c_all, _ = sift.detectAndCompute(cur_gray, None)

    def in_mask(kp, mask):
        return [
            k for k in kp
            if 0 < int(k.pt[1]) < mask.shape[0]
            and 0 < int(k.pt[0]) < mask.shape[1]
            and mask[int(k.pt[1]), int(k.pt[0])] > 0
        ]

    kp_ref = in_mask(kp_r_all, ref_mask)
    kp_cur = in_mask(kp_c_all, cur_mask)
    if len(kp_ref) < 6 or len(kp_cur) < 6:
        return [], kp_ref, kp_cur
    desc_ref = sift.compute(ref_gray, kp_ref)[1]
    desc_cur = sift.compute(cur_gray, kp_cur)[1]

    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    pairs = flann.knnMatch(desc_cur, desc_ref, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < ratio * p[1].distance]
    return good, kp_cur, kp_ref


def _match_orb(
    ref_gray: np.ndarray,
    ref_mask: np.ndarray,
    cur_gray: np.ndarray,
    cur_mask: np.ndarray,
    ratio: float,
) -> tuple[list, list, list]:
    orb = cv2.ORB_create(nfeatures=3000, scaleFactor=1.2, nlevels=8, edgeThreshold=15, fastThreshold=7)
    kp_ref, desc_ref = orb.detectAndCompute(ref_gray, ref_mask)
    kp_cur, desc_cur = orb.detectAndCompute(cur_gray, cur_mask)
    if desc_ref is None or desc_cur is None:
        return [], [], []
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    pairs = bf.knnMatch(desc_cur, desc_ref, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < ratio * p[1].distance]
    return good, kp_cur, kp_ref


def _mask_iou(ref_mask: np.ndarray, warped_mask: np.ndarray) -> float:
    inter = np.logical_and(ref_mask > 0, warped_mask > 0).sum()
    union = np.logical_or(ref_mask > 0, warped_mask > 0).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


def _bbox_of(mask: np.ndarray) -> tuple[int, int, int, int]:
    points = cv2.findNonZero(mask)
    if points is None:
        return 0, 0, 1, 1
    return cv2.boundingRect(points)


def _union_canvas(
    ref_shape: tuple[int, ...],
    ref_bbox: tuple[int, int, int, int],
    cur_corners: np.ndarray,
    margin_ratio: float = 0.12,
    min_margin: int = 24,
) -> tuple[int, int, int, int]:
    """Общая рамка (x, y, w, h): объединение bbox эталона и рамки текущего
    изображения после трансформации, с отступом — ничего не обрезается."""
    rx, ry, rw, rh = ref_bbox
    cx = np.array([cur_corners[:, 0].min(), cur_corners[:, 1].min()])
    cx1 = np.array([cur_corners[:, 0].max(), cur_corners[:, 1].max()])

    x0 = min(rx, cx[0])
    y0 = min(ry, cx[1])
    x1 = max(rx + rw, cx1[0])
    y1 = max(ry + rh, cx1[1])

    margin = max(min_margin, int(margin_ratio * max(x1 - x0, y1 - y0)))
    x0 = int(np.floor(x0 - margin))
    y0 = int(np.floor(y0 - margin))
    x1 = int(np.ceil(x1 + margin))
    y1 = int(np.ceil(y1 + margin))
    return int(x0), int(y0), int(x1 - x0), int(y1 - y0)


def align_part(
    reference_bgr: np.ndarray,
    current_bgr: np.ndarray,
    *,
    reference_mask: np.ndarray | None = None,
    box_ratio: float = 0.48,
    min_area_px: int = 3_000,
    padding_px: int = 35,
    ratio: float = 0.80,
    ransac_threshold: float = 2.5,
    min_inliers: int = 8,
    min_mask_iou: float = 0.50,
) -> AlignmentResult:
    """Выравнивает текущее фото детали под эталон (SIFT+FLANN, fallback ORB, ECC refine).

    Результат — общая canvas-рамка, покрывающая и эталон, и вмапленную текущую
    деталь с запасом: текущая деталь не обрезается, даже если она крупнее
    эталона или смещена (frame = union(bbox) + margin).

    Качество оценивается двумя критериями: число RANSAC-инлайнеров и IoU масок.

    reference_mask можно передать, если маска эталона уже вычислена
    (GrabCut недетерминирован — маска должна считаться один раз).
    """
    if not 0.4 <= ratio <= 0.95:
        raise ValueError("ratio must be in range [0.4, 0.95]")

    if reference_mask is not None and reference_mask.shape[:2] != reference_bgr.shape[:2]:
        raise ValueError("reference_mask shape does not match reference_bgr")
    ref_mask = (
        reference_mask
        if reference_mask is not None
        else part_mask(reference_bgr, box_ratio=box_ratio, min_area_px=min_area_px)
    )
    cur_mask = part_mask(current_bgr, box_ratio=box_ratio, min_area_px=min_area_px)

    # tight-кропы — только для поиска фич (маски отсекают фон)
    ref_crop, ref_crop_mask = crop_with_mask(reference_bgr, ref_mask, padding_px)
    cur_crop, cur_crop_mask = crop_with_mask(current_bgr, cur_mask, padding_px)
    ref_offset = _bbox_of(ref_mask)
    cur_offset = _bbox_of(cur_mask)
    # crop_with_mask возвращает кроп с padding, начало = bbox - padding (clamp к 0)
    rx0 = max(0, ref_offset[0] - padding_px)
    ry0 = max(0, ref_offset[1] - padding_px)
    cx0 = max(0, cur_offset[0] - padding_px)
    cy0 = max(0, cur_offset[1] - padding_px)

    ref_gray = cv2.cvtColor(ref_crop, cv2.COLOR_BGR2GRAY)
    cur_gray = cv2.cvtColor(cur_crop, cv2.COLOR_BGR2GRAY)

    best: tuple[np.ndarray, int, int, float, float] | None = None  # (M, matches, inliers, inlier_ratio, iou)

    for matcher in (_match_sift, _match_orb):
        good, kp_cur, kp_ref = matcher(ref_gray, ref_crop_mask, cur_gray, cur_crop_mask, ratio)
        if len(good) < 6 or len(kp_cur) < 6 or len(kp_ref) < 6:
            continue
        src = np.float32([kp_cur[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp_ref[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        matrix, inlier_mask = cv2.estimateAffinePartial2D(
            src,
            dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_threshold,
            maxIters=10_000,
            confidence=0.999,
            refineIters=30,
        )
        if matrix is None or inlier_mask is None:
            continue
        inliers = int(inlier_mask.ravel().sum())
        if inliers < min_inliers:
            continue
        warped_mask = cv2.warpAffine(cur_crop_mask, matrix, (ref_crop.shape[1], ref_crop.shape[0]), flags=cv2.INTER_NEAREST)
        iou = _mask_iou(ref_crop_mask, warped_mask)
        candidate = (matrix, len(good), inliers, inliers / len(good), iou)
        if best is None or (iou, inliers) > (best[4], best[2]):
            best = candidate

    if best is None:
        raise RuntimeError("Alignment failed: not enough reliable matches or poor mask overlap")

    matrix, matches, inliers, inlier_ratio, iou = best
    if iou < min_mask_iou:
        raise RuntimeError(f"Unreliable alignment: mask IoU={iou:.2f} < {min_mask_iou}")

    # ECC-уточнение (малый остаточный сдвиг/поворот); ошибки игнорируем — уже есть RANSAC
    try:
        aligned_tmp = cv2.warpAffine(cur_gray, matrix, (ref_crop.shape[1], ref_crop.shape[0]))
        _, warp_ref = cv2.findTransformECC(
            ref_gray,
            aligned_tmp,
            np.eye(2, 3, dtype=np.float32),
            cv2.MOTION_EUCLIDEAN,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-7),
            gaussFiltSize=5,
            inputMask=None,
        )
        matrix = warp_ref @ np.vstack([matrix, [0, 0, 1]])
    except cv2.error:
        pass

    # --- canvas: полная трансформация текущий_оригинал -> координаты эталона ---
    to_ref_orig = np.float32([
        [1, 0, rx0],
        [0, 1, ry0],
        [0, 0, 1],
    ]) @ np.vstack([matrix, [0, 0, 1]]) @ np.float32([
        [1, 0, -cx0],
        [0, 1, -cy0],
        [0, 0, 1],
    ])

    # рамка по контуру ДЕТАЛИ (а не всего кадра): иначе canvas раздувается
    # до размера исходного фото, и деталь занимает малую его часть
    cur_pts = cv2.findNonZero(cur_mask)
    cur_pts = cur_pts.reshape(-1, 2).astype(np.float64)
    cur_corners = cur_pts @ to_ref_orig[:2, :2].T + to_ref_orig[:2, 2]

    ref_bbox = _bbox_of(ref_mask)
    kx, ky, kw, kh = _union_canvas(reference_bgr.shape, ref_bbox, cur_corners)

    # warp текущего в canvas, эталон — pad в canvas
    to_canvas = np.float32([
        [1, 0, -kx],
        [0, 1, -ky],
        [0, 0, 1],
    ]) @ to_ref_orig

    aligned = cv2.warpAffine(
        current_bgr,
        to_canvas[:2],
        (kw, kh),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    aligned_mask = cv2.warpAffine(
        cur_mask,
        to_canvas[:2],
        (kw, kh),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    # эталон: canvas-координата c соответствует ref-координате c + (kx, ky);
    # копируем пересечение [0..W)x[0..H) эталона с canvas
    ref_h, ref_w = reference_bgr.shape[:2]
    dx0, dy0 = max(0, -kx), max(0, -ky)
    sx0, sy0 = dx0 + kx, dy0 + ky
    copy_w = min(ref_w - sx0, kw - dx0)
    copy_h = min(ref_h - sy0, kh - dy0)

    canvas_ref = np.zeros((kh, kw, 3), dtype=np.uint8)
    canvas_ref_mask = np.zeros((kh, kw), dtype=np.uint8)
    if copy_w > 0 and copy_h > 0:
        canvas_ref[dy0 : dy0 + copy_h, dx0 : dx0 + copy_w] = reference_bgr[sy0 : sy0 + copy_h, sx0 : sx0 + copy_w]
        canvas_ref_mask[dy0 : dy0 + copy_h, dx0 : dx0 + copy_w] = ref_mask[sy0 : sy0 + copy_h, sx0 : sx0 + copy_w]

    reference_prepared = place_on_background(canvas_ref, canvas_ref_mask)
    aligned_prepared = place_on_background(aligned, aligned_mask)

    rotation_deg = float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))
    scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
    return AlignmentResult(
        aligned_image=aligned_prepared,
        aligned_mask=aligned_mask,
        reference_image=reference_prepared,
        reference_mask=canvas_ref_mask,
        transform=to_canvas,
        rotation_deg=rotation_deg,
        scale=scale,
        matches=matches,
        inliers=inliers,
        inlier_ratio=inlier_ratio,
        mask_iou=iou,
    )
