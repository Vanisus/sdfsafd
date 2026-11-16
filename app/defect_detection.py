from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch

from .feature_extract import get_extractor


@dataclass
class DefectCandidate:
    bbox_xyxy: tuple[int, int, int, int]
    area_px: int
    mean_score: float
    max_score: float
    label: str
    severity: str
    type: str  # edge_overflow | dark_spot | bright_spot | line_defect | anomaly


def match_lighting(
    reference_bgr: np.ndarray,
    current_bgr: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Подгоняет освещение current под reference (per-channel mean/std в Lab, внутри маски)."""
    ref_lab = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    cur_lab = cv2.cvtColor(current_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    inside = mask > 0
    if inside.sum() < 100:
        return current_bgr

    out = cur_lab.copy()
    for ch in range(3):
        r = ref_lab[..., ch][inside]
        c = cur_lab[..., ch][inside]
        r_mean, r_std = float(r.mean()), float(r.std() + 1e-6)
        c_mean, c_std = float(c.mean()), float(c.std() + 1e-6)
        out[..., ch] = (cur_lab[..., ch] - c_mean) * min(r_std / c_std, 2.0) + r_mean
    out = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_LAB2BGR)


def common_edge_mask(
    reference_bgr: np.ndarray,
    aligned_bgr: np.ndarray,
) -> np.ndarray:
    """Зоны, где края есть в ОБОИХ изображениях (общая геометрия: отверстия, пазы).

    Субпиксельный сдвиг таких краёв после warp — не дефект, а артефакт
    выравнивания, поэтому они исключаются из сравнения. Дефект (царапина,
    трещина, наплыв) — край/область только в одном изображении.
    """
    ref_gray = cv2.GaussianBlur(cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY), (0, 0), 2.0)
    cur_gray = cv2.GaussianBlur(cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2GRAY), (0, 0), 2.0)
    ref_edges = cv2.Canny(ref_gray, 60, 150)
    cur_edges = cv2.Canny(cur_gray, 60, 150)
    k = np.ones((9, 9), np.uint8)
    ref_d = cv2.dilate(ref_edges, k)
    cur_d = cv2.dilate(cur_edges, k)
    return (ref_d > 0) & (cur_d > 0)


def structure_diff(
    reference_bgr: np.ndarray,
    aligned_bgr: np.ndarray,
    mask: np.ndarray,
    reference_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Локальные цветовые различия (Lab), устойчивые к освещению и субпиксельному сдвигу.

    - оба изображения размываются (sigma=3) до сравнения: сдвиг контуров
      на 1-2 px перестаёт давать большой diff;
    - high-pass (вычитание размытой версии) убирает плавные градиенты освещения;
    - мёртвая зона ~6 Lab-единиц отсекает шумовой фон;
    - общие края (отверстия/пазы в обоих кадрах) исключаются;
    - область сравнения: пересечение масок + кольцо по краю эталона
      (ловит наплывы/заусенцы за пределами эталонного контура).
    """
    ref_lab = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    cur_lab = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_lab = cv2.GaussianBlur(ref_lab, (0, 0), 3.0)
    cur_lab = cv2.GaussianBlur(cur_lab, (0, 0), 3.0)

    color_mag = np.sqrt(np.sum((ref_lab - cur_lab) ** 2, axis=-1))
    color_mag = cv2.medianBlur(color_mag.astype(np.uint8), 5).astype(np.float32)

    low = cv2.GaussianBlur(color_mag, (0, 0), 15)
    high = np.clip(color_mag - low - 6.0, 0, None)

    # морфологическое открытие: убирает тонкие (<5px) артефакты
    high = cv2.morphologyEx(high, cv2.MORPH_OPEN, np.ones((5, 5), np.float32))

    # область сравнения
    eroded_cur = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=2)
    if reference_mask is not None:
        eroded_ref = cv2.erode(reference_mask, np.ones((5, 5), np.uint8), iterations=1)
        core = (eroded_cur > 0) & (eroded_ref > 0)
        # кольцо строго ВНЕ контура эталона (~20 px) — ловит наплывы/заусенцы;
        # внутренняя часть у края покрыта core, так туда не попадают
        # артефакты границы маски при warp
        ring = cv2.dilate(reference_mask, np.ones((41, 41), np.uint8)) > 0
        ring = ring & (reference_mask == 0) & (mask > 0)
        allowed = core | ring
    else:
        allowed = eroded_cur > 0
    high[~allowed] = 0.0

    # исключаем общие края (сдвиг штатной геометрии — не дефект)
    common = common_edge_mask(reference_bgr, aligned_bgr)
    high[common] = 0.0
    return np.clip(high / 25.0, 0, 1)


def _part_edge_color(reference_bgr: np.ndarray, reference_mask: np.ndarray) -> np.ndarray:
    """Медианный цвет материала детали в тонкой полосе у эталонного контура.

    Настоящий наплыв/заусенец состоит из материала детали и по цвету близок
    к её краю. Фон (стол, тень) материалу детали не соответствует.
    """
    eroded = cv2.erode(reference_mask, np.ones((21, 21), np.uint8))
    band = (reference_mask > 0) & (eroded == 0)
    if band.sum() < 100:
        band = reference_mask > 0
    lab = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    return np.median(lab[band], axis=0)


def _classify_defect(
    component_mask: np.ndarray,
    reference_mask: np.ndarray,
    ref_lab: np.ndarray,
    cur_lab: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> str:
    """Тип дефекта по расположению, форме и знаку изменения яркости."""
    x0, y0, x1, y1 = bbox
    comp = component_mask[y0:y1, x0:x1] > 0

    # за контуром эталона (кольцо) — наплыв/заусенец
    outside = (reference_mask[y0:y1, x0:x1] == 0) & comp
    if outside.sum() > 0.5 * comp.sum():
        return "edge_overflow"

    w, h = x1 - x0, y1 - y0
    if w > 0 and h > 0 and max(w, h) / max(1, min(w, h)) >= 3.0:
        return "line_defect"  # царапина / трещина

    # знак изменения яркости в зоне дефекта
    d_l = (cur_lab[y0:y1, x0:x1][..., 0] - ref_lab[y0:y1, x0:x1][..., 0])[comp]
    if d_l.size == 0:
        return "anomaly"
    return "dark_spot" if float(np.mean(d_l)) < 0 else "bright_spot"


@torch.no_grad()
def _kth_neighbor_distances(query: torch.Tensor, bank: torch.Tensor, k: int = 2, chunk: int = 512) -> torch.Tensor:
    """Расстояние до k-го ближайшего соседа в bank (для baseline k=2:
    1-й сосед патча эталона — он сам, на расстоянии 0)."""
    out = []
    for s in range(0, query.shape[0], chunk):
        q = query[s : s + chunk]
        dist = torch.cdist(q, bank)
        kth = dist.topk(min(k, dist.shape[1]), largest=False).values[:, -1]
        out.append(kth)
    return torch.cat(out, dim=0)


class DefectDetector:
    """Гибридный детектор дефектов: PatchCore (semantic) + структурное различие (CV).

    - эталоны передаются как кэшированные фичи (пересчитывается только новый);
    - PatchCore скор калибруется по baseline-распределению самих эталонов;
    - каждый дефект классифицируется: edge_overflow (заусенец/наплыв за
      контуром эталона), dark_spot / bright_spot (пятно), line_defect
      (вытянутый: царапина, трещина).
    """

    def __init__(
        self,
        pixel_threshold: float = 0.70,
        min_area_px: int = 100,
        patchcore_weight: float = 0.70,
    ) -> None:
        self.pixel_threshold = pixel_threshold
        self.min_area_px = min_area_px
        self.patchcore_weight = patchcore_weight
        self.extractor = get_extractor()
        self.bank: torch.Tensor | None = None
        self.baseline_p95: float | None = None

    def fit(self, patches_per_reference: list[torch.Tensor]) -> None:
        if not patches_per_reference:
            raise ValueError("Need at least one reference")
        self.bank = torch.cat(patches_per_reference, dim=0)
        dists = []
        for feats in patches_per_reference:
            d = _kth_neighbor_distances(feats, self.bank, k=2)
            dists.append(d.cpu().numpy())
        self.baseline_p95 = float(np.percentile(np.concatenate(dists), 95)) + 1e-6

    @torch.no_grad()
    def _query_distances(self, patches: torch.Tensor, chunk: int = 512) -> torch.Tensor:
        if self.bank is None:
            raise RuntimeError("Memory bank is empty")
        out = []
        for s in range(0, patches.shape[0], chunk):
            q = patches[s : s + chunk]
            dist = torch.cdist(q, self.bank)
            out.append(dist.min(dim=1).values)
        return torch.cat(out, dim=0)

    def detect(
        self,
        reference_bgr: np.ndarray,
        aligned_bgr: np.ndarray,
        aligned_mask: np.ndarray,
        reference_mask: np.ndarray | None = None,
    ) -> tuple[list[DefectCandidate], np.ndarray, float]:
        """Возвращает (дефекты, heatmap [0..1], image_score)."""
        if self.bank is None or self.baseline_p95 is None:
            raise RuntimeError("Detector not fitted")

        corrected = match_lighting(reference_bgr, aligned_bgr, aligned_mask)

        query_feats, (H, W) = self.extractor.patch_features(corrected)
        dist = self._query_distances(query_feats)
        patchcore_raw = dist.reshape(H, W).cpu().numpy().astype(np.float32)
        patchcore_map = cv2.resize(
            patchcore_raw,
            (corrected.shape[1], corrected.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        # калибровка: 1.0 = уровень p95 «нормы»
        pc_score = np.clip(patchcore_map / self.baseline_p95, 0, 1.5) / 1.5

        st_score = structure_diff(reference_bgr, corrected, aligned_mask, reference_mask)

        combined = np.maximum(st_score, self.patchcore_weight * pc_score)
        combined[aligned_mask == 0] = 0.0
        # общие края — артефакты выравнивания, гасим и в PatchCore-сигнале
        common = common_edge_mask(reference_bgr, corrected)
        combined[common] = np.minimum(combined[common], st_score[common])

        inside = combined[aligned_mask > 0]
        image_score = float(np.percentile(inside, 99)) if inside.size else 0.0

        boxes = self._extract_boxes(combined, aligned_mask, reference_mask, reference_bgr, corrected)
        return boxes, combined, image_score

    def _extract_boxes(
        self,
        heatmap: np.ndarray,
        mask: np.ndarray,
        reference_mask: np.ndarray | None,
        reference_bgr: np.ndarray,
        corrected_bgr: np.ndarray,
    ) -> list[DefectCandidate]:
        if heatmap.shape[:2] != mask.shape[:2]:
            heatmap = cv2.resize(heatmap, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)

        heatmap = heatmap.copy()
        heatmap[mask == 0] = 0.0

        binary = (heatmap >= self.pixel_threshold).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        ref_lab = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        cur_lab = cv2.cvtColor(corrected_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        edge_color = _part_edge_color(reference_bgr, reference_mask) if reference_mask is not None else None

        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        boxes: list[DefectCandidate] = []
        for label in range(1, count):
            x, y, w, h, area = stats[label]
            if area < self.min_area_px:
                continue
            comp_scores = heatmap[labels == label]
            max_score = float(comp_scores.max())
            mean_score = float(comp_scores.mean())
            severity = "high" if max_score > 0.8 else "medium" if max_score > 0.7 else "low"
            bbox = (int(x), int(y), int(x + w), int(y + h))
            if reference_mask is not None:
                dtype = _classify_defect(labels == label, reference_mask, ref_lab, cur_lab, bbox)
            else:
                dtype = "anomaly"
            if dtype == "edge_overflow" and edge_color is not None:
                # наплыв должен состоять из материала, похожего на деталь;
                # всё, что по цвету не похоже на кромку детали (стол, тень, фон) — не дефект
                comp_med = np.median(cur_lab[labels == label], axis=0)
                if float(np.sqrt(np.sum((comp_med - edge_color) ** 2))) > 40.0:
                    continue
            boxes.append(
                DefectCandidate(
                    bbox_xyxy=bbox,
                    area_px=int(area),
                    mean_score=mean_score,
                    max_score=max_score,
                    label="defect",
                    severity=severity,
                    type=dtype,
                )
            )
        return sorted(boxes, key=lambda b: b.max_score, reverse=True)
