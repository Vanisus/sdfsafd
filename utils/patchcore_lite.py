# patchcore_lite.py
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as tv_models
import torchvision.transforms as T


_DEVICE = torch.device("cpu")
_INPUT_SIZE = 256


@dataclass
class PatchCorePrediction:
    anomaly_map: np.ndarray  # float32, shape (H, W), значения ~[0, 1] после нормализации
    image_score: float


class PatchCoreLite:
    """
    Упрощённый PatchCore:
    - backbone: ResNet18 (layer2 + layer3 feature maps), предобученный на ImageNet
    - memory bank: patch-эмбеддинги нормальных эталонов (без coreset subsampling — ок для 1-10 картинок)
    - anomaly score патча: расстояние до ближайшего соседа в memory bank
    """

    def __init__(self, input_size: int = _INPUT_SIZE) -> None:
        self.input_size = input_size
        self._backbone = self._build_backbone()
        self._transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.memory_bank: torch.Tensor | None = None  # (N, C)
        self.feature_map_size: tuple[int, int] | None = None

    def _build_backbone(self) -> torch.nn.Module:
        weights = tv_models.ResNet18_Weights.IMAGENET1K_V1
        model = tv_models.resnet18(weights=weights)
        model.eval()
        return model.to(_DEVICE)

    @torch.no_grad()
    def _extract_patch_features(self, bgr_image: np.ndarray) -> torch.Tensor:
        """Возвращает patch-эмбеддинги (H*W, C) для одного изображения."""
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        tensor = self._transform(resized).unsqueeze(0).to(_DEVICE)

        features: dict[str, torch.Tensor] = {}

        def hook(name: str):
            def _fn(_module, _input, output):
                features[name] = output
            return _fn

        h2 = self._backbone.layer2.register_forward_hook(hook("layer2"))
        h3 = self._backbone.layer3.register_forward_hook(hook("layer3"))
        self._backbone(tensor)
        h2.remove()
        h3.remove()

        f2 = features["layer2"]  # (1, C2, H2, W2)
        f3 = features["layer3"]  # (1, C3, H3, W3)

        target_size = f2.shape[-2:]
        f3_upsampled = F.interpolate(f3, size=target_size, mode="bilinear", align_corners=False)

        combined = torch.cat([f2, f3_upsampled], dim=1)  # (1, C2+C3, H2, W2)
        self.feature_map_size = (target_size[0], target_size[1])

        _, channels, height, width = combined.shape
        patches = combined.permute(0, 2, 3, 1).reshape(height * width, channels)
        return patches  # (H*W, C)

    def fit(self, reference_images_bgr: list[np.ndarray]) -> None:
        """Строит memory bank из списка нормальных (эталонных) изображений."""
        if not reference_images_bgr:
            raise ValueError("Нужен хотя бы один эталон для построения memory bank")

        all_patches = []
        for image in reference_images_bgr:
            patches = self._extract_patch_features(image)
            all_patches.append(patches)

        self.memory_bank = torch.cat(all_patches, dim=0)  # (N_total_patches, C)

    @torch.no_grad()
    def predict(self, query_image_bgr: np.ndarray) -> "PatchCorePrediction":
        if self.memory_bank is None:
            raise RuntimeError("Memory bank не построен. Сначала вызовите fit().")

        query_patches = self._extract_patch_features(query_image_bgr)  # (H*W, C)

        # Расстояние каждого патча query до ближайшего патча в memory bank.
        # Считаем батчами, чтобы не взорвать память на большом количестве эталонов.
        distances = self._nearest_neighbor_distance(query_patches, self.memory_bank)

        height, width = self.feature_map_size
        anomaly_map = distances.reshape(height, width).cpu().numpy().astype(np.float32)

        image_score = float(np.percentile(anomaly_map, 99))  # устойчивее, чем чистый max

        anomaly_map_resized = cv2.resize(
            anomaly_map,
            (query_image_bgr.shape[1], query_image_bgr.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

        return PatchCorePrediction(anomaly_map=anomaly_map_resized, image_score=image_score)

    @staticmethod
    def _nearest_neighbor_distance(
        query: torch.Tensor,
        bank: torch.Tensor,
        chunk_size: int = 512,
    ) -> torch.Tensor:
        """L2-расстояние от каждого патча query до ближайшего патча в bank, по чанкам."""
        results = []
        for start in range(0, query.shape[0], chunk_size):
            chunk = query[start : start + chunk_size]  # (B, C)
            dist = torch.cdist(chunk, bank)  # (B, N)
            min_dist, _ = dist.min(dim=1)
            results.append(min_dist)
        return torch.cat(results, dim=0)
