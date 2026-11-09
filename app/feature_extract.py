from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as tv_models
import torchvision.transforms as T


_DEVICE = torch.device("cpu")
_INPUT_SIZE = 256


class FeatureExtractor:
    """ResNet18 features (layer2+layer3), предобученный на ImageNet, CPU."""

    def __init__(self, input_size: int = _INPUT_SIZE) -> None:
        self.input_size = input_size
        weights = tv_models.ResNet18_Weights.IMAGENET1K_V1
        self._backbone = tv_models.resnet18(weights=weights).eval().to(_DEVICE)
        self._transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def patch_features(self, bgr_image: np.ndarray) -> tuple[torch.Tensor, tuple[int, int]]:
        """Возвращает (N, C) patch-эмбеддинги + размер feature map."""
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        tensor = self._transform(resized).unsqueeze(0).to(_DEVICE)

        feats: dict[str, torch.Tensor] = {}
        hooks = [
            self._backbone.layer2.register_forward_hook(self._hook("layer2", feats)),
            self._backbone.layer3.register_forward_hook(self._hook("layer3", feats)),
        ]
        try:
            self._backbone(tensor)
        finally:
            for h in hooks:
                h.remove()

        f2 = feats["layer2"]
        f3 = feats["layer3"]
        target = f2.shape[-2:]
        f3_up = F.interpolate(f3, size=target, mode="bilinear", align_corners=False)
        combined = torch.cat([f2, f3_up], dim=1)

        _, C, H, W = combined.shape
        patches = combined.permute(0, 2, 3, 1).reshape(H * W, C)
        return patches, (H, W)

    @staticmethod
    def _hook(name: str, feats: dict[str, torch.Tensor]):
        def fn(_m, _i, out):
            feats[name] = out
        return fn


class PatchCoreMemoryBank:
    """Хранит патч-эмбеддинги всех эталонов, находит ближайшего соседа для query."""

    def __init__(self) -> None:
        self.bank: torch.Tensor | None = None

    def add(self, patches: torch.Tensor) -> None:
        if self.bank is None:
            self.bank = patches
        else:
            self.bank = torch.cat([self.bank, patches], dim=0)

    @torch.no_grad()
    def nearest_neighbor_distances(self, query: torch.Tensor, chunk: int = 512) -> torch.Tensor:
        if self.bank is None:
            raise RuntimeError("Memory bank is empty")
        return self.kth_neighbor_distances(query, k=1, chunk=chunk)

    @torch.no_grad()
    def kth_neighbor_distances(self, query: torch.Tensor, k: int = 2, chunk: int = 512) -> torch.Tensor:
        """Расстояние до k-го ближайшего соседа (k=2 нужен для baseline,
        т.к. 1-й сосед патча эталона — он сам, на расстоянии 0)."""
        if self.bank is None:
            raise RuntimeError("Memory bank is empty")
        out = []
        for s in range(0, query.shape[0], chunk):
            q = query[s : s + chunk]
            dist = torch.cdist(q, self.bank)
            kth = dist.topk(min(k, dist.shape[1]), largest=False).values[:, -1]
            out.append(kth)
        return torch.cat(out, dim=0)
