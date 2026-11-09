from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from utils.part_alignment import align_part_bytes, prepare_reference_bytes, AlignmentResult
from utils.patchcore_lite import PatchCoreLite
from utils.postprocessing import defects_from_heatmap, normalize_heatmap, draw_boxes, heatmap_overlay


def load_bytes(path: Path) -> bytes:
    return path.read_bytes()


def save_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"Failed to encode {path}")
    path.write_bytes(buffer.tobytes())


def decode(image_bytes: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)


def pick_best_alignment(
    reference_paths: list[Path],
    current_bytes: bytes,
) -> tuple[Path, AlignmentResult]:
    results = []
    for ref_path in reference_paths:
        ref_bytes = load_bytes(ref_path)
        try:
            result = align_part_bytes(ref_bytes, current_bytes)
            results.append((ref_path, result))
            print(f"  {ref_path.name}: inlier_ratio={result.inlier_ratio:.2f}, inliers={result.inliers}/{result.matches}")
        except RuntimeError as exc:
            print(f"  {ref_path.name}: alignment failed ({exc})")
            continue

    if not results:
        raise RuntimeError("Alignment не удался ни с одним эталоном")

    return max(results, key=lambda pair: pair[1].inlier_ratio)


def main() -> None:
    parser = argparse.ArgumentParser(description="One-shot/few-shot inspection prototype")
    parser.add_argument("--part-dir", required=True, help="Папка детали с подпапкой references/")
    parser.add_argument("--current", required=True, help="Путь к текущему фото для проверки")
    parser.add_argument("--out", default="./out", help="Папка для артефактов")
    parser.add_argument("--pixel-threshold", type=float, default=0.55)
    parser.add_argument("--min-area-px", type=int, default=40)
    args = parser.parse_args()

    part_dir = Path(args.part_dir)
    references_dir = part_dir / "references"
    reference_paths = sorted(
        p for p in references_dir.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not reference_paths:
        raise RuntimeError(f"Нет эталонов в {references_dir}")

    current_bytes = load_bytes(Path(args.current))
    out_dir = Path(args.out)

    print(f"Найдено эталонов: {len(reference_paths)}")
    print("Выбор лучшего эталона для alignment...")
    best_ref_path, alignment = pick_best_alignment(reference_paths, current_bytes)
    print(f"Лучший эталон для alignment: {best_ref_path.name} (inlier_ratio={alignment.inlier_ratio:.2f})")

    save_bytes(out_dir / "aligned_current.png", alignment.image_bytes)
    save_bytes(out_dir / "aligned_mask.png", alignment.mask_bytes)

    print("Подготовка эталонов для memory bank (crop + единый фон, без alignment)...")
    reference_images_for_bank = []
    for ref_path in reference_paths:
        prepared_bytes, _ = prepare_reference_bytes(load_bytes(ref_path))
        reference_images_for_bank.append(decode(prepared_bytes))
        save_bytes(out_dir / f"reference_prepared_{ref_path.stem}.png", prepared_bytes)

    print(f"Построение PatchCore memory bank из {len(reference_images_for_bank)} эталон(ов)...")
    model = PatchCoreLite()
    model.fit(reference_images_for_bank)

    aligned_image = decode(alignment.image_bytes)
    aligned_mask = decode(alignment.mask_bytes)
    if aligned_mask.ndim == 3:
        aligned_mask = cv2.cvtColor(aligned_mask, cv2.COLOR_BGR2GRAY)

    print("Инференс PatchCore на выровненном текущем фото...")
    prediction = model.predict(aligned_image)

    heatmap_norm = normalize_heatmap(prediction.anomaly_map)
    save_image(out_dir / "heatmap_overlay.png", heatmap_overlay(aligned_image, heatmap_norm))

    boxes = defects_from_heatmap(
        heatmap_norm,
        aligned_mask,
        pixel_threshold=args.pixel_threshold,
        min_area_px=args.min_area_px,
    )

    boxed_image = draw_boxes(aligned_image, boxes)
    save_image(out_dir / "detected_defects.png", boxed_image)

    result = {
        "part_dir": str(part_dir),
        "current_image": str(args.current),
        "best_reference": best_ref_path.name,
        "alignment": {
            "rotation_deg": alignment.rotation_deg,
            "scale": alignment.scale,
            "inlier_ratio": alignment.inlier_ratio,
            "inliers": alignment.inliers,
            "matches": alignment.matches,
        },
        "image_score": prediction.image_score,
        "defects": boxes,
    }

    (out_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print("\n=== Результат ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nАртефакты сохранены в {out_dir}")


if __name__ == "__main__":
    main()
