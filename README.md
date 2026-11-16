# Part Inspection API

Сервис инспекции деталей по эталонам: reference-based дефектоскопия без обучения
под конкретный тип детали. Загружаешь N эталонных фото объекта, затем фото
детали для проверки — получаешь bbox дефектов, скоры и итоговое изображение
с разметкой.

## Как это работает

```
load_reference (N раз) ──► сегментация детали, memory bank
                                     │
inspect_detail ──► сегментация ──► выравнивание под эталон (SIFT/ORB + RANSAC + ECC)
                                     │
                    коррекция освещения (Lab mean/std)
                                     │
              ┌──────────────────────┴──────────────────────┐
        PatchCore-lite                          структурное сравнение
     (ResNet18-фичи, kNN         (Lab diff с high-pass фильтром,
      к memory bank,             исключение общих краёв,
      калибровка по p95          область = пересечение масок
      эталона)                   + кольцо по краю)
              └──────────────────────┬──────────────────────┘
                                     │
                    heatmap = max(structure, w * patchcore)
                                     │
                    bbox + severity + annotated image (base64 PNG)
```

Ключевые решения:
- **Выравнивание**: SIFT+FLANN (fallback ORB), RANSAC partial-affine, затем
  ECC-уточнение. Качество оценивается IoU маски детали после warp с маской
  эталона — защищает от ложного совмещения повторяющихся паттернов.
- **Сегментация**: адаптивный Otsu с валидацией площади, fallback GrabCut.
- **Устойчивость к освещению**: подгонка яркости/контраста текущего фото под
  эталон (per-channel mean/std в Lab) + high-pass фильтр diff-карты.
- **Против ложных срабатываний**: общие края (отверстия/пазы в обоих кадрах)
  исключаются — субпиксельный сдвиг штатной геометрии не считается дефектом.
- **Скоры калиброваны**: PatchCore-дистанция нормируется на p95-уровень самих
  эталонов, порог 0.7 — интерпретируемое значение.

## API

### POST /load_reference

| поле | тип | описание |
|---|---|---|
| object_id | form | идентификатор объекта |
| image | file | фото эталона (png/jpg) |

Можно вызывать несколько раз — эталоны накапливаются, детектор перестраивается.

### POST /inspect_detail

| поле | тип | описание |
|---|---|---|
| object_id | form | идентификатор объекта |
| image | file | фото детали для проверки |
| debug | form, необязательно | `true` — вернуть дополнительно выровненное фото и эталон (base64) |

Ответ:

```json
{
  "ok": true,
  "message": "Inspection complete",
  "best_reference_idx": 0,
  "alignment": {"rotation_deg": 29.7, "scale": 1.11, "mask_iou": 0.92, ...},
  "image_score": 1.0,
  "defects": [
    {"bbox_xyxy": [347, 107, 386, 157], "area_px": 677,
     "mean_score": 0.9, "max_score": 1.0, "label": "defect", "severity": "high",
     "type": "edge_overflow"}
  ],
  "annotated_image_b64": "<PNG с обведёнными дефектами, base64>",
  "heatmap_b64": "<PNG heatmap, base64>",
  "log": [
    {"stage": "segmentation", "time_ms": 1800, "mask_px": 69315, "mask_ratio": 0.093},
    {"stage": "alignment", "reference_idx": 0, "time_ms": 2300, "ok": true,
     "matches": 80, "inliers": 14, "inlier_ratio": 0.18, "mask_iou": 0.92,
     "rotation_deg": 30.5, "scale": 1.11},
    {"stage": "fit_memory_bank", "time_ms": 240, "reference_idx": 0, "bank_patches": 1024},
    {"stage": "detection", "time_ms": 420, "image_score": 1.0, "defect_count": 5,
     "defects_by_type": {"edge_overflow": 2, "dark_spot": 3}},
    {"stage": "total", "time_ms": 4800}
  ]
}
```

Типы дефектов (`type`): `edge_overflow` — материал за пределами контура детали,
`line_defect` — вытянутый дефект (царапина/трещина), `dark_spot` / `bright_spot` —
тёмное/светлое пятно, `anomaly` — прочее.

`log` — поэтапный протокол инспекции: сегментация, выравнивание по каждому эталону
(совпадения, inliers, IoU масок, поворот, масштаб), сборка банка признаков,
детекция с разбивкой по типам и общее время. `debug=true` добавляет в ответ
`aligned_image_b64` и `reference_image_b64`.

Если выровнять не удалось (другая деталь, сильная порча кадра) — `ok: false`
с пояснением, без 5xx. Ошибки входных данных — 400/404.

Прочее: `GET /health`, `DELETE /objects/{object_id}`.

## Запуск

```bash
pip install -r requirements.txt
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

Docker:

```bash
docker build -t part-inspection .
docker run -p 8000:8000 part-inspection
```

## Тесты

```bash
pytest tests/ -v
```

Интеграционный тест на реальных фото (`tests/fixtures/`): эталон — чистая
перфорированная пластина, проверка — та же пластина с наплывами по краю
(поворот ~30°, другое освещение). Ожидается: дефекты найдены по краю,
ложных срабатываний на отверстиях нет; эталон против самого себя — 0 дефектов.

## Ограничения

- Инференс CPU (ResNet18 + kNN по memory bank) — ~2-5 c на фото.
- Один object_id = один тип детали; для сильно разных ракурсов загружайте
  несколько эталонов (лучший выбирается по mask IoU).
- Тонкие (1-2 px) царапины могут подавляться фильтрами шума выравнивания.
