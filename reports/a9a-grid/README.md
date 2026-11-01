# Эксперимент на LIBSVM a9a

Исходные данные: 32 561 объект, 123 признака, бинарная классификация. Строки нормированы по L2. Разделение seed 42: 22 792 train / 4 884 validation / 4 885 test. Доля большинства на test — 75,56% (baseline accuracy при предсказании одного класса).

```bash
OMP_NUM_THREADS=1 stochopt-bench --dataset a9a --device cpu --epochs 15 \
  --lr 0.1 --samplers reshuffle once --schedulers none cosine \
  --batch-sizes 32 128 --output-dir runs/a9a-grid
```

Во всех запусках: momentum 0.9, weight decay 1e-4, одинаковая начальная модель и один и тот же split. Лучшая эпоха выбрана по **validation loss**. Только после этого модель оценена на test. Точные версии пакетов, платформа и SHA-256 датасета записаны в [environment.json](environment.json); история всех эпох — в [results.json](results.json).

| Конфигурация | Validation loss | Test accuracy | Test F1 | Test ROC-AUC |
|---|---:|---:|---:|---:|
| Reshuffle, batch 32, постоянный LR | **0,3262** | 83,60% | 0,615 | 0,895 |
| Reshuffle, batch 128, постоянный LR | 0,3289 | 83,64% | 0,614 | 0,893 |
| Reshuffle, batch 32, cosine | 0,3266 | 83,75% | 0,619 | 0,894 |
| Reshuffle, batch 128, cosine | 0,3327 | 83,23% | 0,603 | 0,891 |
| Shuffle once, batch 32, постоянный LR | 0,3268 | 83,46% | 0,601 | 0,895 |
| Shuffle once, batch 128, постоянный LR | 0,3290 | 83,64% | 0,618 | 0,893 |
| Shuffle once, batch 32, cosine | 0,3266 | 83,60% | 0,615 | 0,895 |
| Shuffle once, batch 128, cosine | 0,3331 | 83,30% | 0,603 | 0,891 |

Выбранная по validation loss конфигурация (`reshuffle`, batch 32, постоянный LR) дала **83,60% test accuracy**, на 8,04 процентного пункта выше baseline большинства. Её test F1 — 0,615, ROC-AUC — 0,895. Разница между двумя способами перемешивания мала в этом прогоне; один seed не позволяет утверждать, что один из них стабильно лучше. При одинаковом числе эпох batch 32 делает больше обновлений весов, чем batch 128, поэтому сравнение не изолирует только эффект размера батча.

[График сходимости](comparison.png) · [Полная таблица](summary.csv)
