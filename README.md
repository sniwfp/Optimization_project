# StochOpt Bench

Воспроизводимый мини-бенчмарк стохастической оптимизации для задач классификации. Проект вырос из [учебного ноутбука](notebookad78edd653.ipynb): реализации SGD с momentum и L2, экспериментов с размером батча, планировщиками learning rate и стратегиями перемешивания данных.

## Что сравнивается

- Собственный `MomentumSGD` и `torch.optim.SGD` с одинаковыми гиперпараметрами.
- `RandomReshuffling` (новая перестановка каждую эпоху), `ShuffleOnce` (одна перестановка на весь запуск) и последовательный порядок.
- Без scheduler, `CosineAnnealingLR`, `ExponentialLR`, `ReduceLROnPlateau`.
- Размеры мини-батча в одном и том же train/validation/test split.

Все конфигурации начинают с одинакового seed. Лучшая эпоха выбирается **по validation loss**; test оценивается один раз после восстановления её весов. В `summary.csv` и `results.json` сохраняются результаты, `environment.json` фиксирует версии и контрольную сумму a9a, а `comparison.png` показывает validation loss, accuracy и learning rate. Для test сохраняются accuracy, F1 и ROC-AUC (для бинарной задачи). Loss усредняется по числу объектов, поэтому значения сравнимы при разных размерах батча.

## Быстрый запуск

Из каталога `Optimization_project`:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/stochopt-bench --dataset synthetic --epochs 5 \
  --samplers reshuffle once --batch-sizes 32 128 \
  --output-dir runs/synthetic
```

`synthetic` генерируется локально и позволяет проверить весь пайплайн без скачивания данных. Для сравнения реализации SGD с PyTorch:

```bash
.venv/bin/stochopt-bench --dataset synthetic --optimizers custom torch \
  --samplers reshuffle --batch-sizes 64 --epochs 5 \
  --output-dir runs/optimizer-check
```

Основной табличный эксперимент на [LIBSVM a9a](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html):

```bash
.venv/bin/stochopt-bench --dataset a9a --download --epochs 20 \
  --samplers reshuffle once sequential \
  --schedulers none cosine --batch-sizes 32 128 \
  --output-dir runs/a9a
```

Флаг `--download` нужен только при первом запуске. Файл хранится в `data/`. Можно также положить файл `a9a` туда вручную и запускать без сетевого доступа.

В репозитории есть [воспроизведённый эксперимент на a9a](reports/a9a-grid/README.md) с конфигурацией, таблицей, графиком и фактическими метриками.

Продолжение экспериментов из ноутбука с изображениями:

```bash
.venv/bin/python -m pip install -e '.[vision]'
.venv/bin/stochopt-bench --dataset fashion_mnist --download --epochs 10 \
  --batch-sizes 32 64 128 256 --samplers reshuffle --output-dir runs/fashion
.venv/bin/stochopt-bench --dataset cifar10 --download --epochs 10 \
  --batch-sizes 128 --samplers reshuffle once \
  --schedulers none cosine --output-dir runs/cifar10
```

Для короткой проверки больших датасетов есть `--train-limit 1024`. CIFAR-10 с ResNet18 обучается существенно дольше без GPU. На CIFAR-10 используется stem 3×3 без первого max-pooling, чтобы сеть подходила к изображениям 32×32.

## Проверки

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Тесты проверяют численное совпадение собственного SGD с PyTorch на нескольких шагах, поведение сэмплеров, воспроизводимость учебного запуска и сохранение результатов.

## Структура и методика

`src/stochopt/` содержит данные, модели, оптимизатор, сэмплеры, цикл обучения, CLI и отчёты. Исходный ноутбук оставлен для истории; запускаемый код не зависит от порядка выполнения его ячеек. `tests/` содержит проверки ключевой логики.

Synthetic и a9a делятся на 70% train / 15% validation / 15% test фиксированной перестановкой; порядок train затем возвращается к порядку исходного датасета, чтобы `sequential` был корректным контролем. FashionMNIST и CIFAR-10 используют официальный test, а 15% официального train отводится под validation. Нормировка строк a9a выполняется независимо для каждого объекта. Основная метрика выбора эпохи — validation loss; test метрики служат итоговой оценкой. При сравнении времени учитывайте, что `training_seconds` зависит от устройства и его текущей нагрузки. При одинаковом числе эпох разные batch size дают разное число шагов оптимизатора.

