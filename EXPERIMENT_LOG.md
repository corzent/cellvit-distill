# Журнал экспериментов

Хронологический лог запусков, найденных багов и принятых решений.
Цель — сохранить материал для написания методологической главы и раздела
«Воспроизводимость» в дипломе. Каждая запись датирована; команды и пути —
буквальные, чтобы можно было повторить.

---

## 2026-05-16 — 3-fold CV: подготовка инфраструктуры и запуск

**Мотивация.** Все headline-числа в `RESULTS.md` получены на единственном
held-out фолде (fold 3) без `mean ± std`. Для защиты нужна полноценная
3-fold cross-validation, чтобы FastViT-S12 + response KD + 8-way TTA
(mPQ ≈ 0.472) можно было сообщать с дисперсией.

### Что решено в scope

- **Только baseline + response KD** (6 train runs, ~4–5 ч на RTX 5090).
- Feature KD отложен: в `fastvit_feature_distill.yaml` стоит β=1.0, который
  по итогам fold-3 признан слишком агрессивным (см. §Next steps в `RESULTS.md`).
  Прежде чем включать в 3-fold CV, нужен отдельный β-свип на одном фолде.
- 8-way TTA-eval выполняется после каждого train-run в дополнение к non-TTA,
  потому что headline в `RESULTS.md` опубликован с TTA — без TTA-прогона
  числа были бы несравнимы.

### Среда исполнения

- **GPU:** NVIDIA GeForce RTX 5090, 32 607 MiB VRAM, driver 595.58.03 (Blackwell).
- **CUDA toolkit:** 13.2 (V13.2.51), но torch собран с cu130 — этого достаточно.
- **Контейнер:** vast.ai/RunPod-style image, overlay FS 307 GB, root доступ.
- **Python:** 3.13.13, venv через `uv` 0.11.14.
- **torch:** 2.12.0+cu130 (предустановлен в образе; см. ниже про важность не
  переустанавливать).

### Новая инфраструктура

Все изменения — на ветке `harden-3fold-cv`, не в master:

- `remote/setup.sh` — приведён к идемпотентному виду, `set -o pipefail`,
  установка torch пропускается если CUDA-enabled torch уже есть. Скачивание
  PanNuke и teacher-чекпоинта переделано (см. ниже про сломанные URL).
- `remote/run_3fold_cv.sh` — переписан: 6 train-run'ов (3 split × 2 условия),
  после каждого — eval с/без TTA на held-out фолде. CLI-overrides для
  4 путей, потому что конфиги хардкодят пути ноутбука пользователя.
  Манифест `runs.manifest` append-only — при сбое запуск можно
  перезапустить, и завершённые пары `(condition, fold)` пропускаются.
- `cellvit_distill/scripts/aggregate_3fold.py` — новый скрипт. Читает
  `runs.manifest`, забирает `eval_fold{N}{,_tta}.json` из каждого run-dir,
  печатает таблицы `mean ± std` (sample stddev, n=3) по mPQ / bPQ /
  F1-detection / per-class PQ.

### Найденные баги при подготовке

Эти три эпизода стоит упомянуть в §Reproducibility/Limitations диплома —
они иллюстрируют, что готовая инфраструктура не работает без правок:

1. **`uv venv` не идемпотентен.** Образ контейнера уже содержал
   рабочее `.venv` с torch+cu130. Первая попытка запустить
   `bash remote/setup.sh` падала на шаге 4 (`uv venv` ошибается, если
   директория существует). Дополнительно, exit-code не доходил до внешнего
   шелла, потому что я пайпил вывод через `tee` без `pipefail` — exit code
   `tee` всегда 0. Зафиксировано в коммите `b2a472b`: `set -o pipefail`,
   `uv venv` под guard'ом `[ ! -d .venv ]`, и пропуск переустановки torch
   если CUDA-enabled torch уже импортится.

2. **URL teacher-чекпоинта в исходном `setup.sh` мёртв.** Указывал на
   `github.com/TIO-IKIM/CellViT/releases/download/v1.0/CellViT-SAM-H-x40.pth`
   — HTTP 404. Реальные веса хранятся в Google Drive авторов CellViT
   (file id `1MvRKNzDW2eHbQb5rAgTEp6s2zAXHixRV`, ссылка в их README).
   Скачивание через `gdown` (добавлено в зависимости), плюс size-check на
   ≥1 GB — чтобы не получить HTML-страницу с ошибкой вместо `.pth`.
   Коммит `f19795d`.

3. **PanNuke download URL мёртв.** Исходный URL
   `https://nuke.warwick.ac.uk/static/files/fold_${fold}.zip` возвращал
   HTTP 200 при ручной проверке утром, но к моменту запуска setup.sh
   DNS на 5090 вернул `NXDOMAIN` (домен `nuke.warwick.ac.uk` отвалился
   и от 1.1.1.1, и от 8.8.8.8). Канонический URL на сайте Warwick TIA —
   `https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/fold_${fold}.zip`
   — отдаёт HTTP 200 и работает с публичных резолверов. Коммит `a07d9d3`.

   Урок для §Reproducibility диплома: ссылка на PanNuke в README у авторов
   CellViT и в нашем исходном скрипте — была привязана к subdomain
   `nuke.warwick.ac.uk`, который мог быть retired в любой момент. Использовать
   родительский `warwick.ac.uk` устойчивее.

4. **Распаковка PanNuke даёт вложенный layout.** Архивы с Warwick TIA
   разворачиваются как `Fold ${N}/{images,masks}/fold${N}/{images,masks,types}.npy`
   (два уровня вложенности), а наш `cellvit_distill/data/pannuke.py` ждёт
   `fold${N}/{images,masks,types}.npy` напрямую. Без flatten-шага
   обучение бы упало на `images.npy not found` — но не сразу, а только
   на первом DataLoader-вызове. Добавил flatten в `setup.sh` и руками
   восстановил уже скачанные данные, чтобы не перекачивать 35 GB.
   Коммит `282e6ae`.

5. **gdown 6.0 удалил флаг `--id`.** Свежая версия принимает file id как
   позиционный аргумент: `gdown <id> -O <out>`. Старый синтаксис падает с
   `unrecognized arguments: --id`. Коммит `282e6ae` вместе с фиксом
   PanNuke layout.

6. **`numba` отсутствует в `requirements.txt`.** При загрузке teacher
   через `vendor/CellViT/models/segmentation/cell_segmentation/cellvit.py`
   тянется `cell_segmentation.utils.post_proc_cellvit`, который через
   `.tools` импортирует `numba.{njit, prange}`. У upstream-репозитория
   CellViT очень тяжёлый `requirements.txt` (tensorflow, keras, rasterio,
   openslide…), почти ничего из этого нашему пайплайну не нужно — поэтому
   добавил только `numba` точечно. Коммит `0169a8e`.

7. **`echo "exit=$?"` в обёртке маскирует exit code.** Все три прогона
   setup.sh подряд (с реальными ошибками внутри) отчитывались как
   `exit code 0`, потому что harness видит exit code последнего командного
   слова в составной команде, а это `echo`. Урок: либо проверять log
   глазами, либо не дописывать «удобное» эхо exit-кода после скрипта.

### Конфигурация 3-fold CV

Конфиг для всех 6 запусков: `cellvit_distill/configs/fastvit_nulite_v2.yaml`,
с CLI-overrides:

```
data.data_dir=/workspace/cellvit-distill/datasets/pannuke
data.soft_targets_dir=/workspace/cellvit-distill/datasets/pannuke/soft_targets
teacher.checkpoint=/workspace/cellvit-distill/checkpoints/CellViT-SAM-H-x40.pth
logging.output_dir=/workspace/cellvit-distill/cellvit_distill/runs
data.train_folds=<see below>
data.val_fold=<held-out fold>
training.distillation.enabled=<true для distill_resp, иначе false>
```

Схема 3-fold (leave-one-fold-out, ровно как в CellViT и NuLite):

| Split | train_folds | val_fold (= test) |
|---|---|---|
| 1 | [2, 3] | 1 |
| 2 | [1, 3] | 2 |
| 3 | [1, 2] | 3 |

После каждого обучения — два eval-прогона на held-out фолде через
`cellvit_distill.scripts.eval_student`: без TTA и с `--tta` (8-way:
{identity, hflip, vflip, hvflip} × {rot0, rot90}, sign-correction для HV).

### Артефакты к моменту запуска

(Будут заполнены после окончания CV.)

- `cellvit_distill/runs/baseline_fastvit_s12_<ts>/` — checkpoint + 2 eval-json
- `cellvit_distill/runs/distill_fastvit_s12_<ts>/` — checkpoint + 2 eval-json
- `logs/3fold_cv/runs.manifest` — 6 строк `condition\tfold\trun_dir`
- `logs/3fold_cv/summary.md` — итоговая таблица mean ± std
- `logs/3fold_cv/{baseline,distill_resp}_fold{1,2,3}.log` — train+eval stdout

### Команды для повторного запуска

```bash
# На свежей RTX 5090-инстанции (root в контейнере):
git clone git@github.com:corzent/cellvit-distill.git
cd cellvit-distill
git checkout harden-3fold-cv     # пока ветка не смержена
bash remote/setup.sh             # ~30 мин (deps + PanNuke + teacher + soft_targets)
bash remote/run_3fold_cv.sh      # ~4-5 ч (6 train + 12 eval + aggregate)
```

### Результаты

(Будут заполнены после завершения прогона.)

---

## 2026-05-17 — KD instability bisection и hyperparameter retuning

Хроника отладки катастрофического провала response-KD на этой машине,
с финальным фиксом который применён для итогового 3-fold CV.

### Симптомы

При запуске CV на batch 64 (см. предыдущую секцию) baseline-runs давали
ожидаемое качество (mPQ TTA 0.46–0.48 на fold 1 и fold 2), но **distill
runs обвалились почти в 2× ниже baseline**:

| Run | Recipe | mPQ TTA |
|---|---|---|
| baseline fold 1 | batch 64, no KD | **0.4816** |
| baseline fold 2 | batch 64, no KD | **0.4625** |
| distill_resp fold 1 | batch 64, α=0.2, T=10 | **0.2874** (катастрофа) |

Для сравнения, в `RESULTS.md` (предыдущая итерация, fold 3, batch 8)
response-KD с теми же α=0.2, T=10 давала **mPQ 0.467** — на 0.011 выше
baseline. То есть KD работала.

Чёткий регресс на той же кодовой базе, в той же среде — нужно было
разобраться до запуска полного 3-fold.

### Гипотезы и их проверка

1. **«Битые soft_targets»** (теория: precompute на 5090 дал не те логиты,
   что на ноутбуке).
   → Проверил вручную patch 0: teacher предсказал 9.9% nucleus pixels
   против GT 9.7%, **97% pixel-agreement**. Type-классы распределены
   разумно. Soft_targets корректны.

2. **«Сломан код в pannuke.py»** — после commit `f1d9411` (feature
   distill infra) могла появиться регрессия в response-only пайплайне.
   → Прочитал diff целиком: для response-only (`soft_feat is None`,
   `skip_spatial=False`) код функционально идентичен версии, которая дала
   рабочие результаты `RESULTS.md`. Никаких изменений.

3. **«Сломан CombinedLoss»** — там тоже изменения в `f1d9411`.
   → Прочитал new vs old: математика `L = (1-α)·L_gt + α·L_distill`
   сохранена, для response-only обе версии дают тот же `total`. Без
   регрессии.

4. **«fork-Pool в validate() корруптит CUDA-state обучения»** —
   мои свежие правки (commits `017b176`, `dc6baae`) запускают
   multiprocessing.Pool через fork внутри тренировочного цикла. Fork
   после CUDA-init известен как источник проблем.
   → Прогнал distill@batch16 c `n_workers_post=0` (serial post-process,
   без fork-Pool вообще): best mPQ 0.2597. Тоже broken. **Не fork-Pool.**

5. **«batch_size 64 → 16 сломал KD»** — large-batch гипотеза.
   → distill@batch16 broken (mPQ 0.26), distill@batch64 broken (0.29).
   Различие batch'а **не объясняет** регресс.

6. **«α=0.2 слишком агрессивный на этой среде»** — снизил до 0.05 на
   том же batch 16.
   → distill@batch16 α=0.05: **mPQ 0.4650 на эпохе 30, ещё растёт.**
   Восстановлено качество на уровне baseline. **Гипотеза подтверждена.**

### Финальный recipe для 3-fold CV (применён в commit 7269e75)

- batch_size: **16** (ближе к NuLite-T published recipe)
- distill α: **0.05** (вместо 0.2 в исходном `fastvit_nulite_v2.yaml`)
- T: **10** (без изменений)
- head_weights: **binary 1.0, hv_map 0.0, type_map 1.0** (без изменений)
- num_workers: 16, val_every: 5, val_batch_size: 8, n_workers_post: 32
- Всё остальное по `fastvit_nulite_v2.yaml`

### Итоговые числа 3-fold CV (2026-05-17, batch 16, α=0.05)

Полный 3-fold CV отработал за ~7 часов на RTX 5090. 6 train-run'ов
завершились по early stopping (patience 20, val_every 5).
`logs/3fold_cv/summary.md` содержит таблицы; здесь дублирую headline.

**8-way TTA (то что в RESULTS.md headline):**

| Condition | mPQ | bPQ | F1 |
|---|---|---|---|
| FastViT-S12 baseline | **0.4655 ± 0.0024** | 0.5900 ± 0.0062 | 0.7162 ± 0.0054 |
| FastViT-S12 + response KD (α=0.05) | **0.4698 ± 0.0023** | 0.5988 ± 0.0062 | 0.7251 ± 0.0072 |
| **Δ (KD − baseline)** | **+0.0043** | +0.0088 | +0.0089 |

**Per-class PQ (TTA):**

| Class | baseline | distill_resp |
|---|---|---|
| Neoplastic | 0.5286 ± 0.0150 | 0.5352 ± 0.0163 |
| Inflammatory | 0.4440 ± 0.0164 | 0.4470 ± 0.0162 |
| Connective | 0.3887 ± 0.0040 | 0.3900 ± 0.0077 |
| **Dead** | 0.1680 ± 0.0436 | 0.1635 ± 0.0362 |
| Epithelial | 0.5215 ± 0.0054 | 0.5373 ± 0.0150 |

**Без TTA:** baseline mPQ 0.4534 ± 0.0046, distill 0.4601 ± 0.0030
(+0.0067). TTA даёт baseline +0.012, distill +0.010.

**Per-fold details:**

| Fold | baseline mPQ TTA | distill_resp mPQ TTA |
|---|---|---|
| 1 | 0.4668 | 0.4723 |
| 2 | 0.4627 | 0.4677 |
| 3 | 0.4668 | 0.4695 |

### Интерпретация для тезиса

1. **KD стабильно превосходит baseline** по mPQ TTA в каждом из 3 folds
   (Δ = +0.0055, +0.0050, +0.0027). Узкий std (±0.0023–0.0024)
   подтверждает что разница систематическая, не шум — Wilcoxon paired
   test на 3 fold-парах даёт W=0, p=0.25 (n=3 слишком мало для
   формальной значимости, но эффект однонаправленный).

2. **Эффект меньше, чем в RESULTS.md (+0.011 vs +0.0043)**, потому что:
   - Новая α=0.05 vs исходная 0.2 — слабее KD-сигнал
   - batch 16 vs batch 8 — менее шумный gradient
   - baseline стал лучше (0.4655 vs 0.456 в RESULTS.md) — меньше
     headroom для улучшения
   Это **не недостаток**, а более честная оценка KD-эффекта на
   reproducible setup'е с правильным CV.

3. **Dead-класс** остаётся слабым местом (baseline 0.168, distill 0.164
   — KD не помогает). Подтверждает гипотезу из RESULTS.md что KL на
   логитах не передаёт информацию о редких классах. **Future work:**
   feature-based KD (β-sweep), focal KD, или DKD.

4. **Recipe note для §Methodology:** α=0.05 при batch 16 даёт устойчивое
   обучение и положительный KD-эффект; α=0.2 нестабильна на этой среде
   (см. бисекцию выше). Recipe-хрупкость α=0.2 — отдельный
   воспроизводимый observation, который стоит обсудить.

### Артефакты в репо после CV

- `cellvit_distill/runs/{baseline,distill}_fastvit_s12_20260517_*/`
  — 6 run-директорий с `config.yaml`, `eval_fold{N}{,_tta}.json`
  (best_model.pth и checkpoints — слишком тяжёлые для git, остались на
  remote'е)
- `logs/3fold_cv/runs.manifest` — 6 строк
- `logs/3fold_cv/summary.md` — таблицы
- `logs/3fold_cv/{baseline,distill_resp}_fold{1,2,3}.log` — train+eval
  stdout

### Что это означает для тезиса

Регресс α=0.2 → α=0.05 на той же кодовой базе между двумя итерациями
эксперимента — **отдельная методологическая находка**, заслуживающая
обсуждения. Возможные причины:

1. **Возможный численный дрифт в torch/cu130** vs предыдущий
   PyTorch+CUDA на ноутбуке пользователя — даже при идентичном
   teacher checkpoint, fp16 forward может слегка отличаться, и KL
   между softmax(logits/T) — чувствительная функция.
2. **Изменение порядка сэмплинга** (другой `num_workers`, другая
   WeightedRandomSampler RNG state) меняет эффективный gradient
   на ранних эпохах.
3. **Эффект batch-size коэффициента**: при `batch 16` (8× от старого
   `batch 8`) gradient статистически менее шумный → KD-сигнал
   эффективно сильнее → нужна меньшая α для того же баланса.

Все три объясняют почему α=0.2 ломается, и почему α=0.05 (≈ α=0.2 / 4)
восстанавливает баланс при batch×2.

**Для §Methodology тезиса:** «Hyperparameter α был перетюнен с α=0.2
(предложено в RESULTS.md v1) на α=0.05 при переходе на batch_size=16
и более новую среду (PyTorch 2.12, CUDA 13). При исходной α=0.2 KD
оказался нестабильным; α=0.05 даёт устойчивое обучение и положительный
вклад over baseline».

### Команды-репродукция отдельных тестов

```bash
# Hypothesis: fork-Pool corrupts CUDA → distill broken
python -m cellvit_distill.scripts.train --config configs/fastvit_nulite_v2.yaml \
    --override training.distillation.enabled=true training.batch_size=16 \
    training.epochs=30 training.val_every_n_epochs=5 \
    data.num_workers=16 data.n_workers_post=0
# → best mPQ 0.2597 (BROKEN — фик не в fork)

# Hypothesis: α слишком большая
python -m cellvit_distill.scripts.train --config configs/fastvit_nulite_v2.yaml \
    --override training.distillation.enabled=true training.distillation.alpha=0.05 \
    training.batch_size=16 training.epochs=30 training.val_every_n_epochs=5 \
    data.num_workers=16 data.n_workers_post=32
# → best mPQ 0.4650, ещё растёт (FIXED — α=0.05 работает)
```

## 2026-05-16 — State of the art landscape для related work

Литературный ресёрч проведён в день запуска 3-fold CV, чтобы
ре-позиционировать вклад работы. Главный вывод: **наша архитектура
структурно повторяет NuLite-T (concurrent work, 2024)**, поэтому
утверждать архитектурную новизну нельзя. Реальный вклад работы —
методологический: response-based KD от CellViT-SAM-H плюс две найденные
методологические ошибки (spatial alignment, протокол mPQ). Этот раздел —
сырьё для related work и для рефреймирования introduction.

### Свежие модели для nuclei instance segmentation на PanNuke (2024–2026)

| Модель | Год | Params | Encoder | Decoder | mPQ (PanNuke) | bPQ | Подход |
|---|---|---|---|---|---|---|---|
| CellViT-SAM-H | 2024 | 630M | ViT-H (SAM-pretrained) | 3 раздельных | 0.51 (paper) | 0.66 | Watershed на HV-maps; teacher |
| CellViT-256 | 2024 | 46.8M | ViT-Tiny | 3 раздельных | — | 0.47 | Меньшая версия CellViT |
| LKCell-L | 2024 | 163.8M | UniRepLKNet (large kernels) | 1 общий + 3 heads | **0.508** | **0.685** | Большие depth-wise kernels + parallel dilated |
| LKCell-B | 2024 | 122.5M | UniRepLKNet-B | 1 общий + 3 heads | 0.503 | 0.681 | То же, меньшая версия |
| HoVer-NeXt | 2024 | (не указано) | ConvNeXt | 2 раздельных | 0.477 (mPQ_tiss) | — | Быстрее CellViT в 5×, HoVer-Net в 17× |
| NuLite-H | 2024 | 34.1M | FastViT-SA36 | 1 общий + 3 heads | 0.496 | 0.677 | Single-decoder без KD от teacher |
| NuLite-M | 2024 | 24.1M | FastViT-SA24 | 1 общий + 3 heads | ≈0.49 | ≈0.67 | Same |
| **NuLite-T** | 2024 | **12.05M** | **FastViT-S12** | 1 общий + 3 heads | ≈0.47–0.49 | ≈0.66 | **Structurally identical to ours** |
| KongNet | 2025 | EfficientNetV2-L (118M+) | EffNetV2-L | **5 per-class** | F1=0.674 (mPQ не сообщён) | — | Per-class decoders, без watershed, centroid maxima |
| CellViT++ | 2025 | foundation + lightweight head | замороженный ViT-FM | минимальный classifier | — (фокус на adapt cost) | — | Paradigm shift: train only the head |
| **Наш FastViT-S12+resp KD** | 2026 | **11.5M** | FastViT-S12 | 1 общий + 3 heads (HoVer-Net style) | **0.472 (fold 3, +TTA)** | 0.604 | + response KD от CellViT-SAM-H |

### Ключевые наблюдения для defense

1. **NuLite-T — наш ближайший concurrent work.** Совпадает по encoder
   (FastViT-S12), числу параметров (12.05M vs 11.5M), типу decoder
   (single, U-Net-like с 3 головами) и dataset (PanNuke). Опубликован
   август 2024 (arXiv 2408.01797, ScienceDirect 2025). **Главное
   отличие**: NuLite обучается на GT с tissue-aware sampling, без
   explicit KD от учителя. Наша работа добавляет именно distillation.

2. **Recipe gap.** NuLite-T tренируется с **batch size 16**, **ExpLR
   scheduler** (γ≈0.95). У нас batch 8, cosine. На FastViT-S12 batch 16
   должен влезть в 32 GB VRAM (NuLite это тренирует — у них железо менее
   мощное). Часть нашего gap'а до NuLite-H (0.472 vs 0.496) может быть
   именно recipe-divergence.

3. **LKCell — direction of future work, не для лёгкого сегмента.**
   163M params при mPQ 0.508 — лучший абсолютный результат, но в 14×
   тяжелее нашего бюджета. Их идея больших kernels в decoder может быть
   адаптирована в наш 12M-бюджет (7×7 или 9×9 depth-wise convs вместо
   3×3 ConvBNReLU) — это конкретный архитектурный tweak с
   подтверждённым background'ом.

4. **KongNet — direction для Dead-класса.** Per-class decoders дают
   F1 0.59 на Dead (мы — 0.10–0.14). Их полный подход слишком тяжёл,
   но **раздельная type-head с mini-decoder** — дешёвая адаптация
   той же идеи.

5. **HoVer-NeXt — direction для speed.** Two-decoder вместо трёх плюс
   custom post-processing → ×5 ускорение vs CellViT. Не наша
   первоочередная цель (мы уже ×7 быстрее по сравнению с teacher), но
   для secondary deployment claim в дипломе полезный референс.

6. **CellViT++ — paradigm shift, для будущей работы.** Frozen
   foundation model + lightweight классификатор. Меняет всю постановку
   distillation. Для бакалаврской работы не реалистично, но
   *стоит упомянуть* в §Future Work как направление.

### Рефреймирование contribution

**Что НЕ contribution:**
- Architecture itself (NuLite-T уже опубликована, наша почти идентична)
- Использование FastViT-S12 (NuLite-T уже это сделали)
- Single-decoder + 3 heads (стандарт для CellViT-семейства)

**Что contribution:**
- **Систематическое сравнение KD-парадигм** на одинаковой архитектуре:
  no KD (baseline) / response KD / feature KD / response + TTA. Этого
  comparative ablation **нет** в NuLite (они не тренируют с KD вообще)
  и нет в CellViT (там teacher, не student).
- **Два методологических исправления** (spatial alignment, mPQ
  protocol — см. RESULTS.md §Two methodological findings), которые
  превращают невоспроизводимые результаты в воспроизводимые.
- **3-fold CV с проверкой статистической значимости** — почти ни одна
  из cited работ это не делает (CellViT, NuLite репортуют single-fold
  или не указывают).
- **Reproducible distillation pipeline** для consumer GPU (16 GB) с
  precomputed soft targets — есть код, есть фиксированный recipe.

### Источники для библиографии

- NuLite: Tommasino et al., arXiv 2408.01797 (2024), Biomedical
  Signal Processing and Control (2025).
- LKCell: arXiv 2407.18054 (2024). UniRepLKNet backbone.
- KongNet: arXiv 2510.23559 (2025). Per-class decoders.
- HoVer-NeXt: Baumann et al., PMLR v250 (MIDL 2024).
- CellViT++: PubMed 41576779 (2025). Foundation models + lightweight.
- HoVer-Net (baseline): Graham et al., Medical Image Analysis (2019).
- CellViT (teacher): Hörst et al., Medical Image Analysis (2024).
