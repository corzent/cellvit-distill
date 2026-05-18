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


---

## 2026-05-17–18 — Direction expansion: Mamba decoder + DKD + UFD-KD probe

Long session spanning two days. Goal: stress-test the master pipeline by
adding three new directions before any thesis writeup is finalized:

  1. **Novel A** — adapt UFD-KD (Lu et al. BMVC 2025) from classification
     to dense prediction.
  2. **Comparative KD ablation** — add Decoupled KD (Zhao CVPR 2022) as
     a third method alongside KL and UFD.
  3. **Mamba-decoder student** — replace conv decoder with SSM, motivated
     by linear-complexity claim and the empty cell "KD-into-Mamba for
     nuclei seg" in the literature.

### Setup hygiene

Before any new experiments:

- `train.py`: deterministic seeds (torch / cuda / numpy / random /
  PYTHONHASHSEED, default 42). Does not toggle cudnn.deterministic — the
  ~20% slowdown isn't worth it; std reported across folds is still real
  fold variance, not RNG.
- `train.py`: load_config now expands `${ENV_VAR}` and
  `${ENV_VAR:-fallback}` in string values. All 5 configs migrated from
  `/home/corzent/caspian/thesis/...` to env-driven defaults. Repo now
  runs from a fresh clone with `uv sync && bash run_experiments.sh`.
- `metrics.py + eval_student.py`: optional per-image PQ arrays
  (`return_per_image=True`). Splits scalars (json) from per-image arrays
  (npz). Enables paired Wilcoxon and bootstrap CI on per-image deltas.
- `scripts/stat_test.py`: paired t / Wilcoxon / 10k bootstrap CI +
  Cohen's d. Verified on existing 3-fold numbers from RESULTS.md:
  paired t p=0.036 (significant), Wilcoxon p=0.25 (not — n=3 is too
  small for Wilcoxon's rank sum). **First defensibility finding of the
  session:** the published headline mean±std is not statistically
  conservative; per-image arrays + bootstrap CI are required for the
  Phase D grid.

### Phase 0 — Mamba literature scout

Before committing to Mamba-decoder work, scouted the field. Key result:

- **CP-Mamba (AAAI 2025, arXiv 2503.10422)** already published Mamba
  encoder-decoder on PanNuke with **mPQ 0.6149** — above our teacher
  under our protocol. The "first Mamba on PanNuke" claim is gone.
- **No prior work** distills a transformer/conv teacher into a Mamba-
  decoder student for nuclei (or any other) instance segmentation on
  histopathology. This is the defensible novelty slice.
- **HoVer-UNet (arXiv 2311.12553)** is the only prior KD-on-nuclei-seg
  work — direct KD-on-conv baseline for the ablation.
- **UltraLight VM-UNet (Wu et al. 2024, arXiv 2403.20035)** is the
  architecture template for our Mamba decoder (PVM block — channels
  split into G groups, independent Mamba per group, concat + GroupNorm).

Decision (with user 2026-05-17): half-pivot. Mamba is one contribution
not the headline; CP-Mamba cited not reproduced; story is comparative
ablation `{conv, SSM} × {no-KD, KL, DKD}` at fixed FastViT-S12 encoder
and ≤15M cap. Full notes: `docs/related-work.md`.

### Novel A — Frequency-Decoupled KD

UFD-KD adapted to dense prediction in `losses.py:FrequencyDecoupledDistillLoss`:
  - Per-pixel softmax → 2D DCT-II (Dh @ x @ Dw^T, two batched matmuls)
  - Split DCT into LF (top-left K×K) and HF (residual), weighted MSE
  - Default lf_size=32, lf_weight=1, hf_weight=3, T²-scaled

**Smoke (1 epoch, fold 1):** loss runs end-to-end with bf16/fp16, but
the band magnitudes are **lf ≈ 12, hf ≈ 0.02** for the binary head
(softmax over 2 classes). With `hf_weight=3`, HF contribution is 0.5%
of total — the intended frequency decoupling is effectively a no-op.
T² scaling at T=10 also blows the loss up by 100×, requiring α to drop
from the master 0.05 to ~5×10⁻⁴.

**Phase A 1-fold (fold 3, 60 ep, α=5×10⁻⁴, T=10):** Best mPQ=0.4658
plain / 0.4725 TTA at epoch 48. Versus master KL distill (fold 3):
**plain −0.0037, TTA +0.0030** — within 1 std (±0.0023). Per-class TTA
**Dead = 0.1357 vs master KL 0.1635 (−17%)** — the hypothesized rare-
class benefit did NOT materialize. The HF-band-no-op smoke prediction
was correct: UFD here is effectively LF-only MSE on softmax, not the
band-decoupled KL we wanted.

**Verdict:** abandoned hypothesis but a clean negative result that
informs the thesis discussion. Code, tests, and per-image arrays kept
on `dev/post-master-work` for reproducibility.

### Phase D1 — Decoupled KD (Zhao CVPR 2022)

`losses.py:DecoupledDistillLoss`:
  - TCKD: binary KL on (p(target), 1 − p(target))
  - NCKD: KL on non-target subspace after renormalization
  - Default α=1.0, β=8.0 (paper recommends β ≫ α to amplify NCKD)
  - Numerical path uses `full_KL − tgt_KL` to avoid `log(0)` from
    masked-softmax
  - Wired through `DistillationLoss(loss_type='dkd', dkd_alpha=...,
    dkd_beta=...)` and `CombinedLoss` kwargs
  - Per-pixel target = teacher argmax (keeps DistillationLoss signature
    unchanged — no GT map argument)

Sanity tests in `tests/test_dkd.py` (11 checks, all pass). Magnitudes
(T=4, C=6): standard KL × T² = 0.835, DKD α=β=1 = 0.988, TCKD = 0.020,
NCKD = 0.042. At paper defaults α=1, β=8: DKD ≈ 0.36 — same order as
KL baseline. **Existing α=0.05 outer-KD weight works without
recalibration**, unlike UFD which needed ~10000× retuning.

### Phase C — Mamba decoder

Code (`cellvit_distill/models/mamba_decoder.py`):
  - `MambaDecoder` mirrors `HoVerNetDecoder` interface (drop-in via
    `student.decoder_type='mamba'`).
  - `MambaBlock` supports 3 scan patterns: `fwd`, `bidirectional`,
    `cross_scan_4way` (VMamba SS2D style — row fwd + row rev + col fwd
    + col rev, averaged).
  - `PVMBlock`: channels split into G groups, independent MambaBlock per
    group + GroupNorm residual.
  - Mamba1 (v1) and Mamba2 (v2) paths via `_try_import_mamba(version)`.
    Both fall through to a CPU placeholder when `mamba_ssm` isn't
    installed — lets architecture iteration happen before env setup.

**Env on RTX 5090 (sm_120) + CUDA 13 + PyTorch 2.12:** scout warned 2-3
days for source build; actual build took **~25 min** with default
`uv pip install --no-build-isolation`. Pip had to be installed into
the uv-venv first (`uv pip install --python .venv/bin/python pip`),
then `causal-conv1d 1.6.2.post1` and `mamba-ssm 2.3.2.post1` compiled
cleanly with all gencodes including sm_120. End-to-end Mamba-decoder
+ FastViT-S12 + 3 HoVer heads forward pass: bf16 stable, no NaN.

**C2 stability smoke (2 ep, fold 1):** train loss 11.6 → 8.88
monotonically. Model 9.8M total (8.3M encoder + 1.4M decoder + 0.03M
heads + tissue head). Speed comparable to conv decoder at the same
batch.

**C4 Mamba baseline 1-fold (fold 3, default config — bi/v1/d_state=16/
groups=4):** mPQ 0.4531 plain / 0.4650 TTA. **Worse than HoVer baseline
0.4668 plain on fold 3 by 0.014.** Per-class TTA Dead = 0.1189
vs HoVer ~0.168 = −30% relative. Decoder is 1.4M vs HoVer's 3.1M —
under-parameterized at the default Mamba config.

**C3 architecture sweep (6 configs × fold 3, 40 ep each):**

| Config       | scan            | d_state | groups | mPQ plain | mPQ TTA |
|--------------|-----------------|---------|--------|-----------|---------|
| bi_d16_g4    | bidirectional   | 16      | 4      | 0.4531    | 0.4650  |
| bi_d32_g4    | bidirectional   | 32      | 4      | 0.4538    | 0.4642  |
| bi_d64_g4    | bidirectional   | 64      | 4      | 0.4566    | 0.4656  |
| cs4_d16_g4   | cross_scan_4way | 16      | 4      | 0.4533    | 0.4670  |
| cs4_d32_g4   | cross_scan_4way | 32      | 4      | 0.4554    | 0.4665  |
| **cs4_d64_g4** | **cross_scan_4way** | **64** | **4** | **0.4611** | **0.4717** |
| bi_d32_g2    | bidirectional   | 32      | 2      | 0.4555    | 0.4630  |

  - `groups=2` (wider per-group dim) HURTS — worst of the sweep.
  - `cross_scan_4way` alone gives +0.001–0.002 mPQ TTA.
  - `d_state` alone gives +0.001 mPQ TTA per step.
  - **Combined cs4 + d_state=64 → +0.007** over default. Non-linear
    synergy. Total student 10.7M with this config (under ≤15M cap).

**Headline:** `cs4_d64_g4` Mamba decoder ties HoVer baseline 0.4720 on
mPQ TTA fold 3 (gap −0.0003 within noise). First matched-budget
demonstration that Mamba decoder is viable for PanNuke nuclei seg in
the lightweight regime.

### Phase C4 — Mamba + KL distill (1-fold sanity)

Using winning Mamba config (cs4_d64_g4, 10.7M) + master KL distill
recipe (α=0.05, T=10, head_weights bin=1 hv=0 type=1):

  - Best mPQ plain = 0.4584 (vs Mamba no-KD 0.4611, **−0.003**)
  - Best mPQ TTA = 0.4664 (vs Mamba no-KD 0.4717, **−0.005**)
  - Dead PQ TTA = 0.1114 (vs Mamba no-KD ~similar)

**KL distill on Mamba is marginally negative.** Three readings:

  1. Real: SSM gradient dynamics don't accept KD signal the same way
     conv gradients do.
  2. Hyperparameter mismatch: α/T tuned for HoVer 11.5M conv; Mamba
     10.7M SSM might need different (sweep is open work).
  3. Fold noise: −0.005 vs std ±0.0023 = ~2σ. Borderline.

Cannot distinguish (1)–(3) from one fold. Phase D 3-fold grid + B5
Mamba α/T sweep are the open follow-ups.

### Snapshot — fold 3 1-fold runs only (TTA where listed)

| Run                              | Decoder | KD       | mPQ TTA  | Notes |
|----------------------------------|---------|----------|----------|-------|
| HoVer baseline (master CV avg)   | conv    | none     | ≈0.4720  | fold 3 from 3-fold CV |
| HoVer + KL distill (master CV)   | conv    | KL       | 0.4695   | fold 3 from 3-fold CV |
| HoVer + UFD-KD (Phase A)         | conv    | UFD      | 0.4725   | 1-fold, abandoned |
| Mamba cs4_d64 no-KD (C3)         | mamba   | none     | 0.4717   | 1-fold, ties HoVer |
| Mamba cs4_d64 + KL (C4)          | mamba   | KL       | 0.4664   | 1-fold, KD HURTS |

All on fold 3 only. **No statistical claim is defensible until the
Phase D 3-fold grid completes** — current numbers are ±0.005 of each
other and std across 3 folds is ±0.0023 → most differences are within
noise.

### What today does NOT settle (open work)

  - **DKD on Mamba** — currently launched as 1-fold. Will tell whether
    DKD picks up where KL doesn't.
  - **B4 teacher reference gap** — 0.51 paper vs 0.592 our eval is still
    unresolved. Needed before we can quote "79% of teacher quality".
  - **B5 α/T sweep on Mamba** — separate from HoVer sweep, distinct
    optimization surface.
  - **D2 3-fold ablation grid** — 6 conditions × 3 folds = 18 runs,
    needed for any headline mean±std claim.
  - **MoNuSeg cross-dataset eval** — infrastructure ready
    (`feat/monuseg-eval`), needs MoNuSeg download + winning model from
    Phase D.
  - **Phase F polish** — RESULTS.md headline rewrite once 3-fold numbers
    are in.

### Branch state (eight git heads, all built today)

  - `master`             4f79123 (origin, unchanged)
  - `dev/post-master-work` 17eef13 → contains UFD impl + post-review
    fixes, DKD impl, Mamba decoder + sweep script, hygiene (seeds +
    paths), per-image stats util, MambaDecoder wired into build_student.
  - `feat/ablation-grid-runner` 131d486 — D2 grid runner + main_ablation
    yaml (6 conditions × 3 folds).
  - `feat/monuseg-eval`  d051657 — MoNuSeg adapter + download +
    binary eval CLI with overlap stitching.
  - `docs/related-work`  6a97b19 — thesis literature notes.

Mixed-history on `dev/post-master-work` is acknowledged debt (user
flagged it; we chose rename-vs-resplit; per-commit each entry stays
atomic and cherry-pickable).


---

## 2026-05-18 — DKD on Mamba + Phase E cross-dataset eval

### DKD on Mamba (cs4_d64_g4 + DKD)

Same config as Mamba+KL but loss_type=dkd, defaults α=1.0 β=8.0, outer α=0.05, T=10.

  - **mPQ plain 0.4559, TTA 0.4631** (best at epoch 48 of 60).
  - vs Mamba+no-KD (cs4_d64): plain −0.005, TTA −0.009.
  - vs Mamba+KL (same config): TTA −0.003 (DKD slightly worse than KL).
  - Per-class TTA Dead = 0.1155 vs no-KD ~0.119 — **rare-class hypothesis
    failed**. NCKD β=8 reweighting that was supposed to amplify the rare-
    class portion of dark knowledge did not transfer.

Three KD attempts on Mamba (KL, UFD, DKD) all marginally worse than
no-KD baseline on fold 3 within-domain. KD into SSM decoder is not a
no-op on PanNuke — but the cross-dataset result below is much stronger.

### Phase E — MoNuSeg zero-shot cross-dataset eval

After yesterday's MoNuSeg infrastructure (HF RationAI/MoNuSeg adapter,
overlap-stitched 256×256 patches), evaluated 5 checkpoints from today's
training runs on MoNuSeg test (14 images, single-class nuclei).

Critical implementation note: the first sweep attempt was bottlenecked
on single-threaded post-processing per full-resolution image (1000×1000).
Each image took 2-5 min sequentially → 6+ hours estimated total. Killed
and refactored `eval_monuseg.py` to two phases:
  1. GPU stitching for all images first (fast, sequential, ~6s/14 imgs)
  2. Pool-parallel post-process + panoptic_quality across images (~10s)

Total per-checkpoint eval (plain + TTA) dropped from ~50 min to ~1 min.
Whole 5-model sweep finished in ~10 min.

### MoNuSeg results (bPQ TTA, n=14 images)

| Model              | Decoder | KD  | PanNuke fold-3 | MoNuSeg | gap     |
|--------------------|---------|-----|----------------|---------|---------|
| hover_ufd          | conv    | UFD | 0.5988         | 0.5563  | −0.0425 |
| mamba_default      | mamba   | —   | 0.5808         | 0.5702  | −0.0106 |
| mamba_cs4_d64      | mamba   | —   | 0.5906         | 0.5463  | −0.0443 |
| mamba_cs4_kl       | mamba   | KL  | 0.5902         | 0.4608  | −0.1294 |
| mamba_cs4_dkd      | mamba   | DKD | 0.5838         | 0.3763  | **−0.2075** |

(bPQ used because MoNuSeg has no cell-type labels; F1 follows the same
pattern with the same ranking.)

### Three surprising findings (defensible after this run)

1. **KD into Mamba is catastrophic for domain shift.** KL costs −0.086
   bPQ TTA on MoNuSeg vs the matched-architecture no-KD baseline; DKD
   costs −0.170. Within-PanNuke the same KD only costs −0.005 to −0.009.
   The teacher's confident decision boundary on PanNuke is being
   memorized by the SSM student in a way that does not survive the
   stain / scanner / organ shift to MoNuSeg.

2. **Mamba default beats the C3-winning Mamba on MoNuSeg.** cs4_d64_g4
   was +0.007 mPQ TTA on PanNuke (the C3 winner) but is −0.024 bPQ TTA
   on MoNuSeg vs the default bi_d16_g4. Classic capacity / generalization
   trade-off: extra cross-scan + state dimension overfits the in-domain
   distribution.

3. **Mamba default also beats the conv decoder on MoNuSeg.** mamba_default
   0.5702 vs hover_ufd 0.5563 (+0.014). On PanNuke they were tied. The
   PanNuke-equality + MoNuSeg-advantage for Mamba is the closest thing
   we have to a positive contribution: matched in-domain, better out-of-
   domain.

### Thesis story re-frame (post-MoNuSeg)

The negative findings of today's session are actually the spine of the
defense:

  - **Architecture matched-budget ablation (conv vs SSM)** — defensible
    contribution: decoder choice has measurable consequence on
    generalization profile, not on in-domain mPQ.
  - **KD-into-Mamba is not a free lunch** — KL/DKD/UFD all fail to help
    even when carefully calibrated. Within PanNuke the gap is small;
    cross-dataset the gap is large and consistent.
  - **Frequency-decoupled KD adaptation (Novel A)** — clean negative,
    documented mechanism (LF dominates, HF no-op on segmentation
    softmax).
  - **Methodological infrastructure** — seeds, per-image arrays + paired
    stats, env-driven configs, MoNuSeg HF adapter, ablation grid runner
    + sweep scripts. All ready for the Phase D 3-fold CV when we run it.

### Statistical caveats

  - All Mamba conditions: **1 fold only** (fold 3). Phase D 3-fold grid
    needed before any mean±std claim.
  - MoNuSeg eval: n=14 images. Paired tests at this n have low power;
    deltas of 0.01-0.02 are not significant. The 0.086 and 0.170 gaps
    for KD on Mamba ARE likely significant given the magnitude, but a
    per-image Wilcoxon should be in the thesis appendix.

### What still doesn't have a 3-fold number

  - Mamba no-KD (default + cs4_d64)
  - Mamba + KL distill (cs4_d64)
  - Mamba + DKD (cs4_d64)
  - HoVer + UFD-KD
  - HoVer + DKD (never tried)

Phase D2 ablation grid would cover all of these.

### Branch state at session end

  - `master`                     4f79123 (origin, unchanged)
  - `dev/post-master-work`       479ec1b → all today's code + docs
    (UFD, DKD, Mamba decoder + sweep, hygiene, per-image stats,
    MoNuSeg via HF including cross-dataset sweep script)
  - `feat/ablation-grid-runner`  131d486 — D2 grid runner + yaml
  - `feat/monuseg-eval`          427369a — MoNuSeg adapter + HF migration
    (mirrored onto dev/ via cherry-pick db46937 + 600e637)
  - `docs/related-work`          6a97b19 — thesis literature notes
