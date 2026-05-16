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
