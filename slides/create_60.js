// 60% progress checkpoint — 8 slides
const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.author = "Thesis progress check";
pres.title = "60% progress checkpoint";

const BG = "FAFAFA";
const INK = "1A1A1A";
const MUTED = "6B6B6B";
const RULE = "D9D9D9";
const ACCENT = "B85042";
const SOFT = "F0ECE8";
const GREEN = "2D6A4F";

// ============= 1. Title =============
let s = pres.addSlide();
s.background = { color: BG };

s.addText("ОТЧЁТ О ПРОГРЕССЕ  ·  60 %", {
  x: 0.8, y: 0.9, w: 11, h: 0.35,
  fontFace: "Calibri", fontSize: 11, bold: true, color: MUTED,
  charSpacing: 6, margin: 0,
});

s.addText("Классификация и сегментация клеток\nна медицинских изображениях\nметодом глубокого обучения", {
  x: 0.8, y: 1.6, w: 11, h: 2.6,
  fontFace: "Georgia", fontSize: 38, bold: true, color: INK,
  valign: "top", margin: 0,
});

s.addShape(pres.shapes.RECTANGLE, {
  x: 0.8, y: 4.5, w: 0.08, h: 1.2,
  fill: { color: ACCENT }, line: { color: ACCENT, width: 0 },
});

s.addText([
  { text: "Дистилляция знаний ", options: { color: INK, bold: true } },
  { text: "из CellViT-SAM-H в FastViT-S12", options: { color: INK } },
], {
  x: 1.1, y: 4.5, w: 10.5, h: 0.5,
  fontFace: "Georgia", fontSize: 18, margin: 0, valign: "top",
});

s.addText("PanNuke · 11.5M параметров · RTX 5060 Ti", {
  x: 1.1, y: 5.05, w: 10.5, h: 0.4,
  fontFace: "Georgia", fontSize: 15, italic: true, color: MUTED, margin: 0,
});

s.addText("Автор    •    Научный руководитель    •    апрель 2026", {
  x: 0.8, y: 6.8, w: 11.7, h: 0.3,
  fontFace: "Calibri", fontSize: 10, color: MUTED, margin: 0,
});

// ============= 2. Problem & Approach (condensed) =============
s = pres.addSlide();
s.background = { color: BG };

s.addText("Задача и подход", {
  x: 0.8, y: 0.6, w: 11, h: 0.6,
  fontFace: "Georgia", fontSize: 30, bold: true, color: INK, margin: 0,
});

// Problem box
s.addText("ПРОБЛЕМА", {
  x: 0.8, y: 1.5, w: 5.7, h: 0.35,
  fontFace: "Calibri", fontSize: 11, bold: true, color: MUTED, charSpacing: 4, margin: 0,
});
s.addText("SOTA-модели для сегментации ядер слишком тяжелы для клинического inference:", {
  x: 0.8, y: 1.9, w: 5.7, h: 0.8,
  fontFace: "Calibri", fontSize: 13, color: INK, margin: 0,
});
s.addText([
  { text: "CellViT-SAM-H", options: { bold: true, color: INK } },
  { text: "  —  630M параметров, ≥24 ГБ VRAM, ~500 мс/патч", options: { color: INK } },
], {
  x: 0.8, y: 2.85, w: 5.7, h: 0.6,
  fontFace: "Calibri", fontSize: 13, margin: 0,
});
s.addText("На потребительских GPU (16 ГБ и менее) запустить невозможно.", {
  x: 0.8, y: 3.5, w: 5.7, h: 0.6,
  fontFace: "Calibri", fontSize: 12, italic: true, color: MUTED, margin: 0,
});

// Vertical divider
s.addShape(pres.shapes.LINE, {
  x: 6.7, y: 1.5, w: 0, h: 5.0,
  line: { color: RULE, width: 1 },
});

// Approach box
s.addText("ПОДХОД", {
  x: 7.0, y: 1.5, w: 5.5, h: 0.35,
  fontFace: "Calibri", fontSize: 11, bold: true, color: MUTED, charSpacing: 4, margin: 0,
});
s.addText("Knowledge distillation: обучить лёгкого ученика воспроизводить поведение тяжёлого учителя.", {
  x: 7.0, y: 1.9, w: 5.5, h: 1.0,
  fontFace: "Calibri", fontSize: 13, color: INK, margin: 0,
});

// Teacher → Student mini diagram
s.addShape(pres.shapes.RECTANGLE, {
  x: 7.0, y: 3.15, w: 2.2, h: 1.2,
  fill: { color: SOFT }, line: { color: RULE, width: 1 },
});
s.addText("Teacher\n630M", {
  x: 7.0, y: 3.3, w: 2.2, h: 0.9,
  fontFace: "Calibri", fontSize: 14, bold: true, color: INK, align: "center", margin: 0,
});

s.addShape(pres.shapes.LINE, {
  x: 9.25, y: 3.75, w: 1.1, h: 0,
  line: { color: ACCENT, width: 2.5, endArrowType: "triangle" },
});

s.addShape(pres.shapes.RECTANGLE, {
  x: 10.35, y: 3.15, w: 2.2, h: 1.2,
  fill: { color: SOFT }, line: { color: ACCENT, width: 2 },
});
s.addText("Student\n11.5M", {
  x: 10.35, y: 3.3, w: 2.2, h: 0.9,
  fontFace: "Calibri", fontSize: 14, bold: true, color: INK, align: "center", margin: 0,
});

s.addText("55× сжатие, 7× ускорение inference, влезает в 2 ГБ VRAM", {
  x: 7.0, y: 4.5, w: 5.5, h: 0.5,
  fontFace: "Calibri", fontSize: 12, italic: true, color: ACCENT, margin: 0,
});

s.addText("Цель: сохранить качество teacher'а в минимально возможной модели.", {
  x: 7.0, y: 5.2, w: 5.5, h: 0.8,
  fontFace: "Calibri", fontSize: 13, color: INK, margin: 0,
});

// ============= 3. Roadmap (overall plan) =============
s = pres.addSlide();
s.background = { color: BG };

s.addText("План работы", {
  x: 0.8, y: 0.6, w: 11, h: 0.6,
  fontFace: "Georgia", fontSize: 30, bold: true, color: INK, margin: 0,
});

s.addText("Пять основных этапов", {
  x: 0.8, y: 1.3, w: 11.5, h: 0.4,
  fontFace: "Calibri", fontSize: 13, italic: true, color: MUTED, margin: 0,
});

// Timeline with checkmarks
const steps = [
  { n: "1", title: "Литературный обзор", body: "Датасеты, метрики, архитектуры", status: "done" },
  { n: "2", title: "Методология",         body: "Teacher, student, loss, KD", status: "done" },
  { n: "3", title: "Реализация",          body: "Pipeline: data → train → eval", status: "done" },
  { n: "4", title: "Эксперименты",         body: "Baseline, response KD, feature KD", status: "mostly" },
  { n: "5", title: "Защита",              body: "Расширения, 3-fold CV, текст", status: "todo" },
];

const stepY = 2.1;
const stepW = 2.3;
const gapX = 0.2;
const totalW = 5 * stepW + 4 * gapX;
const startX = (13.3 - totalW) / 2;

for (let i = 0; i < steps.length; i++) {
  const st = steps[i];
  const x = startX + i * (stepW + gapX);

  // Background card
  const cardColor = st.status === "done" ? SOFT : (st.status === "mostly" ? BG : BG);
  const borderColor = st.status === "done" ? GREEN : (st.status === "mostly" ? ACCENT : RULE);
  s.addShape(pres.shapes.RECTANGLE, {
    x: x, y: stepY, w: stepW, h: 3.0,
    fill: { color: cardColor }, line: { color: borderColor, width: st.status === "todo" ? 1 : 2 },
  });

  // Number
  s.addText(st.n, {
    x: x, y: stepY + 0.2, w: stepW, h: 0.9,
    fontFace: "Georgia", fontSize: 56, bold: true,
    color: st.status === "done" ? GREEN : (st.status === "mostly" ? ACCENT : MUTED),
    align: "center", margin: 0,
  });

  // Title
  s.addText(st.title, {
    x: x + 0.15, y: stepY + 1.3, w: stepW - 0.3, h: 0.5,
    fontFace: "Calibri", fontSize: 14, bold: true, color: INK, align: "center", margin: 0,
  });

  // Body
  s.addText(st.body, {
    x: x + 0.15, y: stepY + 1.85, w: stepW - 0.3, h: 0.7,
    fontFace: "Calibri", fontSize: 11, color: MUTED, align: "center", margin: 0,
  });

  // Status tag
  const tag = st.status === "done" ? "✓  ГОТОВО" : (st.status === "mostly" ? "  В РАБОТЕ" : "  ПЛАНИРУЕТСЯ");
  const tagColor = st.status === "done" ? GREEN : (st.status === "mostly" ? ACCENT : MUTED);
  s.addText(tag, {
    x: x, y: stepY + 2.6, w: stepW, h: 0.3,
    fontFace: "Calibri", fontSize: 10, bold: true, color: tagColor,
    align: "center", charSpacing: 3, margin: 0,
  });
}

// Progress bar bottom
s.addShape(pres.shapes.RECTANGLE, {
  x: 0.8, y: 6.0, w: 11.7, h: 0.15,
  fill: { color: "E8E8E8" }, line: { color: "E8E8E8", width: 0 },
});
s.addShape(pres.shapes.RECTANGLE, {
  x: 0.8, y: 6.0, w: 11.7 * 0.6, h: 0.15,
  fill: { color: ACCENT }, line: { color: ACCENT, width: 0 },
});
s.addText("Общий прогресс:  60 %", {
  x: 0.8, y: 6.3, w: 11.7, h: 0.4,
  fontFace: "Calibri", fontSize: 12, bold: true, color: INK, margin: 0,
});

// ============= 4. What's done (text + evidence) =============
s = pres.addSlide();
s.background = { color: BG };

s.addText("Что уже сделано", {
  x: 0.8, y: 0.6, w: 11, h: 0.6,
  fontFace: "Georgia", fontSize: 30, bold: true, color: INK, margin: 0,
});

const doneItems = [
  {
    title: "Обзор датасетов и архитектур",
    body: "PanNuke, MoNuSeg, CoNSeP, Lizard. U-Net, HoVer-Net, StarDist, CellViT, NuLite. Метрики PQ/mPQ/bPQ.",
  },
  {
    title: "Методология KD",
    body: "Response-based (KL на логитах), feature-based (matching промежуточных представлений), формализация в коде.",
  },
  {
    title: "Рабочий пайплайн на RTX 5060 Ti",
    body: "Precompute soft targets (~1 ч), обучение ученика (~1 ч), eval + TTA. Всё в 16 ГБ VRAM.",
  },
  {
    title: "Найдены и исправлены 2 методологические ошибки",
    body: "(1) рассогласование soft targets с аугментациями, (2) неверный протокол подсчёта mPQ.",
  },
  {
    title: "Обучены и оценены 5 вариантов ученика",
    body: "ConvNeXt baseline, FastViT baseline, FastViT + response KD, FastViT + feature KD, все +TTA.",
  },
  {
    title: "Первый результат: 80 % качества teacher'а при 55× сжатии",
    body: "FastViT-S12 + response KD + TTA достигает mPQ = 0.472 (teacher: 0.592) при 11.5M параметрах.",
  },
];

for (let i = 0; i < doneItems.length; i++) {
  const item = doneItems[i];
  const col = i % 2;
  const row = Math.floor(i / 2);
  const x = 0.8 + col * 5.95;
  const y = 1.5 + row * 1.75;

  // Green checkmark
  s.addShape(pres.shapes.OVAL, {
    x: x, y: y, w: 0.4, h: 0.4,
    fill: { color: GREEN }, line: { color: GREEN, width: 0 },
  });
  s.addText("✓", {
    x: x, y: y, w: 0.4, h: 0.4,
    fontFace: "Calibri", fontSize: 16, bold: true, color: "FFFFFF",
    align: "center", valign: "middle", margin: 0,
  });

  // Title
  s.addText(item.title, {
    x: x + 0.55, y: y - 0.05, w: 5.3, h: 0.5,
    fontFace: "Calibri", fontSize: 13, bold: true, color: INK, margin: 0,
  });

  // Body
  s.addText(item.body, {
    x: x + 0.55, y: y + 0.45, w: 5.3, h: 1.15,
    fontFace: "Calibri", fontSize: 11, color: MUTED, margin: 0,
  });
}

// ============= 5. Preliminary results =============
s = pres.addSlide();
s.background = { color: BG };

s.addText("Предварительные результаты", {
  x: 0.8, y: 0.6, w: 11, h: 0.6,
  fontFace: "Georgia", fontSize: 30, bold: true, color: INK, margin: 0,
});
s.addText("PanNuke fold 3 · единый протокол оценки", {
  x: 0.8, y: 1.25, w: 11.5, h: 0.4,
  fontFace: "Calibri", fontSize: 13, italic: true, color: MUTED, margin: 0,
});

// Main table
const headerOpts = {
  fontFace: "Calibri", fontSize: 11, bold: true, color: MUTED, charSpacing: 2,
  fill: { color: BG }, valign: "middle",
};
const rowOpts = {
  fontFace: "Calibri", fontSize: 13, color: INK, valign: "middle",
};
const bestOpts = {
  fontFace: "Calibri", fontSize: 13, bold: true, color: ACCENT, valign: "middle",
  fill: { color: SOFT },
};

const tbl = [
  [
    { text: "МОДЕЛЬ", options: headerOpts },
    { text: "PARAMS", options: { ...headerOpts, align: "right" } },
    { text: "mPQ", options: { ...headerOpts, align: "right" } },
    { text: "bPQ", options: { ...headerOpts, align: "right" } },
    { text: "F1", options: { ...headerOpts, align: "right" } },
  ],
  [
    { text: "CellViT-SAM-H (teacher)", options: rowOpts },
    { text: "630M", options: { ...rowOpts, align: "right" } },
    { text: "0.592", options: { ...rowOpts, align: "right" } },
    { text: "0.664", options: { ...rowOpts, align: "right" } },
    { text: "0.784", options: { ...rowOpts, align: "right" } },
  ],
  [
    { text: "ConvNeXt-Tiny baseline", options: rowOpts },
    { text: "31.9M", options: { ...rowOpts, align: "right" } },
    { text: "0.468", options: { ...rowOpts, align: "right" } },
    { text: "0.591", options: { ...rowOpts, align: "right" } },
    { text: "0.719", options: { ...rowOpts, align: "right" } },
  ],
  [
    { text: "FastViT-S12 baseline", options: rowOpts },
    { text: "11.5M", options: { ...rowOpts, align: "right" } },
    { text: "0.456", options: { ...rowOpts, align: "right" } },
    { text: "0.578", options: { ...rowOpts, align: "right" } },
    { text: "0.706", options: { ...rowOpts, align: "right" } },
  ],
  [
    { text: "FastViT-S12 + response KD", options: rowOpts },
    { text: "11.5M", options: { ...rowOpts, align: "right" } },
    { text: "0.467", options: { ...rowOpts, align: "right" } },
    { text: "0.598", options: { ...rowOpts, align: "right" } },
    { text: "0.724", options: { ...rowOpts, align: "right" } },
  ],
  [
    { text: "FastViT-S12 + response KD + TTA  ★", options: bestOpts },
    { text: "11.5M", options: { ...bestOpts, align: "right" } },
    { text: "0.472", options: { ...bestOpts, align: "right" } },
    { text: "0.604", options: { ...bestOpts, align: "right" } },
    { text: "0.729", options: { ...bestOpts, align: "right" } },
  ],
];

s.addTable(tbl, {
  x: 0.8, y: 1.9, w: 11.7,
  colW: [5.7, 1.5, 1.5, 1.5, 1.5],
  rowH: 0.5,
  border: { pt: 0, color: BG },
  fontFace: "Calibri",
});

// Three key takeaways below
const takeaways = [
  { label: "компрессия", value: "55×", color: INK },
  { label: "качества teacher'а", value: "80 %", color: ACCENT },
  { label: "прирост от KD", value: "+2.4 %", color: GREEN },
];
for (let i = 0; i < takeaways.length; i++) {
  const t = takeaways[i];
  const x = 0.8 + i * 4.0;
  s.addText(t.value, {
    x: x, y: 5.8, w: 3.8, h: 0.8,
    fontFace: "Georgia", fontSize: 42, bold: true, color: t.color,
    align: "center", margin: 0,
  });
  s.addText(t.label, {
    x: x, y: 6.6, w: 3.8, h: 0.35,
    fontFace: "Calibri", fontSize: 11, color: MUTED,
    align: "center", charSpacing: 3, margin: 0,
  });
}

// ============= 6. Key findings (brief) =============
s = pres.addSlide();
s.background = { color: BG };

s.addText("Ключевые методологические находки", {
  x: 0.8, y: 0.6, w: 11.5, h: 0.6,
  fontFace: "Georgia", fontSize: 28, bold: true, color: INK, margin: 0,
});
s.addText("Не запланированное, но значимое — в сжатой форме, детали в дипломе", {
  x: 0.8, y: 1.25, w: 11.5, h: 0.4,
  fontFace: "Calibri", fontSize: 12, italic: true, color: MUTED, margin: 0,
});

// Finding 1
s.addShape(pres.shapes.RECTANGLE, {
  x: 0.8, y: 2.0, w: 5.8, h: 4.5,
  fill: { color: SOFT }, line: { color: RULE, width: 1 },
});
s.addText("НАХОДКА № 1", {
  x: 1.0, y: 2.15, w: 5.4, h: 0.35,
  fontFace: "Calibri", fontSize: 10, bold: true, color: ACCENT, charSpacing: 4, margin: 0,
});
s.addText("Рассогласование soft targets", {
  x: 1.0, y: 2.55, w: 5.4, h: 0.55,
  fontFace: "Georgia", fontSize: 18, bold: true, color: INK, margin: 0,
});
s.addText("Soft targets не проходили через пространственные аугментации, только 12.5% семплов были корректны.", {
  x: 1.0, y: 3.2, w: 5.4, h: 1.1,
  fontFace: "Calibri", fontSize: 12, color: INK, margin: 0,
});
s.addText("0.5³ = 12.5 %", {
  x: 1.0, y: 4.3, w: 5.4, h: 0.8,
  fontFace: "Georgia", fontSize: 34, bold: true, color: ACCENT,
  align: "center", margin: 0,
});
s.addText("FIX", {
  x: 1.0, y: 5.25, w: 5.4, h: 0.3,
  fontFace: "Calibri", fontSize: 10, bold: true, color: GREEN, charSpacing: 4, margin: 0,
});
s.addText("Synchronous augmentation через additional masks в Albumentations. KD стал работать (+0.011 mPQ).", {
  x: 1.0, y: 5.55, w: 5.4, h: 0.9,
  fontFace: "Calibri", fontSize: 11, color: INK, margin: 0,
});

// Finding 2
s.addShape(pres.shapes.RECTANGLE, {
  x: 6.8, y: 2.0, w: 5.8, h: 4.5,
  fill: { color: SOFT }, line: { color: RULE, width: 1 },
});
s.addText("НАХОДКА № 2", {
  x: 7.0, y: 2.15, w: 5.4, h: 0.35,
  fontFace: "Calibri", fontSize: 10, bold: true, color: ACCENT, charSpacing: 4, margin: 0,
});
s.addText("Протокол подсчёта mPQ", {
  x: 7.0, y: 2.55, w: 5.4, h: 0.55,
  fontFace: "Georgia", fontSize: 18, bold: true, color: INK, margin: 0,
});
s.addText("Наивный усреднитель (PQ = 0 для отсутствующих классов) систематически занижал mPQ на PanNuke.", {
  x: 7.0, y: 3.2, w: 5.4, h: 1.1,
  fontFace: "Calibri", fontSize: 12, color: INK, margin: 0,
});
s.addText([
  { text: "0.184 ", options: { color: MUTED } },
  { text: "→ ", options: { color: INK } },
  { text: "0.468", options: { color: ACCENT, bold: true } },
], {
  x: 7.0, y: 4.3, w: 5.4, h: 0.8,
  fontFace: "Georgia", fontSize: 34, bold: true,
  align: "center", margin: 0,
});
s.addText("FIX", {
  x: 7.0, y: 5.25, w: 5.4, h: 0.3,
  fontFace: "Calibri", fontSize: 10, bold: true, color: GREEN, charSpacing: 4, margin: 0,
});
s.addText("Переход на стандарт CellViT/NuLite: nanmean по классам per image. Числа стали сопоставимы с литературой.", {
  x: 7.0, y: 5.55, w: 5.4, h: 0.9,
  fontFace: "Calibri", fontSize: 11, color: INK, margin: 0,
});

// ============= 7. Remaining work =============
s = pres.addSlide();
s.background = { color: BG };

s.addText("Что осталось до защиты", {
  x: 0.8, y: 0.6, w: 11, h: 0.6,
  fontFace: "Georgia", fontSize: 30, bold: true, color: INK, margin: 0,
});
s.addText("Оставшиеся 40 % с приоритетом и оценкой времени", {
  x: 0.8, y: 1.25, w: 11.5, h: 0.4,
  fontFace: "Calibri", fontSize: 13, italic: true, color: MUTED, margin: 0,
});

const todo = [
  { priority: "P0", title: "Полная 3-fold кросс-валидация", body: "Обучить модели на 3 split'ах для публикационных (mean ± std) чисел", time: "6–8 ч compute" },
  { priority: "P0", title: "Сравнение с NuLite (2026)",       body: "Прогнать их pretrained чекпоинты на нашем eval pipeline", time: "1–2 дня" },
  { priority: "P1", title: "External eval: MoNuSeg",           body: "Проверка обобщающей способности без дообучения",         time: "1 день" },
  { priority: "P1", title: "Финализация текста дипломки",     body: "Глава 4 с новыми результатами, обсуждение, оформление",   time: "3–5 дней" },
  { priority: "P2", title: "(опция) Feature KD с β ∈ [0.3, 0.5]", body: "Текущий β=1.0 доминирует — попробовать меньше",       time: "2–3 ч" },
  { priority: "P2", title: "(опция) INT8 PTQ",                 body: "Демонстрация дополнительного 4× сжатия для edge",        time: "1 день" },
];

for (let i = 0; i < todo.length; i++) {
  const t = todo[i];
  const y = 1.95 + i * 0.78;
  const prioColor = t.priority === "P0" ? ACCENT : (t.priority === "P1" ? INK : MUTED);

  // Priority badge
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.8, y: y + 0.1, w: 0.5, h: 0.5,
    fill: { color: prioColor }, line: { color: prioColor, width: 0 },
  });
  s.addText(t.priority, {
    x: 0.8, y: y + 0.1, w: 0.5, h: 0.5,
    fontFace: "Calibri", fontSize: 10, bold: true, color: "FFFFFF",
    align: "center", valign: "middle", margin: 0,
  });

  // Title + body
  s.addText(t.title, {
    x: 1.5, y: y + 0.05, w: 8.0, h: 0.4,
    fontFace: "Calibri", fontSize: 13, bold: true, color: INK, margin: 0,
  });
  s.addText(t.body, {
    x: 1.5, y: y + 0.4, w: 8.0, h: 0.3,
    fontFace: "Calibri", fontSize: 11, color: MUTED, margin: 0,
  });

  // Time estimate
  s.addText(t.time, {
    x: 9.7, y: y + 0.2, w: 2.8, h: 0.4,
    fontFace: "Calibri", fontSize: 12, italic: true, color: MUTED,
    align: "right", margin: 0,
  });
}

// Legend
s.addText("P0 — обязательно для защиты   ·   P1 — желательно   ·   P2 — если останется время", {
  x: 0.8, y: 6.6, w: 11.7, h: 0.3,
  fontFace: "Calibri", fontSize: 11, italic: true, color: MUTED, margin: 0,
});

// ============= 8. Timeline & blockers =============
s = pres.addSlide();
s.background = { color: BG };

s.addText("График до защиты", {
  x: 0.8, y: 0.6, w: 11, h: 0.6,
  fontFace: "Georgia", fontSize: 30, bold: true, color: INK, margin: 0,
});

// Timeline weeks
const weeks = [
  { week: "Неделя 1",  tasks: "3-fold CV + NuLite" },
  { week: "Неделя 2",  tasks: "MoNuSeg external eval" },
  { week: "Неделя 3",  tasks: "Финал текста дипломки" },
  { week: "Неделя 4",  tasks: "Ревью + подготовка к защите" },
];

const tlY = 1.8;
const tlW = 2.85;
const tlGap = 0.15;

for (let i = 0; i < weeks.length; i++) {
  const w = weeks[i];
  const x = 0.8 + i * (tlW + tlGap);

  s.addShape(pres.shapes.RECTANGLE, {
    x: x, y: tlY, w: tlW, h: 2.0,
    fill: { color: SOFT }, line: { color: RULE, width: 1 },
  });

  s.addText(w.week, {
    x: x + 0.15, y: tlY + 0.25, w: tlW - 0.3, h: 0.4,
    fontFace: "Calibri", fontSize: 11, bold: true, color: MUTED, charSpacing: 3, margin: 0,
  });

  s.addText(w.tasks, {
    x: x + 0.15, y: tlY + 0.75, w: tlW - 0.3, h: 1.1,
    fontFace: "Georgia", fontSize: 14, bold: true, color: INK, margin: 0,
  });

  // arrow between weeks
  if (i < weeks.length - 1) {
    s.addShape(pres.shapes.LINE, {
      x: x + tlW, y: tlY + 1.0, w: tlGap, h: 0,
      line: { color: ACCENT, width: 1.5, endArrowType: "triangle" },
    });
  }
}

// Risks / blockers
s.addText("РИСКИ  /  ОТКРЫТЫЕ ВОПРОСЫ", {
  x: 0.8, y: 4.3, w: 11.7, h: 0.35,
  fontFace: "Calibri", fontSize: 11, bold: true, color: MUTED, charSpacing: 4, margin: 0,
});

const risks = [
  "Compute budget: 3-fold CV занимает ~8 часов на RTX 5060 Ti. Потенциальная аренда A4000/4090 на vast.ai (~$3–5) ускоряет в 3×.",
  "FP16 нестабильность: в одном прогоне baseline диверджировал на эпохе 48. Best_model сохранён, но надо следить.",
  "NuLite pretrained: надо адаптировать их inference pipeline к нашему eval protocol. Может потребовать 1–2 дня отладки.",
];

for (let i = 0; i < risks.length; i++) {
  const r = risks[i];
  const y = 4.8 + i * 0.55;

  // warning dot
  s.addShape(pres.shapes.OVAL, {
    x: 0.8, y: y + 0.1, w: 0.25, h: 0.25,
    fill: { color: ACCENT, transparency: 20 }, line: { color: ACCENT, width: 0 },
  });

  s.addText(r, {
    x: 1.2, y: y, w: 11.3, h: 0.5,
    fontFace: "Calibri", fontSize: 11, color: INK, margin: 0,
  });
}

// Save
pres.writeFile({ fileName: "thesis_60_progress.pptx" })
  .then(fileName => console.log(`OK: ${fileName}`));
