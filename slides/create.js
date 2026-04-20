// Minimalist defense slides
const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";  // 13.3" × 7.5"
pres.author = "Thesis";
pres.title = "Knowledge distillation for cell nuclei segmentation";

// Palette — charcoal minimal + one warm accent
const BG = "FAFAFA";
const INK = "1A1A1A";       // primary text
const MUTED = "6B6B6B";     // secondary text
const RULE = "D9D9D9";      // thin rules
const ACCENT = "B85042";    // terracotta, single accent for emphasis
const SOFT = "F0ECE8";      // soft background tiles

// ============= Slide 1 — Title =============
let s = pres.addSlide();
s.background = { color: BG };

// Small eyebrow
s.addText("БАКАЛАВРСКАЯ ДИПЛОМНАЯ РАБОТА", {
  x: 0.8, y: 0.9, w: 11, h: 0.35,
  fontFace: "Calibri", fontSize: 11, bold: true, color: MUTED,
  charSpacing: 6, margin: 0,
});

// Big title
s.addText("Классификация и сегментация клеток\nна медицинских изображениях\nметодом глубокого обучения", {
  x: 0.8, y: 1.6, w: 11, h: 2.6,
  fontFace: "Georgia", fontSize: 40, bold: true, color: INK,
  valign: "top", margin: 0,
});

// Accent vertical line
s.addShape(pres.shapes.RECTANGLE, {
  x: 0.8, y: 4.4, w: 0.08, h: 1.3,
  fill: { color: ACCENT }, line: { color: ACCENT, width: 0 },
});

// Subtitle
s.addText([
  { text: "Дистилляция знаний ", options: { color: INK, bold: true } },
  { text: "из CellViT-SAM-H в lightweight ученика FastViT-S12", options: { color: INK } },
], {
  x: 1.1, y: 4.4, w: 10.5, h: 0.5,
  fontFace: "Georgia", fontSize: 20, margin: 0, valign: "top",
});

s.addText("на датасете PanNuke", {
  x: 1.1, y: 5.0, w: 10.5, h: 0.4,
  fontFace: "Georgia", fontSize: 18, italic: true, color: MUTED, margin: 0,
});

// Footer
s.addText("Автор    •    Научный руководитель    •    2026", {
  x: 0.8, y: 6.8, w: 11.7, h: 0.3,
  fontFace: "Calibri", fontSize: 10, color: MUTED, margin: 0,
});

// ============= Slide 2 — Problem =============
s = pres.addSlide();
s.background = { color: BG };

s.addText("Проблема", {
  x: 0.8, y: 0.6, w: 11, h: 0.6,
  fontFace: "Georgia", fontSize: 36, bold: true, color: INK, margin: 0,
});

s.addText("SOTA-модели для сегментации ядер слишком тяжелы для клинического inference.", {
  x: 0.8, y: 1.5, w: 11.5, h: 0.8,
  fontFace: "Calibri", fontSize: 18, color: MUTED, margin: 0,
});

// Big stats row
const stats = [
  { val: "630M",   label: "параметров",           x: 0.8 },
  { val: "≥ 24 ГБ", label: "видеопамяти",         x: 5.0 },
  { val: "500 мс", label: "на патч 256×256",      x: 9.2 },
];
for (const st of stats) {
  s.addText(st.val, {
    x: st.x, y: 3.0, w: 3.8, h: 1.3,
    fontFace: "Georgia", fontSize: 72, bold: true, color: INK, margin: 0,
    align: "left",
  });
  s.addText(st.label, {
    x: st.x, y: 4.4, w: 3.8, h: 0.4,
    fontFace: "Calibri", fontSize: 12, color: MUTED, margin: 0,
    charSpacing: 2,
  });
}

// Footnote
s.addText("CellViT-SAM-H (Hörst et al., 2024) — state-of-the-art на PanNuke (mPQ 0.51 published)", {
  x: 0.8, y: 6.2, w: 11.7, h: 0.4,
  fontFace: "Calibri", fontSize: 11, italic: true, color: MUTED, margin: 0,
});
// thin rule
s.addShape(pres.shapes.LINE, {
  x: 0.8, y: 6.15, w: 11.7, h: 0,
  line: { color: RULE, width: 0.5 },
});

// ============= Slide 3 — Approach =============
s = pres.addSlide();
s.background = { color: BG };

s.addText("Подход", {
  x: 0.8, y: 0.6, w: 11, h: 0.6,
  fontFace: "Georgia", fontSize: 36, bold: true, color: INK, margin: 0,
});

s.addText("Knowledge distillation: сжатие SOTA-teacher'а в лёгкого ученика.", {
  x: 0.8, y: 1.5, w: 11.5, h: 0.6,
  fontFace: "Calibri", fontSize: 18, color: MUTED, margin: 0,
});

// Teacher box (left)
s.addShape(pres.shapes.RECTANGLE, {
  x: 1.0, y: 3.0, w: 3.5, h: 2.3,
  fill: { color: SOFT }, line: { color: RULE, width: 1 },
});
s.addText("TEACHER", {
  x: 1.2, y: 3.15, w: 3.1, h: 0.3,
  fontFace: "Calibri", fontSize: 10, bold: true, color: MUTED, charSpacing: 4, margin: 0,
});
s.addText("CellViT-SAM-H", {
  x: 1.2, y: 3.55, w: 3.1, h: 0.5,
  fontFace: "Georgia", fontSize: 20, bold: true, color: INK, margin: 0,
});
s.addText("ViT-H из SAM\n+ HoVer-Net decoder\n+ 3 головы", {
  x: 1.2, y: 4.1, w: 3.1, h: 0.9,
  fontFace: "Calibri", fontSize: 12, color: INK, margin: 0,
});
s.addText("630M params", {
  x: 1.2, y: 5.0, w: 3.1, h: 0.3,
  fontFace: "Calibri", fontSize: 11, italic: true, color: MUTED, margin: 0,
});

// Arrow / KD label (middle)
s.addShape(pres.shapes.LINE, {
  x: 4.7, y: 4.15, w: 3.7, h: 0,
  line: { color: ACCENT, width: 2.5, endArrowType: "triangle" },
});
s.addText("soft targets", {
  x: 4.7, y: 3.5, w: 3.7, h: 0.4,
  fontFace: "Calibri", fontSize: 13, italic: true, color: ACCENT, bold: true,
  align: "center", margin: 0,
});
s.addText("α=0.2,  T=10\nresponse KD", {
  x: 4.7, y: 4.35, w: 3.7, h: 0.6,
  fontFace: "Calibri", fontSize: 11, color: MUTED, align: "center", margin: 0,
});

// Student box (right)
s.addShape(pres.shapes.RECTANGLE, {
  x: 8.6, y: 3.0, w: 3.5, h: 2.3,
  fill: { color: SOFT }, line: { color: ACCENT, width: 2 },
});
s.addText("STUDENT", {
  x: 8.8, y: 3.15, w: 3.1, h: 0.3,
  fontFace: "Calibri", fontSize: 10, bold: true, color: MUTED, charSpacing: 4, margin: 0,
});
s.addText("FastViT-S12", {
  x: 8.8, y: 3.55, w: 3.1, h: 0.5,
  fontFace: "Georgia", fontSize: 20, bold: true, color: INK, margin: 0,
});
s.addText("Hybrid CNN+ViT\n+ FPN decoder\n+ 3 головы + tissue aux", {
  x: 8.8, y: 4.1, w: 3.1, h: 0.9,
  fontFace: "Calibri", fontSize: 12, color: INK, margin: 0,
});
s.addText("11.5M params  ·  55× меньше", {
  x: 8.8, y: 5.0, w: 3.1, h: 0.3,
  fontFace: "Calibri", fontSize: 11, italic: true, color: ACCENT, margin: 0,
});

// Footer note
s.addText("Teacher запускается один раз (fp16 inference, cache на диск), ученик обучается на cached soft targets.", {
  x: 0.8, y: 6.3, w: 11.7, h: 0.4,
  fontFace: "Calibri", fontSize: 11, italic: true, color: MUTED, margin: 0,
});

// ============= Slide 4 — Finding 1 =============
s = pres.addSlide();
s.background = { color: BG };

s.addText("Находка № 1", {
  x: 0.8, y: 0.6, w: 11, h: 0.35,
  fontFace: "Calibri", fontSize: 11, bold: true, color: MUTED,
  charSpacing: 6, margin: 0,
});
s.addText("Пространственное рассогласование soft targets", {
  x: 0.8, y: 1.0, w: 11.5, h: 0.6,
  fontFace: "Georgia", fontSize: 30, bold: true, color: INK, margin: 0,
});

s.addText("Soft targets предрасчитаны на оригинальных изображениях. Без синхронной аугментации только часть батчей корректна:", {
  x: 0.8, y: 1.9, w: 11.5, h: 0.8,
  fontFace: "Calibri", fontSize: 15, color: INK, margin: 0,
});

// The big formula
s.addText([
  { text: "0.5", options: { bold: true, color: INK } },
  { text: "³", options: { bold: true, color: INK, superscript: true } },
  { text: "  =  12.5 %", options: { bold: true, color: ACCENT } },
], {
  x: 0.8, y: 3.3, w: 11.5, h: 1.3,
  fontFace: "Georgia", fontSize: 72, margin: 0, align: "center",
});

s.addText("согласованных семплов при трёх пространственных аугментациях p = 0.5 (hflip, vflip, rotate90)", {
  x: 0.8, y: 4.7, w: 11.5, h: 0.5,
  fontFace: "Calibri", fontSize: 13, italic: true, color: MUTED,
  align: "center", margin: 0,
});

// Fix callout box
s.addShape(pres.shapes.RECTANGLE, {
  x: 0.8, y: 5.6, w: 11.7, h: 1.2,
  fill: { color: SOFT }, line: { color: RULE, width: 1 },
});
s.addText("FIX", {
  x: 1.0, y: 5.75, w: 0.7, h: 0.3,
  fontFace: "Calibri", fontSize: 10, bold: true, color: ACCENT, charSpacing: 4, margin: 0,
});
s.addText("Soft targets проходят через те же spatial-трансформации, что и картинка — через additional masks в Albumentations. KD стал давать +0.011 mPQ (+2.4%) на том же студенте.", {
  x: 1.8, y: 5.75, w: 10.5, h: 0.9,
  fontFace: "Calibri", fontSize: 13, color: INK, margin: 0,
});

// ============= Slide 5 — Finding 2 =============
s = pres.addSlide();
s.background = { color: BG };

s.addText("Находка № 2", {
  x: 0.8, y: 0.6, w: 11, h: 0.35,
  fontFace: "Calibri", fontSize: 11, bold: true, color: MUTED,
  charSpacing: 6, margin: 0,
});
s.addText("Неверный протокол подсчёта mPQ", {
  x: 0.8, y: 1.0, w: 11.5, h: 0.6,
  fontFace: "Georgia", fontSize: 30, bold: true, color: INK, margin: 0,
});

s.addText("CellViT/NuLite репортят иерархическое усреднение: nanmean по классам на каждом изображении → mean по изображениям. Наивный усреднитель с PQ = 0 для отсутствующих классов систематически занижает.", {
  x: 0.8, y: 1.9, w: 11.5, h: 1.1,
  fontFace: "Calibri", fontSize: 14, color: INK, margin: 0,
});

// Before / After comparison
const boxY = 3.3;
const boxH = 2.8;

// "Before"
s.addShape(pres.shapes.RECTANGLE, {
  x: 0.8, y: boxY, w: 5.7, h: boxH,
  fill: { color: SOFT }, line: { color: RULE, width: 1 },
});
s.addText("ДО", {
  x: 1.05, y: boxY + 0.2, w: 1.5, h: 0.3,
  fontFace: "Calibri", fontSize: 10, bold: true, color: MUTED, charSpacing: 4, margin: 0,
});
s.addText("0.184", {
  x: 1.05, y: boxY + 0.6, w: 5.3, h: 1.4,
  fontFace: "Georgia", fontSize: 72, bold: true, color: MUTED, margin: 0,
});
s.addText("багнутый протокол\n(PQ = 0 для отсутствующих классов)", {
  x: 1.05, y: boxY + 2.0, w: 5.3, h: 0.7,
  fontFace: "Calibri", fontSize: 11, italic: true, color: MUTED, margin: 0,
});

// "After"
s.addShape(pres.shapes.RECTANGLE, {
  x: 6.8, y: boxY, w: 5.7, h: boxH,
  fill: { color: SOFT }, line: { color: ACCENT, width: 2 },
});
s.addText("ПОСЛЕ", {
  x: 7.05, y: boxY + 0.2, w: 1.5, h: 0.3,
  fontFace: "Calibri", fontSize: 10, bold: true, color: ACCENT, charSpacing: 4, margin: 0,
});
s.addText("0.468", {
  x: 7.05, y: boxY + 0.6, w: 5.3, h: 1.4,
  fontFace: "Georgia", fontSize: 72, bold: true, color: INK, margin: 0,
});
s.addText("CellViT/NuLite протокол\n(nanmean по классам per image)", {
  x: 7.05, y: boxY + 2.0, w: 5.3, h: 0.7,
  fontFace: "Calibri", fontSize: 11, italic: true, color: INK, margin: 0,
});

s.addText("тот же ConvNeXt-Tiny baseline, тот же checkpoint — разница только в реализации метрики", {
  x: 0.8, y: 6.4, w: 11.7, h: 0.4,
  fontFace: "Calibri", fontSize: 11, italic: true, color: MUTED, align: "center", margin: 0,
});

// ============= Slide 6 — Architecture =============
s = pres.addSlide();
s.background = { color: BG };

s.addText("Архитектура ученика", {
  x: 0.8, y: 0.6, w: 11, h: 0.6,
  fontFace: "Georgia", fontSize: 30, bold: true, color: INK, margin: 0,
});

s.addText("FastViT-S12 encoder (Apple, 2023) + HoVer-Net style FPN decoder + 3 головы + tissue aux", {
  x: 0.8, y: 1.4, w: 11.5, h: 0.5,
  fontFace: "Calibri", fontSize: 14, color: MUTED, margin: 0,
});

// Encoder blocks (horizontal flow)
const encoderBlocks = [
  { label: "Stage 1", shape: "64×64×64",   x: 0.8 },
  { label: "Stage 2", shape: "32×32×128",  x: 3.3 },
  { label: "Stage 3", shape: "16×16×256",  x: 5.8 },
  { label: "Stage 4", shape: "8×8×512",    x: 8.3 },
];
for (const b of encoderBlocks) {
  s.addShape(pres.shapes.RECTANGLE, {
    x: b.x, y: 2.5, w: 2.3, h: 1.4,
    fill: { color: SOFT }, line: { color: RULE, width: 1 },
  });
  s.addText(b.label, {
    x: b.x, y: 2.65, w: 2.3, h: 0.35,
    fontFace: "Calibri", fontSize: 11, bold: true, color: MUTED,
    charSpacing: 3, align: "center", margin: 0,
  });
  s.addText(b.shape, {
    x: b.x, y: 3.1, w: 2.3, h: 0.5,
    fontFace: "Consolas", fontSize: 14, color: INK, align: "center", margin: 0,
  });
}

// Decoder + heads band
s.addShape(pres.shapes.RECTANGLE, {
  x: 0.8, y: 4.3, w: 9.8, h: 1.3,
  fill: { color: BG }, line: { color: ACCENT, width: 1 },
});
s.addText("FPN decoder  ·  256 → 128 → 64 → 32 каналов  ·  upsample до 256×256", {
  x: 0.8, y: 4.45, w: 9.8, h: 0.45,
  fontFace: "Calibri", fontSize: 13, color: INK, align: "center", margin: 0,
});
s.addText("binary head  •  HV head (tanh)  •  type head (6 классов)  •  tissue head (19 классов)", {
  x: 0.8, y: 4.95, w: 9.8, h: 0.55,
  fontFace: "Calibri", fontSize: 11, italic: true, color: MUTED, align: "center", margin: 0,
});

// Big param count callout on the right
s.addText("11.5M", {
  x: 10.6, y: 2.5, w: 1.9, h: 1.3,
  fontFace: "Georgia", fontSize: 40, bold: true, color: ACCENT, align: "right", margin: 0,
});
s.addText("параметров", {
  x: 10.6, y: 3.85, w: 1.9, h: 0.4,
  fontFace: "Calibri", fontSize: 11, color: MUTED, align: "right", margin: 0,
});

// Footer
s.addText("Encoder 8.3M (pretrained ImageNet-1K) + Decoder 3.1M + Heads ~30K", {
  x: 0.8, y: 6.3, w: 11.7, h: 0.4,
  fontFace: "Calibri", fontSize: 11, italic: true, color: MUTED, margin: 0,
});

// ============= Slide 7 — Training recipe =============
s = pres.addSlide();
s.background = { color: BG };

s.addText("Рецепт обучения", {
  x: 0.8, y: 0.6, w: 11, h: 0.6,
  fontFace: "Georgia", fontSize: 30, bold: true, color: INK, margin: 0,
});

s.addText("Что реально даёт качество помимо KD", {
  x: 0.8, y: 1.35, w: 11.5, h: 0.4,
  fontFace: "Calibri", fontSize: 14, italic: true, color: MUTED, margin: 0,
});

// Recipe grid 2×4
const recipes = [
  { title: "Focal Tversky Loss",     body: "для бинарной головы; штрафует false-negatives на мелких ядрах" },
  { title: "Tissue aux head",         body: "19-классовая классификация ткани — multi-task регуляризация" },
  { title: "Class-balanced sampling", body: "Weighted sampler по частоте классов; поднимает Dead (<2% патчей)" },
  { title: "Strong augmentations",    body: "Elastic + Affine + ColorJitter + CoarseDropout + spatial" },
  { title: "AdamW β = (0.85, 0.95)",  body: "быстрее адаптация к changing gradients, из NuLite recipe" },
  { title: "Early stopping",          body: "patience 20 на val mPQ каждую эпоху, ловит пик" },
  { title: "FP16 mixed precision",    body: "x2 память, ~x1.5 скорость; fp32 softmax/KL для стабильности" },
  { title: "Test-time augmentation",  body: "8-way flips/rotations с sign correction HV; +0.005 mPQ" },
];
const cols = 2;
const cellW = 5.85;
const cellH = 1.15;
const startX = 0.8;
const startY = 2.0;
const gap = 0.15;

for (let i = 0; i < recipes.length; i++) {
  const r = recipes[i];
  const col = i % cols;
  const row = Math.floor(i / cols);
  const cx = startX + col * (cellW + gap);
  const cy = startY + row * (cellH + gap);

  // Accent stripe
  s.addShape(pres.shapes.RECTANGLE, {
    x: cx, y: cy, w: 0.06, h: cellH,
    fill: { color: ACCENT }, line: { color: ACCENT, width: 0 },
  });
  s.addText(r.title, {
    x: cx + 0.2, y: cy, w: cellW - 0.25, h: 0.45,
    fontFace: "Calibri", fontSize: 14, bold: true, color: INK, margin: 0, valign: "middle",
  });
  s.addText(r.body, {
    x: cx + 0.2, y: cy + 0.45, w: cellW - 0.25, h: cellH - 0.45,
    fontFace: "Calibri", fontSize: 11, color: MUTED, margin: 0, valign: "top",
  });
}

// ============= Slide 8 — Results table =============
s = pres.addSlide();
s.background = { color: BG };

s.addText("Результаты", {
  x: 0.8, y: 0.5, w: 11, h: 0.55,
  fontFace: "Georgia", fontSize: 30, bold: true, color: INK, margin: 0,
});
s.addText("PanNuke fold 3 · единый протокол оценки", {
  x: 0.8, y: 1.15, w: 11.5, h: 0.4,
  fontFace: "Calibri", fontSize: 13, italic: true, color: MUTED, margin: 0,
});

// Table rows
const headerOpts = {
  fontFace: "Calibri", fontSize: 11, bold: true, color: MUTED, charSpacing: 2,
  fill: { color: BG },
  valign: "middle",
};
const rowOpts = {
  fontFace: "Calibri", fontSize: 13, color: INK, valign: "middle",
};
const highlightOpts = {
  fontFace: "Calibri", fontSize: 13, bold: true, color: ACCENT, valign: "middle",
  fill: { color: SOFT },
};

const tableData = [
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
    { text: "CellViT-256 (x20→x40 mismatch)", options: rowOpts },
    { text: "46.8M", options: { ...rowOpts, align: "right" } },
    { text: "0.317", options: { ...rowOpts, align: "right" } },
    { text: "0.471", options: { ...rowOpts, align: "right" } },
    { text: "0.598", options: { ...rowOpts, align: "right" } },
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
    { text: "FastViT-S12 + feature KD (β = 1.0)", options: rowOpts },
    { text: "11.8M", options: { ...rowOpts, align: "right" } },
    { text: "0.461", options: { ...rowOpts, align: "right" } },
    { text: "0.591", options: { ...rowOpts, align: "right" } },
    { text: "0.720", options: { ...rowOpts, align: "right" } },
  ],
  [
    { text: "FastViT-S12 + response KD", options: rowOpts },
    { text: "11.5M", options: { ...rowOpts, align: "right" } },
    { text: "0.467", options: { ...rowOpts, align: "right" } },
    { text: "0.598", options: { ...rowOpts, align: "right" } },
    { text: "0.724", options: { ...rowOpts, align: "right" } },
  ],
  [
    { text: "FastViT-S12 + response KD + TTA  ★", options: highlightOpts },
    { text: "11.5M", options: { ...highlightOpts, align: "right" } },
    { text: "0.472", options: { ...highlightOpts, align: "right" } },
    { text: "0.604", options: { ...highlightOpts, align: "right" } },
    { text: "0.729", options: { ...highlightOpts, align: "right" } },
  ],
];

s.addTable(tableData, {
  x: 0.8, y: 1.7, w: 11.7,
  colW: [5.7, 1.5, 1.5, 1.5, 1.5],
  rowH: 0.45,
  border: { pt: 0, color: BG },
  fontFace: "Calibri",
});

s.addText("★  Best lightweight variant  ·  +0.011 mPQ от KD поверх baseline  ·  +0.005 mPQ от TTA", {
  x: 0.8, y: 6.2, w: 11.7, h: 0.4,
  fontFace: "Calibri", fontSize: 11, italic: true, color: ACCENT, margin: 0,
});
s.addShape(pres.shapes.LINE, {
  x: 0.8, y: 6.15, w: 11.7, h: 0,
  line: { color: RULE, width: 0.5 },
});

// ============= Slide 9 — Headline result =============
s = pres.addSlide();
s.background = { color: INK };  // dark slide for emphasis

s.addText("ГЛАВНЫЙ РЕЗУЛЬТАТ", {
  x: 0.8, y: 0.8, w: 11.7, h: 0.4,
  fontFace: "Calibri", fontSize: 11, bold: true, color: "888888", charSpacing: 6,
  align: "center", margin: 0,
});

// Two huge stats
s.addText("80 %", {
  x: 0.5, y: 1.8, w: 6.2, h: 2.8,
  fontFace: "Georgia", fontSize: 180, bold: true, color: "FFFFFF",
  align: "center", margin: 0,
});
s.addText("качества teacher'а", {
  x: 0.5, y: 4.6, w: 6.2, h: 0.5,
  fontFace: "Calibri", fontSize: 16, color: "CCCCCC",
  align: "center", margin: 0,
});

// vertical divider
s.addShape(pres.shapes.LINE, {
  x: 6.65, y: 2.2, w: 0, h: 2.7,
  line: { color: "555555", width: 1 },
});

s.addText("55×", {
  x: 6.8, y: 1.8, w: 6.2, h: 2.8,
  fontFace: "Georgia", fontSize: 180, bold: true, color: ACCENT,
  align: "center", margin: 0,
});
s.addText("меньше параметров", {
  x: 6.8, y: 4.6, w: 6.2, h: 0.5,
  fontFace: "Calibri", fontSize: 16, color: "CCCCCC",
  align: "center", margin: 0,
});

s.addText("0.472 mPQ  /  0.592 mPQ   •   11.5M  /  630M   •   одно fold 3 · один пайплайн · один метрики", {
  x: 0.5, y: 6.4, w: 12.3, h: 0.4,
  fontFace: "Calibri", fontSize: 12, italic: true, color: "999999",
  align: "center", margin: 0,
});

// ============= Slide 10 — Efficiency =============
s = pres.addSlide();
s.background = { color: BG };

s.addText("Вычислительная эффективность", {
  x: 0.8, y: 0.6, w: 11.5, h: 0.6,
  fontFace: "Georgia", fontSize: 26, bold: true, color: INK, margin: 0,
});
s.addText("RTX 5060 Ti · batch 1 · 256×256", {
  x: 0.8, y: 1.2, w: 11.5, h: 0.4,
  fontFace: "Calibri", fontSize: 12, italic: true, color: MUTED, margin: 0,
});

// Comparison chart: horizontal bars for teacher vs student
const barY = 2.1;
const barH = 0.6;
const barGap = 1.3;

function addBar(y, label, value, maxValue, color, suffix) {
  s.addText(label, {
    x: 0.8, y: y - 0.05, w: 4.0, h: 0.35,
    fontFace: "Calibri", fontSize: 12, color: INK, margin: 0,
  });
  s.addText(value + suffix, {
    x: 0.8, y: y + 0.35, w: 4.0, h: 0.35,
    fontFace: "Calibri", fontSize: 18, bold: true, color: color, margin: 0,
  });
  const totalBarW = 6.8;
  const w = Math.max(0.15, totalBarW * (value / maxValue));
  s.addShape(pres.shapes.RECTANGLE, {
    x: 5.2, y: y + 0.2, w: totalBarW, h: 0.3,
    fill: { color: "EEEEEE" }, line: { color: "EEEEEE", width: 0 },
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: 5.2, y: y + 0.2, w: w, h: 0.3,
    fill: { color }, line: { color, width: 0 },
  });
}

// Metric 1: Params
addBar(barY + barGap * 0, "TEACHER · параметров (млн)", 630, 630, MUTED, " M");
addBar(barY + barGap * 0 + 0.75, "STUDENT · параметров (млн)", 11.5, 630, ACCENT, " M");

// Metric 2: Inference ms
addBar(barY + barGap * 1 + 0.2, "TEACHER · inference (мс/патч)", 500, 500, MUTED, " ms");
addBar(barY + barGap * 1 + 0.95, "STUDENT · inference (мс/патч)", 70, 500, ACCENT, " ms");

// Metric 3: VRAM
addBar(barY + barGap * 2 + 0.4, "TEACHER · VRAM (ГБ)", 24, 24, MUTED, " ГБ");
addBar(barY + barGap * 2 + 1.15, "STUDENT · VRAM (ГБ)", 2, 24, ACCENT, " ГБ");

s.addText("RTX 5060 Ti (16 ГБ) тянет ученика со свободной памятью; teacher требует A100/H100", {
  x: 0.8, y: 6.3, w: 11.7, h: 0.4,
  fontFace: "Calibri", fontSize: 11, italic: true, color: MUTED, margin: 0,
});

// ============= Slide 11 — Per-class =============
s = pres.addSlide();
s.background = { color: BG };

s.addText("Per-class PQ", {
  x: 0.8, y: 0.6, w: 11.5, h: 0.6,
  fontFace: "Georgia", fontSize: 30, bold: true, color: INK, margin: 0,
});
s.addText("Редкий класс Dead — единственная точка где teacher сильно превосходит", {
  x: 0.8, y: 1.25, w: 11.5, h: 0.4,
  fontFace: "Calibri", fontSize: 13, italic: true, color: MUTED, margin: 0,
});

// Grouped bar chart
const classNames = ["Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial"];
const teacherData  = [0.668, 0.576, 0.516, 0.443, 0.670];
const studentData  = [0.550, 0.446, 0.384, 0.104, 0.544];

s.addChart(pres.charts.BAR, [
  { name: "Teacher (CellViT-SAM-H)", labels: classNames, values: teacherData },
  { name: "Student (FastViT + KD)",  labels: classNames, values: studentData },
], {
  x: 0.8, y: 1.9, w: 11.7, h: 4.3, barDir: "col",
  chartColors: [MUTED, ACCENT],
  chartArea: { fill: { color: BG }, roundedCorners: false },
  catAxisLabelColor: INK, catAxisLabelFontFace: "Calibri", catAxisLabelFontSize: 12,
  valAxisLabelColor: MUTED, valAxisLabelFontFace: "Calibri", valAxisLabelFontSize: 10,
  valGridLine: { color: "E8E8E8", size: 0.5 },
  catGridLine: { style: "none" },
  showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 9, dataLabelColor: INK,
  dataLabelFormatCode: "0.00",
  showLegend: true, legendPos: "b", legendFontSize: 11, legendFontFace: "Calibri", legendColor: INK,
  valAxisMaxVal: 0.9, valAxisMinVal: 0,
});

s.addText("Gap особенно велик на Dead (<2% патчей). Student'у не хватает ёмкости запомнить редкий класс.", {
  x: 0.8, y: 6.4, w: 11.7, h: 0.4,
  fontFace: "Calibri", fontSize: 11, italic: true, color: MUTED, margin: 0,
});

// ============= Slide 12 — Conclusions + future =============
s = pres.addSlide();
s.background = { color: BG };

s.addText("Итоги", {
  x: 0.8, y: 0.6, w: 11.5, h: 0.6,
  fontFace: "Georgia", fontSize: 32, bold: true, color: INK, margin: 0,
});

// Three key takeaways — left column
s.addText("Что сделано", {
  x: 0.8, y: 1.5, w: 5.7, h: 0.35,
  fontFace: "Calibri", fontSize: 11, bold: true, color: MUTED, charSpacing: 4, margin: 0,
});

const done = [
  { n: "01", text: "Lightweight student FastViT-S12 достигает 0.472 mPQ — 80% teacher'а при 55× компрессии" },
  { n: "02", text: "Исправлен критический баг: пространственная синхронизация soft targets в KD для dense prediction" },
  { n: "03", text: "Исправлен протокол mPQ под стандарт CellViT/NuLite; опубликованные числа теперь сопоставимы" },
];
for (let i = 0; i < done.length; i++) {
  const d = done[i];
  const y = 1.95 + i * 1.25;
  s.addText(d.n, {
    x: 0.8, y: y, w: 0.8, h: 0.5,
    fontFace: "Georgia", fontSize: 26, bold: true, color: ACCENT, margin: 0,
  });
  s.addText(d.text, {
    x: 1.7, y: y + 0.05, w: 4.8, h: 1.05,
    fontFace: "Calibri", fontSize: 13, color: INK, margin: 0,
  });
}

// Vertical divider
s.addShape(pres.shapes.LINE, {
  x: 6.7, y: 1.5, w: 0, h: 5.0,
  line: { color: RULE, width: 1 },
});

// Right column — Future work
s.addText("Что дальше", {
  x: 7.0, y: 1.5, w: 5.5, h: 0.35,
  fontFace: "Calibri", fontSize: 11, bold: true, color: MUTED, charSpacing: 4, margin: 0,
});

const future = [
  "3-fold cross-validation на PanNuke для публикационных чисел",
  "External eval: MoNuSeg, CoNSeP — проверка domain shift robustness",
  "Upgrade teacher: SAM 3 (2025/2026) или pathology foundation models (UNI 2, Virchow 2)",
  "Тюнинг β для feature-based KD, сейчас β=1.0 даёт чуть хуже response-only",
  "INT8 post-training quantization → дополнительное 4× сжатие для edge deployment",
];

s.addText(future.map((item, i) => ({
  text: item,
  options: { bullet: { code: "25AA" }, breakLine: i < future.length - 1 },
})), {
  x: 7.0, y: 1.95, w: 5.5, h: 5.0,
  fontFace: "Calibri", fontSize: 12, color: INK, paraSpaceAfter: 8, margin: 0, valign: "top",
});

// Footer
s.addShape(pres.shapes.LINE, {
  x: 0.8, y: 6.9, w: 11.7, h: 0,
  line: { color: RULE, width: 0.5 },
});
s.addText("Спасибо за внимание", {
  x: 0.8, y: 7.0, w: 11.7, h: 0.4,
  fontFace: "Georgia", fontSize: 14, italic: true, color: MUTED, align: "center", margin: 0,
});

// Save
pres.writeFile({ fileName: "thesis_defense.pptx" })
  .then(fileName => console.log(`OK: ${fileName}`));
