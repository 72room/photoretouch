import { FilesetResolver, FaceLandmarker } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18';
import { downloadZip } from 'https://cdn.jsdelivr.net/npm/client-zip@2.4.6/index.js';
import { mosaicDims, ellipsePoints, expandPolygon, aspectValue, fitCrop, constrainSize, tuneFilter, tuneNeutral, scaleSteps } from './geom.js';

// ponytail: CDN-loaded model+wasm — vendor locally if offline use is needed
const WASM = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm';
const MODEL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task';
const EMOJIS = [
  '😀', '😎', '🙂', '😍', '😭', '😡', '🤡', '😈', '👽', '👾', '🤖', '👻', '💀',
  '😺', '🐶', '🐱', '🐻', '🐼', '🦊', '🦁', '🐯', '🐸', '🐵', '🐷', '🐰', '🐨',
  '🦄', '🐥', '🦉', '🐙', '🦖', '🌟', '⭐', '🔥', '⚡', '❤️', '🌈', '🎃', '🧸', '🔒',
];
const SVGNS = 'http://www.w3.org/2000/svg';

// --- state -----------------------------------------------------------------
// images: [{ id, name, img:HTMLImageElement, faces:[{ points:[{x,y}], box:{x,y,w,h}, effect }] }]
// points trace the face outline (image px); effect is null | 'blur' | 'mosaic' | 'sticker'
const images = [];
let landmarker = null;
let nextId = 1;
const settings = {
  effect: 'blur',
  blur: 16,
  block: 12,
  feather: 0.3,
  sticker: { type: 'emoji', value: EMOJIS[0] },
  prefix: 'redacted-',
  format: 'png',
  scale: 100,
  ratio: 'free',
  orient: 'portrait',
  // global tune applied to every photo (appearance only — safe with faces/crop)
  adjust: { brightness: 100, contrast: 100, saturate: 100, temp: 0, bw: 0, vignette: 0 },
};

const PRESETS = {
  None:  { brightness: 100, contrast: 100, saturate: 100, temp: 0, bw: 0, vignette: 0 },
  Vivid: { brightness: 103, contrast: 112, saturate: 140, temp: 10, bw: 0, vignette: 10 },
  Warm:  { brightness: 104, contrast: 104, saturate: 112, temp: 45, bw: 0, vignette: 15 },
  Cool:  { brightness: 102, contrast: 106, saturate: 108, temp: -40, bw: 0, vignette: 10 },
  'B&W': { brightness: 102, contrast: 115, saturate: 100, temp: 0, bw: 100, vignette: 20 },
  Noir:  { brightness: 96, contrast: 135, saturate: 100, temp: 0, bw: 100, vignette: 55 },
};
let drawMode = false;
let cropMode = false;
// images[].crop is null or { x, y, w, h } in natural-image px

// --- landmarker ------------------------------------------------------------
async function getLandmarker() {
  if (landmarker) return landmarker;
  const vision = await FilesetResolver.forVisionTasks(WASM);
  landmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: MODEL, delegate: 'GPU' },
    runningMode: 'IMAGE',
    numFaces: 20,
    minFaceDetectionConfidence: 0.3,
    minFacePresenceConfidence: 0.3,
  });
  return landmarker;
}

// Ordered landmark indices tracing the face outline, chained from the model's
// FACE_OVAL connection pairs (each connection's end is the next start).
let ovalIdx = null;
function faceOvalRing() {
  if (ovalIdx) return ovalIdx;
  const conns = FaceLandmarker.FACE_LANDMARKS_FACE_OVAL;
  const next = new Map(conns.map((c) => [c.start, c.end]));
  const out = [];
  let k = conns[0].start;
  for (let i = 0; i < conns.length; i++) { out.push(k); k = next.get(k); if (k === undefined) break; }
  return (ovalIdx = out);
}

function boundsOf(points) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const p of points) { if (p.x < x0) x0 = p.x; if (p.y < y0) y0 = p.y; if (p.x > x1) x1 = p.x; if (p.y > y1) y1 = p.y; }
  return { x: Math.round(x0), y: Math.round(y0), w: Math.round(x1 - x0), h: Math.round(y1 - y0) };
}

// --- effects (canvas 2d only) ----------------------------------------------
function tracePath(ctx, points, ox, oy) {
  ctx.beginPath();
  ctx.moveTo(points[0].x - ox, points[0].y - oy);
  for (let i = 1; i < points.length; i++) ctx.lineTo(points[i].x - ox, points[i].y - oy);
  ctx.closePath();
}

function applyEffect(ctx, img, face) {
  const { box, points, effect } = face;
  const { x, y, w, h } = box;
  if (w <= 0 || h <= 0) return;
  if (effect === 'blur' || effect === 'mosaic') {
    const t = document.createElement('canvas');
    t.width = w; t.height = h;
    const tc = t.getContext('2d');
    if (effect === 'blur') {
      tc.filter = `blur(${settings.blur}px)`;
      tc.drawImage(img, x, y, w, h, 0, 0, w, h);
      tc.filter = 'none';
    } else {
      const { w: mw, h: mh } = mosaicDims(w, h, settings.block);
      const tiny = document.createElement('canvas');
      tiny.width = mw; tiny.height = mh;
      tiny.getContext('2d').drawImage(img, x, y, w, h, 0, 0, mw, mh);
      tc.imageSmoothingEnabled = false;
      tc.drawImage(tiny, 0, 0, mw, mh, 0, 0, w, h);
      tc.imageSmoothingEnabled = true;
    }
    // keep only the face-outline polygon; blurring the mask fill feathers its edge
    const featherPx = settings.feather * 0.5 * Math.min(w, h);
    tc.globalCompositeOperation = 'destination-in';
    if (featherPx > 0.5) tc.filter = `blur(${featherPx}px)`;
    tracePath(tc, points, x, y);
    tc.fillStyle = '#000';
    tc.fill();
    tc.filter = 'none';
    tc.globalCompositeOperation = 'source-over';
    ctx.drawImage(t, x, y);
  } else if (effect === 'sticker') {
    const s = settings.sticker;
    if (s.type === 'png' && s.img) {
      ctx.drawImage(s.img, x, y, w, h);
    } else {
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.font = `${h}px serif`;
      ctx.fillText(s.value, x + w / 2, y + h / 2);
    }
  }
}

// The image with global tune (brightness/contrast/saturation/temperature/B&W/vignette)
// baked in, at natural size. Faces sample from this too, so a blur matches the tone.
// ponytail: rebuilds a full-size canvas per image on every slider tick — cache/debounce
// if it drags with many large photos.
function adjustedSource(rec) {
  const a = settings.adjust;
  if (tuneNeutral(a)) return rec.img;
  const W = rec.img.naturalWidth, H = rec.img.naturalHeight;
  const c = document.createElement('canvas');
  c.width = W; c.height = H;
  const x = c.getContext('2d');
  x.filter = tuneFilter(a);
  x.drawImage(rec.img, 0, 0);
  x.filter = 'none';
  if (a.temp) {
    x.globalCompositeOperation = 'soft-light';
    x.fillStyle = a.temp > 0 ? `rgba(255,150,40,${a.temp / 100 * 0.5})` : `rgba(40,120,255,${-a.temp / 100 * 0.5})`;
    x.fillRect(0, 0, W, H);
    x.globalCompositeOperation = 'source-over';
  }
  if (a.vignette) {
    const g = x.createRadialGradient(W / 2, H / 2, Math.min(W, H) * 0.3, W / 2, H / 2, Math.max(W, H) * 0.7);
    g.addColorStop(0, 'rgba(0,0,0,0)');
    g.addColorStop(1, `rgba(0,0,0,${a.vignette / 100 * 0.85})`);
    x.fillStyle = g;
    x.fillRect(0, 0, W, H);
  }
  return c;
}

function renderImage(rec) {
  const src = adjustedSource(rec);
  const ctx = rec.canvas.getContext('2d');
  ctx.clearRect(0, 0, rec.canvas.width, rec.canvas.height);
  ctx.drawImage(src, 0, 0);
  for (const f of rec.faces) if (f.effect) applyEffect(ctx, src, f);
}

function renderAll() { for (const rec of images) renderImage(rec); }

// --- UI --------------------------------------------------------------------
const grid = document.getElementById('grid');

function buildCard(rec) {
  const card = document.createElement('div');
  card.className = 'card';

  const wrap = document.createElement('div');
  wrap.className = 'wrap';
  rec.canvas = document.createElement('canvas');
  rec.canvas.width = rec.img.naturalWidth;
  rec.canvas.height = rec.img.naturalHeight;
  wrap.appendChild(rec.canvas);

  const overlay = document.createElement('div');
  overlay.className = 'overlay';
  rec.overlay = overlay;
  wrap.appendChild(overlay);

  const cropper = document.createElement('div');
  cropper.className = 'cropper';
  cropper.innerHTML = '<div class="crop-rect">'
    + '<span class="ch nw"></span><span class="ch ne"></span>'
    + '<span class="ch sw"></span><span class="ch se"></span></div>';
  rec.cropper = cropper;
  rec.cropRect = cropper.firstChild;
  wrap.appendChild(cropper);

  const rm = document.createElement('button');
  rm.className = 'remove';
  rm.textContent = '✕';
  rm.title = 'Remove this photo';
  rm.onclick = () => {
    const i = images.indexOf(rec);
    if (i >= 0) images.splice(i, 1);
    card.remove();
    status(images.length ? `${images.length} image(s) loaded.` : 'No images loaded.');
  };
  wrap.appendChild(rm);
  card.appendChild(wrap);

  const bar = document.createElement('div');
  bar.className = 'cardbar';
  const name = document.createElement('span');
  name.textContent = rec.name;
  name.className = 'fname';
  const dims = document.createElement('span');
  dims.className = 'dims';
  rec.dimsEl = dims;
  const selAll = document.createElement('button');
  selAll.textContent = 'All faces';
  selAll.onclick = () => { for (const f of rec.faces) f.effect = settings.effect; refresh(rec); };
  const clear = document.createElement('button');
  clear.textContent = 'Clear';
  clear.onclick = () => { for (const f of rec.faces) f.effect = null; refresh(rec); };
  const dl = document.createElement('button');
  dl.textContent = 'Download';
  dl.onclick = async () => saveBlob(await blobOf(exportCanvas(rec)), outName(rec.name));
  bar.append(name, dims, selAll, clear, dl);
  card.appendChild(bar);

  grid.appendChild(card);
  enableDraw(rec);
  enableCrop(rec);
  drawBoxes(rec);
  positionCrop(rec);
}

function drawBoxes(rec) {
  const { overlay, img } = rec;
  overlay.innerHTML = '';
  const svg = document.createElementNS(SVGNS, 'svg');
  svg.setAttribute('viewBox', `0 0 ${img.naturalWidth} ${img.naturalHeight}`);
  svg.setAttribute('preserveAspectRatio', 'none');
  svg.setAttribute('class', 'reticles');
  rec.faces.forEach((f, i) => {
    const poly = document.createElementNS(SVGNS, 'polygon');
    poly.setAttribute('points', f.points.map((p) => `${p.x},${p.y}`).join(' '));
    poly.setAttribute('class', 'facebox' + (f.effect ? ' on' : ''));
    poly.setAttribute('vector-effect', 'non-scaling-stroke');
    const title = document.createElementNS(SVGNS, 'title');
    title.textContent = (f.effect ? `${f.effect} — click to clear` : 'click to apply') + ' · double-click to delete';
    poly.appendChild(title);
    poly.addEventListener('click', () => { f.effect = f.effect === settings.effect ? null : settings.effect; refresh(rec); });
    poly.addEventListener('dblclick', (ev) => { ev.stopPropagation(); rec.faces.splice(i, 1); refresh(rec); });
    svg.appendChild(poly);
  });
  overlay.appendChild(svg);
}

// Drag on empty overlay space (in Draw mode) to add a face region by hand — for faces
// the detector missed. Coordinates convert from displayed pixels to natural-image pixels.
function enableDraw(rec) {
  const ov = rec.overlay;
  let start = null, preview = null;
  ov.addEventListener('pointerdown', (e) => {
    if (!drawMode || e.target !== ov) return;
    const r = ov.getBoundingClientRect();
    start = { ox: e.clientX - r.left, oy: e.clientY - r.top, r };
    preview = document.createElement('div');
    preview.className = 'draw-preview';
    ov.appendChild(preview);
    ov.setPointerCapture(e.pointerId);
  });
  ov.addEventListener('pointermove', (e) => {
    if (!start) return;
    const { r, ox, oy } = start;
    const cx = e.clientX - r.left, cy = e.clientY - r.top;
    const L = Math.min(cx, ox), T = Math.min(cy, oy), W = Math.abs(cx - ox), H = Math.abs(cy - oy);
    Object.assign(preview.style, {
      left: L / r.width * 100 + '%', top: T / r.height * 100 + '%',
      width: W / r.width * 100 + '%', height: H / r.height * 100 + '%',
    });
  });
  const finish = (e) => {
    if (!start) return;
    const { r, ox, oy } = start;
    const sx = rec.img.naturalWidth / r.width, sy = rec.img.naturalHeight / r.height;
    const cx = e.clientX - r.left, cy = e.clientY - r.top;
    const box = {
      x: Math.round(Math.min(cx, ox) * sx), y: Math.round(Math.min(cy, oy) * sy),
      w: Math.round(Math.abs(cx - ox) * sx), h: Math.round(Math.abs(cy - oy) * sy),
    };
    preview.remove(); start = null;
    if (box.w > 6 && box.h > 6) { rec.faces.push({ box, points: ellipsePoints(box), effect: settings.effect }); refresh(rec); }
  };
  ov.addEventListener('pointerup', finish);
  ov.addEventListener('pointercancel', () => { if (preview) preview.remove(); start = null; });
}

function refresh(rec) { renderImage(rec); drawBoxes(rec); }

// --- crop ------------------------------------------------------------------
// Dimensions the current settings will export this image at: (crop or full) × scale.
function exportSize(rec) {
  const f = Math.min(400, Math.max(10, settings.scale || 100)) / 100;
  const w = rec.crop ? rec.crop.w : rec.img.naturalWidth;
  const h = rec.crop ? rec.crop.h : rec.img.naturalHeight;
  return { w: Math.max(1, Math.round(w * f)), h: Math.max(1, Math.round(h * f)) };
}
function updateDims(rec) {
  if (!rec.dimsEl) return;
  const { w, h } = exportSize(rec);
  rec.dimsEl.textContent = `${w} × ${h} px`;
}
function updateAllDims() { for (const rec of images) updateDims(rec); }

function positionCrop(rec) {
  updateDims(rec);
  rec.cropper.style.display = rec.crop ? 'block' : 'none';
  if (!rec.crop) return;
  const c = rec.crop, W = rec.img.naturalWidth, H = rec.img.naturalHeight;
  Object.assign(rec.cropRect.style, {
    left: c.x / W * 100 + '%', top: c.y / H * 100 + '%',
    width: c.w / W * 100 + '%', height: c.h / H * 100 + '%',
  });
}

// Give every image a crop rect at the current ratio (used when entering crop mode).
function ensureCrops() {
  const ar = aspectValue(settings.ratio, settings.orient);
  for (const rec of images) {
    if (!rec.crop) rec.crop = fitCrop(rec.img.naturalWidth, rec.img.naturalHeight, ar);
    positionCrop(rec);
  }
}

// Re-fit existing crops when the ratio/orientation changes.
function applyRatio() {
  const ar = aspectValue(settings.ratio, settings.orient);
  for (const rec of images) {
    if (rec.crop) rec.crop = fitCrop(rec.img.naturalWidth, rec.img.naturalHeight, ar);
    positionCrop(rec);
  }
}

// Drag the crop body to move it, or a corner handle to resize it (resize honors the
// chosen aspect ratio). All math is in natural-image px; the rect renders in %.
function enableCrop(rec) {
  const el = rec.cropRect;
  let mode = null, start = null, anchor = null, scaleX = 1, scaleY = 1, box = null;
  const toImg = (e) => ({ x: (e.clientX - box.left) * scaleX, y: (e.clientY - box.top) * scaleY });
  const begin = (e, m) => {
    if (!cropMode || !rec.crop) return;
    box = rec.cropper.getBoundingClientRect();
    scaleX = rec.img.naturalWidth / box.width;
    scaleY = rec.img.naturalHeight / box.height;
    mode = m;
    const p = toImg(e);
    start = { ...rec.crop, px: p.x, py: p.y };
    if (m !== 'move') anchor = {
      x: m.includes('e') ? rec.crop.x : rec.crop.x + rec.crop.w,
      y: m.includes('s') ? rec.crop.y : rec.crop.y + rec.crop.h,
    };
    el.setPointerCapture(e.pointerId);
    e.preventDefault(); e.stopPropagation();
  };
  el.addEventListener('pointerdown', (e) => begin(e, 'move'));
  el.querySelectorAll('.ch').forEach((h) => {
    const m = [...h.classList].find((c) => /^[ns][ew]$/.test(c));
    h.addEventListener('pointerdown', (e) => begin(e, m));
  });
  el.addEventListener('pointermove', (e) => {
    if (!mode) return;
    const W = rec.img.naturalWidth, H = rec.img.naturalHeight;
    const p = toImg(e);
    p.x = Math.max(0, Math.min(W, p.x));
    p.y = Math.max(0, Math.min(H, p.y));
    if (mode === 'move') {
      rec.crop.x = Math.round(Math.max(0, Math.min(W - rec.crop.w, start.x + (p.x - start.px))));
      rec.crop.y = Math.round(Math.max(0, Math.min(H - rec.crop.h, start.y + (p.y - start.py))));
    } else {
      let { w, h } = constrainSize(Math.abs(p.x - anchor.x), Math.abs(p.y - anchor.y),
        aspectValue(settings.ratio, settings.orient));
      if (w >= 8 && h >= 8) rec.crop = {
        x: Math.round(mode.includes('e') ? anchor.x : anchor.x - w),
        y: Math.round(mode.includes('s') ? anchor.y : anchor.y - h),
        w, h,
      };
    }
    positionCrop(rec);
  });
  const end = () => { mode = null; };
  el.addEventListener('pointerup', end);
  el.addEventListener('pointercancel', end);
}

// --- loading & detection ---------------------------------------------------
async function addFiles(fileList) {
  const files = [...fileList].filter((f) => f.type.startsWith('image/'));
  if (!files.length) return;
  for (const file of files) {
    const img = await loadImage(URL.createObjectURL(file));
    const rec = { id: nextId++, name: file.name, img, faces: [] };
    images.push(rec);
    buildCard(rec);
    renderImage(rec);
  }
  status(`${images.length} image(s) loaded — click “Detect faces” or draw faces manually.`);
}

// Run detection on every loaded image (replaces existing face boxes). On-demand so the
// model only loads when asked.
async function detectAll() {
  if (!images.length) return;
  status('Detecting faces…');
  const fl = await getLandmarker();
  const idx = faceOvalRing();
  let total = 0;
  for (const rec of images) {
    const W = rec.img.naturalWidth, H = rec.img.naturalHeight;
    const res = fl.detect(rec.img);
    rec.faces = (res.faceLandmarks || []).map((lm) => {
      const raw = idx.map((i) => ({ x: lm[i].x * W, y: lm[i].y * H }));
      const points = expandPolygon(raw, 1.25);   // cover hairline/jaw, not just the skin oval
      return { points, box: boundsOf(points), effect: null };
    });
    total += rec.faces.length;
    refresh(rec);
  }
  status(`${total} face(s) detected across ${images.length} image(s).`);
}

// Remove all face boxes (detected + manually drawn) from every image.
function clearFaces() {
  for (const rec of images) { rec.faces = []; refresh(rec); }
  status('Face selections cleared.');
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const im = new Image();
    im.onload = () => resolve(im);
    im.onerror = reject;
    im.src = src;
  });
}

// --- export ----------------------------------------------------------------
const MIME = { png: 'image/png', jpeg: 'image/jpeg' };
const EXT = { png: 'png', jpeg: 'jpg' };
function outName(name) { return settings.prefix + name.replace(/\.[^.]+$/, '') + '.' + EXT[settings.format]; }

// High-quality resample of a canvas by `factor`. Downscales in halving steps so
// edges stay sharp; always uses high smoothing.
function scaleCanvas(src, factor) {
  let cur = src;
  for (const [w, h] of scaleSteps(src.width, src.height, factor)) {
    const c = document.createElement('canvas');
    c.width = w; c.height = h;
    const x = c.getContext('2d');
    x.imageSmoothingEnabled = true;
    x.imageSmoothingQuality = 'high';
    x.drawImage(cur, 0, 0, w, h);
    cur = c;
  }
  return cur;
}

// The canvas to export: the full render (or crop region), scaled to settings.scale%.
function exportCanvas(rec) {
  const factor = Math.min(400, Math.max(10, settings.scale || 100)) / 100;
  let src = rec.canvas;
  if (rec.crop) {
    const c = rec.crop, t = document.createElement('canvas');
    t.width = c.w; t.height = c.h;
    const tc = t.getContext('2d');
    if (settings.format === 'jpeg') { tc.fillStyle = '#fff'; tc.fillRect(0, 0, c.w, c.h); } // jpeg has no alpha
    tc.drawImage(rec.canvas, c.x, c.y, c.w, c.h, 0, 0, c.w, c.h);
    src = t;
  }
  return scaleCanvas(src, factor);
}
function blobOf(canvas) {
  return new Promise((res) => canvas.toBlob(res, MIME[settings.format], settings.format === 'jpeg' ? 0.92 : undefined));
}

function saveBlob(blob, filename) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

async function downloadAll() {
  if (!images.length) return;
  const entries = [];
  for (const rec of images) entries.push({ name: outName(rec.name), input: await blobOf(exportCanvas(rec)) });
  const blob = await downloadZip(entries).blob();
  saveBlob(blob, settings.prefix + 'photos.zip');
}

// --- wiring ----------------------------------------------------------------
const statusEl = document.getElementById('status');
function status(msg) { statusEl.textContent = msg; }

document.getElementById('files').addEventListener('change', (e) => addFiles(e.target.files));

const drop = document.getElementById('drop');
['dragover', 'dragenter'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('hot'); }));
['dragleave', 'drop'].forEach((ev) => drop.addEventListener(ev, () => drop.classList.remove('hot')));
drop.addEventListener('drop', (e) => { e.preventDefault(); addFiles(e.dataTransfer.files); });

document.querySelectorAll('input[name=effect]').forEach((r) =>
  r.addEventListener('change', (e) => { settings.effect = e.target.value; syncControls(); }));

const blurRange = document.getElementById('blur');
blurRange.addEventListener('input', (e) => { settings.blur = +e.target.value; renderAll(); });
const blockRange = document.getElementById('block');
blockRange.addEventListener('input', (e) => { settings.block = +e.target.value; renderAll(); });
document.getElementById('feather').addEventListener('input', (e) => { settings.feather = +e.target.value / 100; renderAll(); });

// tune sliders -> settings.adjust
const TUNE = [['t-bright', 'brightness'], ['t-contrast', 'contrast'], ['t-sat', 'saturate'], ['t-temp', 'temp'], ['t-bw', 'bw'], ['t-vig', 'vignette']];
TUNE.forEach(([id, key]) =>
  document.getElementById(id).addEventListener('input', (e) => {
    settings.adjust[key] = +e.target.value;
    presetRow.querySelectorAll('button.sel').forEach((x) => x.classList.remove('sel'));
    renderAll();
  }));
function syncTune() { for (const [id, key] of TUNE) document.getElementById(id).value = settings.adjust[key]; }

const presetRow = document.getElementById('presets');
Object.entries(PRESETS).forEach(([name, vals]) => {
  const b = document.createElement('button');
  b.textContent = name;
  b.onclick = () => {
    settings.adjust = { ...vals };
    syncTune();
    presetRow.querySelectorAll('button').forEach((x) => x.classList.toggle('sel', x === b));
    renderAll();
  };
  presetRow.appendChild(b);
});

document.getElementById('detect').onclick = detectAll;
document.getElementById('clearFaces').onclick = clearFaces;

const drawBtn = document.getElementById('draw');
const cropBtn = document.getElementById('crop');
function setDraw(on) { drawMode = on; drawBtn.classList.toggle('sel', on); document.body.classList.toggle('drawing', on); }
function setCrop(on) { cropMode = on; cropBtn.classList.toggle('sel', on); document.body.classList.toggle('cropping', on); }
drawBtn.onclick = () => { setDraw(!drawMode); if (drawMode) setCrop(false); };
cropBtn.onclick = () => { setCrop(!cropMode); if (cropMode) { setDraw(false); ensureCrops(); } };

document.getElementById('ratio').addEventListener('change', (e) => { settings.ratio = e.target.value; applyRatio(); });
document.querySelectorAll('input[name=orient]').forEach((r) =>
  r.addEventListener('change', (e) => { settings.orient = e.target.value; applyRatio(); }));
document.getElementById('clearCrop').onclick = () => { for (const rec of images) { rec.crop = null; positionCrop(rec); } };
document.getElementById('format').addEventListener('change', (e) => { settings.format = e.target.value; });
document.getElementById('scale').addEventListener('input', (e) => { settings.scale = +e.target.value || 100; updateAllDims(); });

// sticker pickers
const stickerRow = document.getElementById('stickers');
EMOJIS.forEach((em) => {
  const b = document.createElement('button');
  b.textContent = em;
  b.className = 'emoji';
  b.onclick = () => { settings.sticker = { type: 'emoji', value: em }; markSticker(b); renderAll(); };
  stickerRow.appendChild(b);
});
document.getElementById('pngsticker').addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const img = await loadImage(URL.createObjectURL(file));
  settings.sticker = { type: 'png', img };
  markSticker(null);
  renderAll();
});
function markSticker(btn) {
  document.querySelectorAll('#stickers .emoji').forEach((b) => b.classList.toggle('sel', b === btn));
}

document.getElementById('applyAll').onclick = () => {
  for (const rec of images) { for (const f of rec.faces) f.effect = settings.effect; refresh(rec); }
};
document.getElementById('clearAll').onclick = () => {
  for (const rec of images) { for (const f of rec.faces) f.effect = null; refresh(rec); }
};
document.getElementById('zip').onclick = downloadAll;
document.getElementById('prefix').addEventListener('input', (e) => { settings.prefix = e.target.value; });
document.getElementById('reset').onclick = () => {
  images.length = 0;
  grid.innerHTML = '';
  document.getElementById('files').value = '';   // so the same file re-triggers change
  settings.adjust = { ...PRESETS.None };
  settings.scale = 100;
  document.getElementById('scale').value = 100;
  syncTune();
  presetRow.querySelectorAll('button.sel').forEach((x) => x.classList.remove('sel'));
  status('Reset — no images loaded.');
};

function syncControls() {
  document.getElementById('blurctl').hidden = settings.effect !== 'blur';
  document.getElementById('blockctl').hidden = settings.effect !== 'mosaic';
  document.getElementById('stickerctl').hidden = settings.effect !== 'sticker';
  document.getElementById('featherctl').hidden = settings.effect === 'sticker';
}
syncControls();
markSticker(stickerRow.querySelector('.emoji'));
