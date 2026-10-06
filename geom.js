// Pure geometry helpers, shared by the browser app (app.js) and node self-test
// (selftest.mjs). No DOM, no canvas — just math, so it stays testable.

// Scale a MediaPipe detection box {originX, originY, width, height} (in source-image
// pixels) to target-canvas coordinates, expanded by `pad` fraction on each side and
// clamped to the canvas. pad ~0.1 covers the face fully — detector boxes run tight.
export function mapBox(box, imgW, imgH, canvasW, canvasH, pad = 0.1) {
  const sx = canvasW / imgW;
  const sy = canvasH / imgH;
  let x = box.originX * sx;
  let y = box.originY * sy;
  let w = box.width * sx;
  let h = box.height * sy;
  x -= w * pad;
  y -= h * pad;
  w += 2 * w * pad;
  h += 2 * h * pad;
  // clamp to canvas
  if (x < 0) { w += x; x = 0; }
  if (y < 0) { h += y; y = 0; }
  if (x + w > canvasW) w = canvasW - x;
  if (y + h > canvasH) h = canvasH - y;
  return { x: Math.round(x), y: Math.round(y), w: Math.round(w), h: Math.round(h) };
}

// Grow (or shrink) a polygon outward from its centroid by `factor` — the detector's
// face-oval hugs the skin tightly, so a little expansion covers hairline/jaw/chin.
export function expandPolygon(points, factor) {
  const cx = points.reduce((s, p) => s + p.x, 0) / points.length;
  const cy = points.reduce((s, p) => s + p.y, 0) / points.length;
  return points.map((p) => ({ x: Math.round(cx + (p.x - cx) * factor), y: Math.round(cy + (p.y - cy) * factor) }));
}

// Sample an ellipse inscribed in `box` into `n` outline points — used for
// manually-drawn faces so they share the polygon code path with detected ones.
export function ellipsePoints(box, n = 28) {
  const cx = box.x + box.w / 2, cy = box.y + box.h / 2, rx = box.w / 2, ry = box.h / 2;
  const pts = [];
  for (let i = 0; i < n; i++) {
    const a = (i / n) * Math.PI * 2;
    pts.push({ x: Math.round(cx + rx * Math.cos(a)), y: Math.round(cy + ry * Math.sin(a)) });
  }
  return pts;
}

// Aspect ratio as width/height for a "a:b" string under the given orientation.
// 'free' => null (unconstrained). Portrait keeps a:b (taller when a<b); landscape swaps.
export function aspectValue(ratio, orient) {
  if (ratio === 'free') return null;
  const [a, b] = ratio.split(':').map(Number);
  return orient === 'landscape' ? b / a : a / b;
}

// Largest centered crop rect of aspect `ar` (w/h) that fits in imgW×imgH.
// ar null => 90% of the image (free crop starting point).
export function fitCrop(imgW, imgH, ar) {
  if (!ar) {
    const w = Math.round(imgW * 0.9), h = Math.round(imgH * 0.9);
    return { x: Math.round((imgW - w) / 2), y: Math.round((imgH - h) / 2), w, h };
  }
  let w = imgW, h = Math.round(imgW / ar);
  if (h > imgH) { h = imgH; w = Math.round(imgH * ar); }
  return { x: Math.round((imgW - w) / 2), y: Math.round((imgH - h) / 2), w, h };
}

// Shrink the longer side of a w×h rect so it matches aspect `ar` (w/h). Shrinking
// (never growing) keeps the result inside the bounds the caller already clamped to.
export function constrainSize(w, h, ar) {
  if (!ar) return { w: Math.round(w), h: Math.round(h) };
  if (w / h > ar) w = h * ar; else h = w / ar;
  return { w: Math.round(w), h: Math.round(h) };
}

// CSS filter string for the tune sliders (brightness/contrast/saturate/B&W as
// percentages; 100/0 = neutral). Temperature and vignette are applied separately
// (they aren't expressible as a single CSS filter).
export function tuneFilter(a) {
  return `brightness(${a.brightness}%) contrast(${a.contrast}%) saturate(${a.saturate}%) grayscale(${a.bw}%)`;
}

// True when no tune/vignette is active — lets the renderer skip the whole pass.
export function tuneNeutral(a) {
  return a.brightness === 100 && a.contrast === 100 && a.saturate === 100
    && a.temp === 0 && a.bw === 0 && a.vignette === 0;
}

// Resize steps to scale srcW×srcH by `factor`. Downscaling halves repeatedly before
// the final step — a single large bilinear downscale aliases and softens; halving keeps
// edges sharp. Upscaling is one step (smoothing can't add real detail). [] when factor===1.
export function scaleSteps(srcW, srcH, factor) {
  if (factor === 1) return [];
  const tW = Math.max(1, Math.round(srcW * factor)), tH = Math.max(1, Math.round(srcH * factor));
  const steps = [];
  let w = srcW, h = srcH;
  if (factor < 1) {
    while (w * 0.5 > tW) { w = Math.max(tW, Math.round(w * 0.5)); h = Math.max(tH, Math.round(h * 0.5)); steps.push([w, h]); }
  }
  steps.push([tW, tH]);
  return steps;
}

// Tiny-canvas dimensions for a mosaic/pixelate pass: shrink the region by blockSize,
// never below 1px. Larger blockSize => chunkier mosaic.
export function mosaicDims(w, h, blockSize) {
  return {
    w: Math.max(1, Math.round(w / blockSize)),
    h: Math.max(1, Math.round(h / blockSize)),
  };
}
