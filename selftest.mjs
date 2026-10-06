// Runnable check for the pure math. `node selftest.mjs` — exits non-zero on failure.
import assert from 'node:assert/strict';
import { mapBox, mosaicDims, ellipsePoints, expandPolygon, aspectValue, fitCrop, constrainSize, tuneFilter, tuneNeutral, scaleSteps } from './geom.js';

// identity scale, no padding => box unchanged
assert.deepEqual(
  mapBox({ originX: 10, originY: 20, width: 30, height: 40 }, 100, 100, 100, 100, 0),
  { x: 10, y: 20, w: 30, h: 40 },
);

// 2x scale
assert.deepEqual(
  mapBox({ originX: 10, originY: 20, width: 30, height: 40 }, 100, 100, 200, 200, 0),
  { x: 20, y: 40, w: 60, h: 80 },
);

// padding expands by 10% each side (w 30 -> 36, origin shifts left 3)
assert.deepEqual(
  mapBox({ originX: 10, originY: 10, width: 30, height: 30 }, 100, 100, 100, 100, 0.1),
  { x: 7, y: 7, w: 36, h: 36 },
);

// clamp: a box at the top-left corner with padding must not go negative
const clamped = mapBox({ originX: 0, originY: 0, width: 20, height: 20 }, 100, 100, 100, 100, 0.5);
assert.equal(clamped.x, 0);
assert.equal(clamped.y, 0);

// clamp: a box at the bottom-right must not overflow the canvas
const br = mapBox({ originX: 90, originY: 90, width: 20, height: 20 }, 100, 100, 100, 100, 0.2);
assert.ok(br.x + br.w <= 100, 'right edge clamped');
assert.ok(br.y + br.h <= 100, 'bottom edge clamped');

// mosaic dims shrink by block size, floor at 1
assert.deepEqual(mosaicDims(100, 50, 10), { w: 10, h: 5 });
assert.deepEqual(mosaicDims(8, 8, 100), { w: 1, h: 1 });

// ellipse outline: n points, right-most point first (angle 0), centered on the box
const ep = ellipsePoints({ x: 0, y: 0, w: 100, h: 60 }, 28);
assert.equal(ep.length, 28);
assert.deepEqual(ep[0], { x: 100, y: 30 });          // cx+rx, cy
assert.deepEqual(ep[7], { x: 50, y: 60 });           // quarter turn: bottom-center

// expandPolygon: factor 1 keeps points; factor 2 doubles distance from centroid
const sq = [{ x: 0, y: 0 }, { x: 10, y: 0 }, { x: 10, y: 10 }, { x: 0, y: 10 }]; // centroid (5,5)
assert.deepEqual(expandPolygon(sq, 1), sq);
assert.deepEqual(expandPolygon(sq, 2), [{ x: -5, y: -5 }, { x: 15, y: -5 }, { x: 15, y: 15 }, { x: -5, y: 15 }]);

// aspectValue: free is unconstrained; portrait keeps a:b, landscape swaps
assert.equal(aspectValue('free', 'portrait'), null);
assert.equal(aspectValue('3:4', 'portrait'), 3 / 4);
assert.equal(aspectValue('3:4', 'landscape'), 4 / 3);
assert.equal(aspectValue('1:1', 'landscape'), 1);

// fitCrop: centered, largest rect of the ratio fitting the image
assert.deepEqual(fitCrop(100, 100, 1), { x: 0, y: 0, w: 100, h: 100 });      // square in square
assert.deepEqual(fitCrop(200, 100, 1), { x: 50, y: 0, w: 100, h: 100 });     // square centered in wide
assert.deepEqual(fitCrop(100, 200, 3 / 4), { x: 0, y: 34, w: 100, h: 133 }); // 3:4 width-bound
const free = fitCrop(100, 100, null);
assert.deepEqual(free, { x: 5, y: 5, w: 90, h: 90 });

// constrainSize: free passes through; otherwise shrink the longer side to the ratio
assert.deepEqual(constrainSize(80, 40, null), { w: 80, h: 40 });
assert.deepEqual(constrainSize(80, 40, 1), { w: 40, h: 40 });   // too wide -> width shrinks
assert.deepEqual(constrainSize(40, 80, 1), { w: 40, h: 40 });   // too tall -> height shrinks

// tuneNeutral: true only when every knob sits at its neutral value
const neutral = { brightness: 100, contrast: 100, saturate: 100, temp: 0, bw: 0, vignette: 0 };
assert.equal(tuneNeutral(neutral), true);
assert.equal(tuneNeutral({ ...neutral, vignette: 5 }), false);
assert.equal(tuneNeutral({ ...neutral, temp: -1 }), false);
assert.equal(
  tuneFilter({ brightness: 110, contrast: 100, saturate: 120, bw: 0 }),
  'brightness(110%) contrast(100%) saturate(120%) grayscale(0%)',
);

// scaleSteps: no-op at 100%; single step up; halving steps down to the target
assert.deepEqual(scaleSteps(1000, 1000, 1), []);
assert.deepEqual(scaleSteps(1000, 800, 2), [[2000, 1600]]);           // upscale: one step
assert.deepEqual(scaleSteps(1000, 1000, 0.5), [[500, 500]]);         // exactly half: one step
assert.deepEqual(scaleSteps(1000, 1000, 0.25), [[500, 500], [250, 250]]); // halve, then final
assert.deepEqual(scaleSteps(1000, 1000, 0.1), [[500, 500], [250, 250], [125, 125], [100, 100]]);
assert.ok(scaleSteps(1000, 1000, 0.1).at(-1)[0] === 100, 'ends on the exact target width');

console.log('selftest ok');
