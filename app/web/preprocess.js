// Reproduces model/backbone.py preprocessing: PIL JPEG draft (DCT downscale ~ box filter),
// PIL antialiased bicubic resize of shortest side to 224, center crop, CLIP normalization.
export const SIZE = 224;
const MEAN = [0.48145466, 0.4578275, 0.40821073];
const STD = [0.26862954, 0.26130258, 0.27577711];

function draft(rgb, w, h) {
  const req = SIZE * 2;
  const scale = Math.min(Math.floor(w / req), Math.floor(h / req));
  let a = 1;
  for (const s of [8, 4, 2, 1]) { if (scale >= s) { a = s; break; } }
  if (a === 1) return { rgb, w, h };
  const nw = Math.ceil(w / a), nh = Math.ceil(h / a);
  const out = new Float32Array(nw * nh * 3);
  for (let y = 0; y < nh; y++) {
    for (let x = 0; x < nw; x++) {
      let r = 0, g = 0, b = 0, n = 0;
      for (let dy = 0; dy < a; dy++) {
        const sy = y * a + dy; if (sy >= h) break;
        for (let dx = 0; dx < a; dx++) {
          const sx = x * a + dx; if (sx >= w) break;
          const i = (sy * w + sx) * 3;
          r += rgb[i]; g += rgb[i + 1]; b += rgb[i + 2]; n++;
        }
      }
      const o = (y * nw + x) * 3;
      out[o] = Math.round(r / n); out[o + 1] = Math.round(g / n); out[o + 2] = Math.round(b / n);
    }
  }
  return { rgb: out, w: nw, h: nh };
}

function cubic(x) {
  const a = -0.5;
  x = Math.abs(x);
  if (x < 1) return ((a + 2) * x - (a + 3)) * x * x + 1;
  if (x < 2) return (((x - 5) * x + 8) * x - 4) * a;
  return 0;
}

function coeffs(inSize, outSize) {
  const scale = inSize / outSize;
  const fs = Math.max(scale, 1);
  const support = 2 * fs;
  const res = [];
  for (let i = 0; i < outSize; i++) {
    const center = (i + 0.5) * scale;
    const xmin = Math.max(Math.trunc(center - support + 0.5), 0);
    const xmax = Math.min(Math.trunc(center + support + 0.5), inSize);
    const w = [];
    let sum = 0;
    for (let x = xmin; x < xmax; x++) { const v = cubic((x - center + 0.5) / fs); w.push(v); sum += v; }
    res.push({ xmin, w: w.map(v => v / sum) });
  }
  return res;
}

const clamp8 = v => Math.min(255, Math.max(0, Math.round(v)));

function resize(rgb, w, h, nw, nh) {
  const cx = coeffs(w, nw), cy = coeffs(h, nh);
  const tmp = new Float32Array(nw * h * 3);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < nw; x++) {
      const { xmin, w: k } = cx[x];
      let r = 0, g = 0, b = 0;
      for (let j = 0; j < k.length; j++) {
        const i = (y * w + xmin + j) * 3;
        r += rgb[i] * k[j]; g += rgb[i + 1] * k[j]; b += rgb[i + 2] * k[j];
      }
      const o = (y * nw + x) * 3;
      tmp[o] = clamp8(r); tmp[o + 1] = clamp8(g); tmp[o + 2] = clamp8(b);
    }
  }
  const out = new Float32Array(nw * nh * 3);
  for (let y = 0; y < nh; y++) {
    const { xmin, w: k } = cy[y];
    for (let x = 0; x < nw; x++) {
      let r = 0, g = 0, b = 0;
      for (let j = 0; j < k.length; j++) {
        const i = ((xmin + j) * nw + x) * 3;
        r += tmp[i] * k[j]; g += tmp[i + 1] * k[j]; b += tmp[i + 2] * k[j];
      }
      const o = (y * nw + x) * 3;
      out[o] = clamp8(r); out[o + 1] = clamp8(g); out[o + 2] = clamp8(b);
    }
  }
  return out;
}

// rgb: Uint8Array/Uint8ClampedArray of length w*h*3 → Float32Array [3, 224, 224]
export function preprocess(rgb, w, h) {
  ({ rgb, w, h } = draft(rgb, w, h));
  const s = SIZE / Math.min(w, h);
  const nw = Math.max(SIZE, Math.round(w * s)), nh = Math.max(SIZE, Math.round(h * s));
  const img = resize(rgb, w, h, nw, nh);
  const left = Math.floor((nw - SIZE) / 2), top = Math.floor((nh - SIZE) / 2);
  const out = new Float32Array(3 * SIZE * SIZE);
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      const i = ((y + top) * nw + (x + left)) * 3;
      for (let c = 0; c < 3; c++) out[c * SIZE * SIZE + y * SIZE + x] = (img[i + c] / 255 - MEAN[c]) / STD[c];
    }
  }
  return out;
}
