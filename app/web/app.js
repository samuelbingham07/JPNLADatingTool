import { env, CLIPVisionModelWithProjection, Tensor } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2/dist/transformers.min.js';
import { preprocess, SIZE } from './preprocess.js';
import { createEngine } from './engine.js';

const BACKBONE = 'Xenova/clip-vit-base-patch32';
const OCR_MAX = 1600;
const MAX_PHOTOS = 8;
const $ = id => document.getElementById(id);

const state = { photos: [], decades: null, tagText: '', typed: false, engine: null, busy: false, runs: 0 };
let model = null, ocr = null, modelLoading = null;

const fileProgress = {};
function onProgress(p) {
  if (p.status === 'progress' && p.file?.endsWith('.onnx')) {
    fileProgress[p.file] = p;
    const loaded = Object.values(fileProgress).reduce((a, x) => a + (x.loaded || 0), 0);
    const total = Object.values(fileProgress).reduce((a, x) => a + (x.total || 0), 0);
    setStatus(`Downloading the model, first visit only: ${(loaded / 1e6).toFixed(0)} / ${(total / 1e6).toFixed(0)} MB`, loaded / total);
  }
}

function setStatus(text, frac) {
  $('status').hidden = !text;
  $('status-text').textContent = text || '';
  $('status-bar').style.width = frac == null ? '0' : `${Math.round(frac * 100)}%`;
  $('status-track').hidden = frac == null;
}

async function loadModel() {
  if (model) return model;
  if (modelLoading) return modelLoading;
  modelLoading = (async () => {
    env.allowLocalModels = false;
    const attempts = [];
    if (navigator.gpu) attempts.push({ device: 'webgpu', dtype: 'fp16' });
    attempts.push({ device: 'wasm', dtype: 'fp16' }, { device: 'wasm', dtype: 'fp32' });
    let lastErr;
    for (const opt of attempts) {
      try {
        model = await CLIPVisionModelWithProjection.from_pretrained(BACKBONE, { ...opt, progress_callback: onProgress });
        console.info('backbone loaded', opt);
        return model;
      } catch (e) { lastErr = e; console.warn('backbone load failed', opt, e); }
    }
    throw lastErr;
  })();
  return modelLoading;
}

async function loadOcr() {
  if (ocr) return ocr;
  const { ort, Ocr } = await import('./ocr.bundle.js');
  ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/';
  ocr = await Ocr.create({ models: {
    detectionPath: 'ocr/ch_PP-OCRv4_det_infer.onnx',
    recognitionPath: 'ocr/ch_PP-OCRv4_rec_infer.onnx',
    dictionaryPath: 'ocr/ppocr_keys_v1.txt',
  } });
  return ocr;
}

async function decode(blob) {
  const bmp = await createImageBitmap(blob, { imageOrientation: 'from-image', colorSpaceConversion: 'none' });
  const c = document.createElement('canvas');
  c.width = bmp.width; c.height = bmp.height;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(bmp, 0, 0);
  const rgba = ctx.getImageData(0, 0, c.width, c.height).data;
  const rgb = new Uint8Array(c.width * c.height * 3);
  for (let i = 0, j = 0; i < rgba.length; i += 4, j += 3) { rgb[j] = rgba[i]; rgb[j + 1] = rgba[i + 1]; rgb[j + 2] = rgba[i + 2]; }
  return { rgb, w: c.width, h: c.height, bmp };
}

function ocrCanvasUrl(bmp) {
  const s = Math.min(1, OCR_MAX / Math.max(bmp.width, bmp.height));
  const c = document.createElement('canvas');
  c.width = Math.round(bmp.width * s); c.height = Math.round(bmp.height * s);
  c.getContext('2d').drawImage(bmp, 0, 0, c.width, c.height);
  return new Promise(res => c.toBlob(b => res(URL.createObjectURL(b)), 'image/jpeg', 0.92));
}

async function embedAll(decoded) {
  const m = await loadModel();
  const embeds = [];
  for (const d of decoded) {
    const px = preprocess(d.rgb, d.w, d.h);
    const { image_embeds } = await m({ pixel_values: new Tensor('float32', px, [1, 3, SIZE, SIZE]) });
    const e = Float32Array.from(image_embeds.data);
    const n = Math.hypot(...e);
    embeds.push(e.map(v => v / n));
  }
  return embeds;
}

async function readTags(decoded) {
  const o = await loadOcr();
  const lines = [];
  for (let i = 0; i < decoded.length; i++) {
    setStatus(`Reading tag text (photo ${i + 1} of ${decoded.length})…`, i / decoded.length);
    const url = await ocrCanvasUrl(decoded[i].bmp);
    try {
      const { texts } = await o.detect(url);
      for (const t of texts) if (t.mean > 0.5) lines.push(t.text);
    } finally { URL.revokeObjectURL(url); }
  }
  return lines.join('\n');
}

function observations() {
  return [...document.querySelectorAll('#obs input:checked')].map(x => x.value);
}

function rerender() {
  if (!state.decades) return;
  const r = state.engine.dateFromDecades(state.decades, state.tagText, observations());
  render(r);
}

async function run({ presetText = null } = {}) {
  if (!state.photos.length || state.busy) return;
  state.busy = true;
  $('go').disabled = true;
  try {
    setStatus('Preparing photos…', null);
    const decoded = await Promise.all(state.photos.map(p => decode(p.blob)));
    if (!model) setStatus('Loading the model…', null);
    const embeds = await embedAll(decoded);
    setStatus('Scoring…', null);
    state.decades = state.engine.headProbs(embeds);
    if (presetText !== null) { state.tagText = presetText; state.typed = true; }
    else if (!state.typed) state.tagText = '';
    $('tag-text').value = state.tagText;
    rerender();
    if (presetText === null && !state.typed) {
      $('result').classList.add('refining');
      state.tagText = await readTags(decoded);
      $('tag-text').value = state.tagText;
      $('result').classList.remove('refining');
      rerender();
    }
    decoded.forEach(d => d.bmp.close());
    setStatus('');
  } catch (e) {
    console.error(e);
    setStatus(`Something went wrong: ${e.message || e}. Try a different photo or refresh.`, null);
  } finally {
    state.busy = false;
    $('go').disabled = false;
    $('result').dataset.runs = String(++state.runs);
  }
}

function setPhotos(list) {
  state.photos = list.slice(0, MAX_PHOTOS);
  state.decades = null;
  state.typed = false;
  state.tagText = '';
  $('tag-text').value = '';
  const g = $('thumbs');
  g.innerHTML = '';
  for (const p of state.photos) {
    const img = document.createElement('img');
    img.src = p.url; img.alt = p.name;
    g.appendChild(img);
  }
  g.hidden = !state.photos.length;
  $('go').disabled = !state.photos.length;
  $('drop-hint').textContent = state.photos.length ? `${state.photos.length} photo${state.photos.length > 1 ? 's' : ''} ready` : 'Drop photos here or click to choose';
}

// ---------- rendering ----------
const esc = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const pct = v => `${Math.round(v * 100)}%`;
const KIND = { tell: 'Clear tell', early: 'Early decade', mid: 'Mid decade', late: 'Late decade', full: 'Full decade', low: 'Wide range' };

function signalHtml(s) {
  const kind = s.hard ? 'Hard bound' : 'Soft anchor';
  const src = s.source === 'tag text' ? 'from tag text' : 'you observed';
  const ev = s.source === 'tag text' ? ` — <i>“${esc(s.evidence)}”</i>` : '';
  const flag = s.decisive ? ' · <b>deciding factor</b>' : '';
  return `<div class="sig"><div class="sig-head"><span>${esc(s.label)}</span><span class="sig-span">${esc(s.span)}</span></div>
    <div class="sig-note">${kind}, ${s.confidence} confidence, ${src}${ev}${flag}</div>
    <div class="sig-note muted">${esc(s.notes)}</div></div>`;
}

function chartSvg(r) {
  const W = 560, H = 150, P = { l: 8, r: 8, t: 10, b: 22 };
  const pts = r.years.map((y, i) => [y, r.posterior[i]]).filter(([y]) => y >= 1925);
  const x0 = pts[0][0], x1 = pts[pts.length - 1][0];
  const ymax = Math.max(...pts.map(p => p[1])) || 1;
  const X = y => P.l + (y - x0) / (x1 - x0) * (W - P.l - P.r);
  const Yv = v => H - P.b - v / ymax * (H - P.t - P.b);
  const line = pts.map(([y, v], i) => `${i ? 'L' : 'M'}${X(y).toFixed(1)},${Yv(v).toFixed(1)}`).join('');
  const area = `${line}L${X(x1).toFixed(1)},${H - P.b}L${X(x0).toFixed(1)},${H - P.b}Z`;
  const band = `<rect x="${X(r.start - 0.5)}" y="${P.t}" width="${Math.max(2, X(r.end + 0.5) - X(r.start - 0.5))}" height="${H - P.t - P.b}" class="band"/>`;
  const cutoff = `<line x1="${X(2005.5)}" x2="${X(2005.5)}" y1="${P.t}" y2="${H - P.b}" class="cutoff"/><text x="${X(2005.5) + 4}" y="${P.t + 10}" class="cutoff-label">2006</text>`;
  const ticks = [];
  for (let y = 1930; y <= 2020; y += 10) ticks.push(`<text x="${X(y)}" y="${H - 6}" text-anchor="middle" class="tick">${y}s</text>`);
  return `<svg viewBox="0 0 ${W} ${H}" class="chart" role="img" aria-label="Probability by year">${band}<path d="${area}" class="area"/><path d="${line}" class="line"/>${cutoff}${ticks.join('')}</svg>`;
}

function render(r) {
  const decided = r.decided_by === 'rules'
    ? '<span class="pill rules">Decided by knowledge-base rules</span>'
    : '<span class="pill model">Decided by visual model</span>';
  const v = r.vintage_prob;
  const verdict = v >= 0.8 ? 'Vintage' : v <= 0.2 ? 'Not vintage' : 'Borderline vintage';
  const decs = Object.keys(r.model.decades).map(Number);
  const top = r.model.top;
  const topLabel = `${top === Math.min(...decs) ? '≤' : ''}${top}s · ${pct(r.model.decades[top])}`;
  const sigs = r.signals.slice(0, 3).map(signalHtml).join('');
  const more = r.signals.length > 3 ? `<details class="more"><summary>${r.signals.length - 3} more signal(s)</summary>${r.signals.slice(3).map(signalHtml).join('')}</details>` : '';
  const dropped = r.dropped.length ? `<div class="sig-note muted">Conflicting rules ignored: ${esc(r.dropped.join(', '))}</div>` : '';
  const ask = r.needs_more ? `<div class="ask"><span class="pill warn">Want it tighter?</span> Add a clear, straight-on photo of the neck tag, the care/content tag (often a side seam), and any RN/WPL or union label.</div>` : '';
  const bars = decs.map(d => `<div class="bar-row"><span>${d === Math.min(...decs) ? '≤' : ''}${d}s</span><div class="bar"><div style="width:${(r.model.decades[d] * 100).toFixed(1)}%"></div></div><span>${pct(r.model.decades[d])}</span></div>`).join('');
  $('result').innerHTML = `
    <div class="tagline">Estimated production date · ${KIND[r.kind]}</div>
    <div class="range">${esc(r.range)}</div>
    <div class="meta">${decided}<span class="pill">Confidence ${pct(r.confidence)}</span><span class="pill ${v >= 0.8 ? 'vintage' : ''}">${verdict} (pre-2006): ${pct(v)}</span></div>
    <div class="conf-track"><div class="conf-fill" style="width:${pct(r.confidence)}"></div></div>
    ${ask}
    <div class="refine-note">Reading tag text… the estimate may tighten in a moment.</div>
    <div class="tagline gap">What drove it</div>
    ${sigs}
    <div class="sig"><div class="sig-head"><span>Visual model</span><span class="sig-span">${topLabel}</span></div>
      <div class="sig-note">Trained decade classifier over your photos. On its own it would say ${esc(r.model.range)}.</div></div>
    ${more}${dropped}
    <div class="tagline gap">Probability by year, after rules</div>
    ${chartSvg(r)}
    <details class="more"><summary>Visual model by decade</summary><div class="bars">${bars}</div></details>`;
}

// ---------- boot ----------
async function boot() {
  const [cfg, meta, bin, metrics, examples] = await Promise.all([
    fetch('engine.json').then(r => r.json()), fetch('head.json').then(r => r.json()),
    fetch('head.bin').then(r => r.arrayBuffer()), fetch('metrics.json').then(r => r.json()),
    fetch('examples/examples.json').then(r => r.json()),
  ]);
  state.engine = createEngine(cfg, { meta, data: new Float32Array(bin) });

  $('obs').innerHTML = cfg.observations.map((o, i) =>
    `<label><input type="checkbox" value="${esc(o.label)}" id="obs${i}"> ${esc(o.label)}</label>`).join('');
  $('obs').addEventListener('change', rerender);

  const t = metrics.test;
  $('m-acc').textContent = pct(t.accuracy);
  $('m-base').textContent = pct(t.majority_baseline_accuracy);
  $('m-one').textContent = pct(t.within_one_decade);
  $('m-n').textContent = t.listings.toLocaleString();
  $('m-train').textContent = metrics.train_listings.toLocaleString();
  $('m-imgs').textContent = metrics.train_images.toLocaleString();
  $('m-recall').textContent = Object.entries(t.per_decade_recall).map(([k, v]) => `${k} ${pct(v)}`).join(' · ');

  $('examples').innerHTML = examples.map((e, i) =>
    `<button class="example" data-i="${i}"><img src="${e.files[0]}" alt=""><span>${esc(e.label)}</span></button>`).join('');
  $('examples').addEventListener('click', async ev => {
    const b = ev.target.closest('.example'); if (!b || state.busy) return;
    const e = examples[+b.dataset.i];
    const photos = await Promise.all(e.files.map(async f => {
      const blob = await (await fetch(f)).blob();
      return { name: f, blob, url: URL.createObjectURL(blob) };
    }));
    document.querySelectorAll('#obs input').forEach(x => { x.checked = false; });
    setPhotos(photos);
    run({ presetText: e.tag_text ?? null });
  });

  const input = $('files'), drop = $('drop');
  const take = files => setPhotos([...files].filter(f => f.type.startsWith('image/')).map(f => ({ name: f.name, blob: f, url: URL.createObjectURL(f) })));
  input.addEventListener('change', () => take(input.files));
  drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('over'); });
  drop.addEventListener('dragleave', () => drop.classList.remove('over'));
  drop.addEventListener('drop', e => { e.preventDefault(); drop.classList.remove('over'); take(e.dataTransfer.files); });
  $('go').addEventListener('click', () => run());
  $('tag-text').addEventListener('input', () => { state.typed = true; });
  $('apply-text').addEventListener('click', () => { state.tagText = $('tag-text').value; state.typed = true; rerender(); });

  $('ready').hidden = false;
  // Warm the model in the background so the first click is quick.
  (window.requestIdleCallback || setTimeout)(() => loadModel().then(() => setStatus('')).catch(() => {}));
}

boot().catch(e => { console.error(e); setStatus(`Failed to start: ${e.message || e}`, null); });
