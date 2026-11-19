// JS port of model/constraints.py and the post-model half of model/predict.py.
// Config (patterns, rules, RN table) is exported from Python by app/build_web.py.

export function createEngine(cfg, head) {
  const Y0 = cfg.years[0], Y1 = cfg.years[1];
  const YEARS = Array.from({ length: Y1 - Y0 + 1 }, (_, i) => Y0 + i);
  const N = YEARS.length;
  const re = p => new RegExp(p);

  const isHard = s => s.rule.type === 'hard_bound';
  const span = s => isHard(s) ? [s.rule.earliest ?? null, s.rule.latest ?? null]
                              : [s.rule.likely_start ?? null, s.rule.likely_end ?? null];
  function describeSpan(s) {
    const [lo, hi] = span(s);
    if (lo && hi) return `${lo}–${hi}`;
    return lo ? `no earlier than ${lo}` : `no later than ${hi}`;
  }

  function rnLookup(rn) {
    const c = cfg.rn.cache[String(rn)];
    if (c) return { earliest_year: c.earliest_year, company: c.company, note: '' };
    if (rn < 101) return { earliest_year: null };
    if (rn >= cfg.rn.gap[0] && rn <= cfg.rn.gap[1]) return { earliest_year: null };
    for (const [lo, hi, ys, ye] of cfg.rn.ranges) {
      if (rn >= lo && rn < hi) return { earliest_year: ys, note: `Estimated from RN range ${lo}–${hi - 1} ≈ ${ys}–${ye}` };
    }
    return { earliest_year: null };
  }

  function extractSignals(text, observations = []) {
    const t = ' ' + text.toLowerCase().replace(/\s+/g, ' ') + ' ';
    let out = [];
    const add = (key, label, rule, evidence, source = 'tag text') => {
      if (rule.type === 'hard_bound' && rule.earliest == null && rule.latest == null) return;
      if (out.every(s => s.key !== key)) out.push({ key, label, rule, evidence, source });
    };

    for (const c of cfg.countries) {
      const m = re(`(?:ma[do]e|hecho|fabriqu[eé]|manufactured)\\s*(?:in|en)\\s*(?:the\\s+)?(?:${c.pattern})\\b`).exec(t);
      if (m) {
        const ctx = t.slice(Math.max(0, m.index - 40), m.index + m[0].length + 40);
        if (!/assembled|imported\s+(?:fabric|materials)/.test(ctx))
          add(`country:${c.name}`, `Made in ${c.name.replace(/_/g, ' ')}`, c.rule, m[0].trim());
      }
    }
    for (const c of cfg.anywhere_countries) {
      const m = re(c.pattern).exec(t);
      if (m) add(`country:${c.name}`, `${c.name} on label`, c.rule, m[0].trim());
    }

    let m = /\bw\.?\s?p\.?\s?l\.?\s*(?:no\.?|#)?\s*(\d{3,6})/.exec(t);
    if (m && !cfg.wpl_kept_after_1971.includes(parseInt(m[1], 10)))
      add('wpl', `WPL ${m[1]}`, cfg.wpl_rule, m[0].trim());
    m = /\br\.?\s?n\.?\s*(?:no\.?|#)?\s*(\d{3,6})\b/.exec(t);
    if (m) {
      const rn = parseInt(m[1], 10);
      const info = rnLookup(rn);
      if (info.earliest_year) {
        const who = info.company ? ` (${info.company})` : '';
        const rule = { type: 'hard_bound', earliest: info.earliest_year, confidence: 'high',
          notes: `RN ${rn}${who} registered ~${info.earliest_year}; a garment can't predate its RN. ${info.note || ''}`.trim() };
        add('rn', `RN ${rn}${who}`, rule, m[0].trim());
      } else {
        add('rn', `RN ${rn}`, cfg.rn_present_rule, m[0].trim());
      }
    }

    for (const u of cfg.unions) {
      const um = re(u.pattern).exec(t);
      if (um) add(`union:${u.name}`, `${u.name} union label`, u.rule, um[0].trim());
    }
    for (const r of cfg.text_rules) {
      const rm = re(r.pattern).exec(t);
      if (rm) add(r.key, r.label, r.rule, rm[0].trim());
    }

    const keys = new Set(out.map(s => s.key));
    if (keys.has('wpl') && keys.has('present')) out = out.filter(s => s.key !== 'wpl');
    if (keys.has('talon_lightning')) out = out.filter(s => s.key !== 'talon');
    if (keys.has('sanforized_plus')) out = out.filter(s => s.key !== 'sanforized');

    for (const label of observations) {
      const o = cfg.observations.find(x => x.label === label);
      if (o) add(`obs:${label}`, label, o.rule, 'you observed this', 'you');
    }
    return out;
  }

  function bounds(signals) {
    let lo = Y0, hi = Y1;
    for (const s of signals) {
      if (!isHard(s)) continue;
      const [e, l] = span(s);
      lo = Math.max(lo, e ?? lo); hi = Math.min(hi, l ?? hi);
    }
    return [lo, hi];
  }

  function apply(prior, signals) {
    let hard = signals.filter(isHard);
    let dropped = [];
    let [lo, hi] = bounds(hard);
    if (lo > hi) {
      for (const level of ['low', 'medium']) {
        dropped = dropped.concat(hard.filter(s => s.rule.confidence === level));
        hard = hard.filter(s => s.rule.confidence !== level);
        [lo, hi] = bounds(hard);
        if (lo <= hi) break;
      }
      if (lo > hi) { dropped = dropped.concat(hard); hard = []; lo = Y0; hi = Y1; }
    }
    let post = prior.map((p, i) => (YEARS[i] >= lo && YEARS[i] <= hi) ? p : 0);
    const soft = signals.filter(s => !isHard(s));
    for (const s of soft) {
      const [a, b] = span(s);
      const w = cfg.soft_weight[s.rule.confidence];
      post = post.map((p, i) => (YEARS[i] >= (a ?? 0) && YEARS[i] <= (b ?? 9999)) ? p * w : p);
    }
    let sum = post.reduce((x, y) => x + y, 0);
    if (sum === 0) { post = YEARS.map(y => (y >= lo && y <= hi) ? 1 : 0); sum = post.reduce((x, y) => x + y, 0); }
    return { post: post.map(p => p / sum), active: hard.concat(soft), dropped, bounds: [lo, hi] };
  }

  // Narrowest 5/10/15-year window holding >= tau of the mass, else a 20-year 'low' range.
  // Masses rounded to 9 decimals so ties resolve to the earliest start exactly as in Python.
  const WINDOWS = [[5, 'five'], [10, 'ten'], [15, 'fifteen']];
  const r9 = v => Math.round(v * 1e9) / 1e9;
  function pickRange(post, [lo, hi]) {
    if (hi - lo <= 5) return { start: lo, end: hi, confidence: 1, kind: 'tell' };
    const cum = [0];
    for (const p of post) cum.push(cum[cum.length - 1] + p);
    const mass = (a, b) => cum[b - Y0 + 1] - cum[a - Y0];
    const bestWindow = width => {
      let best = null;
      for (let a = lo; a <= Math.max(lo, hi - width + 1); a++) {
        const b = Math.min(a + width - 1, hi), m = r9(mass(a, b));
        if (best === null || m > best[2]) best = [a, b, m];
      }
      return best;
    };
    for (const [width, kind] of WINDOWS) {
      if (width > hi - lo + 1) break;
      const [a, b, m] = bestWindow(width);
      if (m >= r9(head.meta.tau)) return { start: a, end: b, confidence: m, kind };
    }
    if (hi - lo <= 15) return { start: lo, end: hi, confidence: 1, kind: 'tell' };
    const [a, b, m] = bestWindow(20);
    return { start: a, end: b, confidence: m, kind: 'low' };
  }

  function headProbs(embeds) {
    // embeds: array of Float32Array(512), L2-normalized
    const T = Object.fromEntries(head.meta.tensors.map(t => [t.name, t]));
    const get = name => {
      const t = T[name], n = t.shape.reduce((a, b) => a * b, 1);
      return { data: head.data.subarray(t.offset, t.offset + n), shape: t.shape };
    };
    const mean = get('mean').data;
    const logps = embeds.map(e => {
      let h = Float32Array.from(e, (v, i) => v - mean[i]);
      for (let l = 0; l < head.meta.layers; l++) {
        const W = get(`W${l}`), b = get(`b${l}`).data;
        const [nin, nout] = W.shape;
        const o = new Float32Array(nout);
        for (let j = 0; j < nout; j++) o[j] = b[j];
        for (let i = 0; i < nin; i++) {
          const hi = h[i]; if (hi === 0) continue;
          const row = i * nout;
          for (let j = 0; j < nout; j++) o[j] += hi * W.data[row + j];
        }
        if (l < head.meta.layers - 1) for (let j = 0; j < nout; j++) o[j] = Math.max(o[j], 0);
        h = o;
      }
      const z = Array.from(h, v => v / head.meta.temperature);
      const mx = Math.max(...z);
      const lse = Math.log(z.reduce((acc, v) => acc + Math.exp(v - mx), 0));
      return z.map(v => v - mx - lse);
    });
    const k = head.meta.classes.length;
    const s = Array.from({ length: k }, (_, j) => logps.reduce((acc, p) => acc + p[j], 0) / logps.length);
    const smx = Math.max(...s);
    const ex = s.map(v => Math.exp(v - smx));
    const tot = ex.reduce((x, y) => x + y, 0);
    const avg = ex.map(v => v / tot);
    return Object.fromEntries(head.meta.classes.map((c, j) => [c, avg[j]]));
  }

  function yearPrior(bins) {
    const bs = Object.keys(bins).map(Number).sort((a, b) => a - b);
    const width = bs[1] - bs[0];
    const prior = new Array(N).fill(0);
    for (const d of bs) {
      const inSpan = YEARS.map(y => y >= (d === bs[0] ? d - 10 : d) && y <= d + width - 1);
      const n = inSpan.filter(Boolean).length;
      inSpan.forEach((v, i) => { if (v) prior[i] += bins[d] / n; });
    }
    return prior.map(p => (1 - cfg.floor) * p + cfg.floor / N);
  }

  const fmt = r => r.start === r.end ? String(r.start) : `${r.start}–${r.end}`;
  const sameRange = (a, b) => a.start === b.start && a.end === b.end;

  function toDecades(bins) {
    const out = {};
    for (const b of Object.keys(bins).map(Number)) { const d = Math.floor(b / 10) * 10; out[d] = (out[d] || 0) + bins[b]; }
    return out;
  }

  function dateFromBins(bins, tagText, observations) {
    const prior = yearPrior(bins);
    const decades = toDecades(bins);
    const signals = extractSignals(tagText, observations);
    const { post, active, dropped, bounds: bnds } = apply(prior, signals);
    const result = pickRange(post, bnds);
    const modelOnly = pickRange(prior, [Y0, Y1]);
    const massIn = (p, r) => p.reduce((a, v, i) => a + (YEARS[i] >= r.start && YEARS[i] <= r.end ? v : 0), 0);

    const drivers = active.map(s => {
      const wo = apply(prior, active.filter(x => x !== s));
      const changed = !sameRange(pickRange(wo.post, wo.bounds), result);
      return { label: s.label, span: describeSpan(s), hard: isHard(s), confidence: s.rule.confidence,
               source: s.source, evidence: s.evidence, notes: s.rule.notes || '', decisive: changed,
               impact: massIn(post, result) - massIn(wo.post, result) + (changed ? 1 : 0) };
    });
    drivers.sort((a, b) => r9(b.impact) - r9(a.impact));

    const top = Number(Object.keys(decades).reduce((a, b) => decades[a] >= decades[b] ? a : b));
    return {
      range: fmt(result), start: result.start, end: result.end, confidence: result.confidence, kind: result.kind,
      decided_by: sameRange(modelOnly, result) ? 'model' : 'rules',
      model: { bins, decades, top, range: fmt(modelOnly), confidence: modelOnly.confidence },
      signals: drivers, dropped: dropped.map(s => s.label), bounds: bnds, tag_text: tagText,
      posterior: post, years: YEARS, needs_more: result.kind === 'low',
      vintage_prob: post.reduce((a, v, i) => a + (YEARS[i] < cfg.vintage_before ? v : 0), 0),
    };
  }

  return { extractSignals, apply, pickRange, headProbs, yearPrior, dateFromBins, YEARS };
}
