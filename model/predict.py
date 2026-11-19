import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from model import constraints as C  # noqa: E402
from model.backbone import embed, load_backbone, load_image, preprocess  # noqa: E402

HEAD = ROOT / "model" / "weights" / "head.npz"
OCR_MAX = 1600
FLOOR = 0.03
VINTAGE_BEFORE = 2006  # mass spread uniformly so rules can push outside the model's training decades


@lru_cache(maxsize=1)
def _backbone():
    return load_backbone()


@lru_cache(maxsize=1)
def _head():
    z = np.load(HEAD)
    layers = [(z[f"W{i}"], z[f"b{i}"]) for i in range(len([k for k in z.files if k.startswith("W")]))]
    return layers, z["mean"], [int(c) for c in z["classes"]], round(float(z["temperature"]), 4), round(float(z["tau"]), 4)


@lru_cache(maxsize=1)
def _ocr():
    from rapidocr_onnxruntime import RapidOCR
    return RapidOCR()


def ocr(images) -> str:
    lines = []
    for img in images:
        img = img.convert("RGB")
        img.thumbnail((OCR_MAX, OCR_MAX))
        res, _ = _ocr()(np.asarray(img))
        lines += [r[1] for r in (res or []) if float(r[2]) > 0.5]
    return "\n".join(lines)


def head_probs(x: np.ndarray) -> np.ndarray:
    """Bin probabilities for one garment from its photos' embeddings: tempered log-probs averaged."""
    layers, mean, _, temperature, _ = _head()
    h = x - mean
    for i, (W, b) in enumerate(layers):
        h = h @ W + b
        if i < len(layers) - 1:
            h = np.maximum(h, 0)
    z = h / temperature
    lp = z - z.max(axis=1, keepdims=True)
    lp = lp - np.log(np.exp(lp).sum(axis=1, keepdims=True))
    s = lp.mean(axis=0)
    p = np.exp(s - s.max())
    return p / p.sum()


def bin_probs(images) -> dict[int, float]:
    batch = torch.stack([preprocess(load_image(img)) for img in images])
    x = embed(_backbone(), batch).numpy()
    return dict(zip(_head()[2], head_probs(x).tolist()))


def to_decades(bins: dict[int, float]) -> dict[int, float]:
    out = {}
    for b, p in bins.items():
        out[b // 10 * 10] = out.get(b // 10 * 10, 0.0) + p
    return out


def year_prior(bins: dict[int, float]) -> np.ndarray:
    """Spread each bin's mass over its years; the lowest bin also covers the 10 years before it."""
    starts = sorted(bins)
    width = starts[1] - starts[0]
    prior = np.zeros(len(C.YEARS))
    for b, p in bins.items():
        span = (C.YEARS >= (b - 10 if b == starts[0] else b)) & (C.YEARS <= b + width - 1)
        prior += p * span / span.sum()
    return (1 - FLOOR) * prior + FLOOR / len(C.YEARS)


def _fmt(r):
    return str(r["start"]) if r["start"] == r["end"] else f"{r['start']}–{r['end']}"


def date_garment(images, tag_text: str | None = None, observations=()):
    images = [i if isinstance(i, Image.Image) else Image.open(i) for i in images]
    if tag_text is None:
        tag_text = ocr(images)
    return date_from_bins(bin_probs([load_image(i.copy()) for i in images]), tag_text, observations)


def date_from_bins(bins: dict[int, float], tag_text: str, observations=()):
    tau = _head()[4]
    prior = year_prior(bins)
    signals = C.extract_signals(tag_text, observations)
    post, active, dropped, bounds = C.apply(prior, signals)
    result = C.pick_range(post, bounds, tau)
    model_only = C.pick_range(prior, (int(C.YEARS[0]), int(C.YEARS[-1])), tau)

    def mass(p, r):
        return float(p[(C.YEARS >= r["start"]) & (C.YEARS <= r["end"])].sum())

    drivers = []
    for s in active:
        p_wo, _, _, b_wo = C.apply(prior, [x for x in active if x is not s])
        alt = C.pick_range(p_wo, b_wo, tau)
        changed = (alt["start"], alt["end"]) != (result["start"], result["end"])
        drivers.append({"label": s.label, "span": s.describe_span(), "hard": s.is_hard,
                        "confidence": s.rule["confidence"], "source": s.source, "evidence": s.evidence,
                        "notes": s.rule.get("notes", ""), "decisive": changed,
                        "impact": mass(post, result) - mass(p_wo, result) + (1.0 if changed else 0.0)})
    drivers.sort(key=lambda d: -round(d["impact"], 9))

    decades = to_decades(bins)
    top_decade = max(decades, key=decades.get)
    rules_decided = (model_only["start"], model_only["end"]) != (result["start"], result["end"])
    return {
        "range": _fmt(result), "start": result["start"], "end": result["end"],
        "confidence": round(result["confidence"], 3), "kind": result["kind"],
        "decided_by": "rules" if rules_decided else "model",
        "model": {"bins": bins, "decades": decades, "top": top_decade, "range": _fmt(model_only),
                  "confidence": round(model_only["confidence"], 3)},
        "signals": drivers, "dropped": [s.label for s in dropped], "bounds": bounds,
        "tag_text": tag_text, "posterior": post.tolist(),
        "needs_more": result["kind"] == "low",
        "vintage_prob": float(post[C.YEARS < VINTAGE_BEFORE].sum()),
    }


if __name__ == "__main__":
    out = date_garment(sys.argv[1:])
    out.pop("posterior")
    print(json.dumps(out, indent=2, ensure_ascii=False))
