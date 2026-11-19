"""Train the date head on frozen backbone embeddings.

The head predicts 5-year bins (1940-2019; the first bin also absorbs 1930s). Targets are soft: a
listing labelled "1970s" spreads over two bins, an exact year lands in one, so exact-year titles teach
sub-decade resolution. Photos of a listing are combined by averaging tempered log-probabilities.
The range threshold tau is calibrated on validation so the model-only range contains the true year
for TARGET_HIT of exact-year listings. Split is by listing (md5), so no listing leaks across splits.
Writes model/weights/head.npz and model/weights/metrics.json.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from model import constraints as C  # noqa: E402

ITEMS = ROOT / "data" / "embeddings_b16" / "items"
OUT = ROOT / "model" / "weights"
BINS = np.arange(1940, 2020, 5)
DECADES = list(range(1940, 2020, 10))
TARGET_HIT = 0.80
HIDDEN, DROP, EPOCHS, WD, POWER = 1024, 0.5, 40, 1e-3, 0.5
SEED = 0


def split_of(item_id):
    h = int(hashlib.md5(item_id.encode()).hexdigest(), 16) % 10
    return "test" if h == 0 else "val" if h == 1 else "train"


def load():
    from model.embed import listings
    X, lab, items = [], [], []
    for it in listings():
        f = ITEMS / f"{it['item_id']}.npy"
        if it["end"] - it["start"] > 20 or not f.exists():
            continue
        e = np.load(f).astype(np.float32)
        X.append(e)
        lab += [(min(max(it["start"], 1930), 2019), min(max(it["end"], 1930), 2019))] * len(e)
        items += [it["item_id"]] * len(e)
    items = np.array(items)
    return np.concatenate(X), np.array(lab), items, np.array([split_of(i) for i in items])


def soft_targets(lab):
    T = np.zeros((len(lab), len(BINS)), np.float32)
    for k, (s, e) in enumerate(lab):
        ys = np.arange(s, e + 1)
        np.add.at(T[k], np.clip((ys - BINS[0]) // 5, 0, len(BINS) - 1), 1.0 / len(ys))
    return T


def fit(X, T, w):
    torch.manual_seed(SEED)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], HIDDEN), torch.nn.ReLU(), torch.nn.Dropout(DROP),
                              torch.nn.Linear(HIDDEN, T.shape[1]))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    Xt, Tt, wt = torch.from_numpy(X), torch.from_numpy(T), torch.from_numpy(w.astype(np.float32))
    for _ in range(EPOCHS):
        net.train()
        for b in torch.randperm(len(Xt)).split(256):
            loss = (-(Tt[b] * torch.log_softmax(net(Xt[b]), 1)).sum(1) * wt[b]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    l1, l2 = net[0], net[3]
    return [(l1.weight.detach().numpy().T.copy(), l1.bias.detach().numpy().copy()),
            (l2.weight.detach().numpy().T.copy(), l2.bias.detach().numpy().copy())]


def logits(layers, X):
    h = X
    for i, (W, b) in enumerate(layers):
        h = h @ W + b
        if i < len(layers) - 1:
            h = np.maximum(h, 0)
    return h


def combine(Z, temperature):
    """Average tempered log-probs over a listing's photos, renormalize. Mirrors predict.bin_probs."""
    z = Z / temperature
    lp = z - z.max(1, keepdims=True)
    lp = lp - np.log(np.exp(lp).sum(1, keepdims=True))
    s = lp.mean(0)
    p = np.exp(s - s.max())
    return p / p.sum()


def listings_of(items):
    g = {}
    for i, it in enumerate(items):
        g.setdefault(it, []).append(i)
    return list(g.values())


def evaluate(Z, lab, items, temperature, tau):
    from model.predict import year_prior
    full = (int(C.YEARS[0]), int(C.YEARS[-1]))
    rows = listings_of(items)
    widths, exact, ranged, nll, dec_true, dec_pred, kinds = [], [], [], [], [], [], {}
    for r in rows:
        p = combine(Z[r], temperature)
        s, e = lab[r[0]]
        prior = year_prior(dict(zip(BINS.tolist(), p.tolist())))
        nll.append(-np.log(prior[(C.YEARS >= s) & (C.YEARS <= e)].sum()))
        res = C.pick_range(prior, full, tau)
        kinds[res["kind"]] = kinds.get(res["kind"], 0) + 1
        widths.append(res["end"] - res["start"] + 1)
        (exact if e - s <= 2 else ranged).append(res["start"] <= e and res["end"] >= s)
        dec_true.append((min(max((s + e) // 2, 1940), 2019) - 1940) // 10)
        dec_pred.append(int(p.reshape(-1, 2).sum(1).argmax()))
    dt, dpd = np.array(dec_true), np.array(dec_pred)
    recall = {f"{d}s": float((dpd[dt == k] == k).mean()) for k, d in enumerate(DECADES) if (dt == k).any()}
    return {"listings": len(rows), "nll_true_range": float(np.mean(nll)),
            "mean_range_width_years": float(np.mean(widths)),
            "exact_year_hit": float(np.mean(exact)), "range_label_overlap": float(np.mean(ranged)),
            "range_kinds": kinds, "decade_accuracy": float((dt == dpd).mean()),
            "within_one_decade": float((np.abs(dt - dpd) <= 1).mean()), "per_decade_recall": recall}


def main():
    X, lab, items, sp = load()
    mean = X[sp == "train"].mean(0)
    X = X - mean
    tr, va, te = (sp == s for s in ("train", "val", "test"))
    dec = np.clip((lab[tr].mean(1).astype(int) - 1940) // 10, 0, 7)
    cnt = np.bincount(dec, minlength=8)
    print("train images per decade:", dict(zip(DECADES, cnt.tolist())))
    print(f"listings: train {len(set(items[tr]))}, val {len(set(items[va]))}, test {len(set(items[te]))}")
    w = np.minimum((cnt.max() / cnt[dec]) ** POWER, 20.0)
    layers = fit(X[tr], soft_targets(lab[tr]), w / w.mean())

    Zv, Zt = logits(layers, X[va]), logits(layers, X[te])
    from model.predict import year_prior
    rows = listings_of(items[va])

    def val_nll(t):
        out = []
        for r in rows:
            prior = year_prior(dict(zip(BINS.tolist(), combine(Zv[r], t).tolist())))
            s, e = lab[va][r[0]]
            out.append(-np.log(prior[(C.YEARS >= s) & (C.YEARS <= e)].sum()))
        return np.mean(out)

    temperature = float(min(np.arange(0.4, 1.31, 0.1).round(2), key=val_nll))
    taus = np.arange(0.30, 0.96, 0.01).round(2)
    tau = float(next(t for t in taus if evaluate(Zv, lab[va], items[va], temperature, t)["exact_year_hit"] >= TARGET_HIT))
    val = evaluate(Zv, lab[va], items[va], temperature, tau)
    test = evaluate(Zt, lab[te], items[te], temperature, tau)
    print(f"temperature {temperature}, tau {tau}\nval: {json.dumps(val)}\ntest: {json.dumps(test, indent=1)}")

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "head.npz", mean=mean.astype(np.float32), classes=BINS, temperature=np.float32(temperature),
             tau=np.float32(tau), **{f"W{i}": W for i, (W, _) in enumerate(layers)},
             **{f"b{i}": b for i, (_, b) in enumerate(layers)})
    (OUT / "metrics.json").write_text(json.dumps({
        "head": f"MLP {HIDDEN} hidden, dropout {DROP}, 5-year bins, class weights (max/count)^{POWER}",
        "bins": BINS.tolist(), "temperature": temperature, "tau": tau, "target_exact_year_hit": TARGET_HIT,
        "label_source": "eBay listing titles (weak labels)",
        "backbone": "openai/clip-vit-base-patch16 (frozen)",
        "train_images": int(tr.sum()), "train_listings": len(set(items[tr])),
        "val": val, "test": test}, indent=1))


if __name__ == "__main__":
    main()
