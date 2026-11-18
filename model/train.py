"""Train the decade head on frozen backbone embeddings.

Labels are title-extracted (weak). Split is by listing so no listing's photos leak across splits.
Writes model/weights/head.npz and model/weights/metrics.json.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, log_loss

ROOT = Path(__file__).resolve().parent.parent
EMB = ROOT / "data" / "embeddings"
OUT = ROOT / "model" / "weights"
DECADES = [1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010]
SEED = 0


def split_of(item_id):
    h = int(hashlib.md5(item_id.encode()).hexdigest(), 16) % 10
    return "test" if h == 0 else "val" if h == 1 else "train"


def load():
    X = np.load(EMB / "embeddings.npy").astype(np.float32)
    idx = json.loads((EMB / "index.json").read_text())
    keep, y, items, splits = [], [], [], []
    for i, r in enumerate(idx):
        if r["end"] - r["start"] > 20:
            continue
        d = int((r["start"] + r["end"]) / 2) // 10 * 10
        keep.append(i)
        y.append(DECADES.index(min(max(d, DECADES[0]), DECADES[-1])))
        items.append(r["item_id"])
        splits.append(split_of(r["item_id"]))
    return X[keep], np.array(y), np.array(items), np.array(splits)


def listing_level(P, y, items):
    """Average per-image probabilities within each listing."""
    order = {}
    for i, it in enumerate(items):
        order.setdefault(it, []).append(i)
    rows = list(order.values())
    return np.stack([P[r].mean(0) for r in rows]), np.array([y[r[0]] for r in rows])


def evaluate(P, y, items):
    Pl, yl = listing_level(P, y, items)
    pred = Pl.argmax(1)
    cm = confusion_matrix(yl, pred, labels=range(len(DECADES)))
    recall = cm.diagonal() / np.maximum(cm.sum(1), 1)
    return {
        "listings": int(len(yl)),
        "accuracy": float((pred == yl).mean()),
        "within_one_decade": float((np.abs(pred - yl) <= 1).mean()),
        "macro_recall": float(recall[cm.sum(1) > 0].mean()),
        "log_loss": float(log_loss(yl, Pl, labels=range(len(DECADES)))),
        "balanced_log_loss": float(log_loss(yl, Pl, labels=range(len(DECADES)),
                                            sample_weight=1.0 / np.bincount(yl, minlength=len(DECADES))[yl])),
        "per_decade_recall": {f"{d}s": float(r) for d, r, n in zip(DECADES, recall, cm.sum(1)) if n},
        "confusion": cm.tolist(),
    }


def fit_logreg(Xtr, ytr, C, weights):
    clf = LogisticRegression(C=C, max_iter=2000)
    clf.fit(Xtr, ytr, sample_weight=weights)
    return [(clf.coef_.T.astype(np.float32), clf.intercept_.astype(np.float32))]


def fit_mlp(Xtr, ytr, weights, hidden=512, epochs=30, wd=1e-3, drop=0.3):
    torch.manual_seed(SEED)
    k = len(DECADES)
    net = torch.nn.Sequential(torch.nn.Linear(Xtr.shape[1], hidden), torch.nn.ReLU(),
                              torch.nn.Dropout(drop), torch.nn.Linear(hidden, k))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=wd)
    X, y, w = torch.from_numpy(Xtr), torch.from_numpy(ytr), torch.from_numpy(weights.astype(np.float32))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for _ in range(epochs):
        net.train()
        for b in torch.randperm(len(X)).split(256):
            loss = (torch.nn.functional.cross_entropy(net(X[b]), y[b], reduction="none") * w[b]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    l1, l2 = net[0], net[3]
    return [(l1.weight.detach().numpy().T.copy(), l1.bias.detach().numpy().copy()),
            (l2.weight.detach().numpy().T.copy(), l2.bias.detach().numpy().copy())]


def probs(layers, X):
    h = X
    for i, (W, b) in enumerate(layers):
        h = h @ W + b
        if i < len(layers) - 1:
            h = np.maximum(h, 0)
    h = h - h.max(1, keepdims=True)
    p = np.exp(h)
    return p / p.sum(1, keepdims=True)


def main():
    X, y, items, splits = load()
    mean = X[splits == "train"].mean(0)
    X = X - mean
    tr, va, te = (splits == s for s in ("train", "val", "test"))
    counts = np.bincount(y[tr], minlength=len(DECADES))
    print("train images per decade:", dict(zip(DECADES, counts.tolist())))
    print(f"listings: train {len(set(items[tr]))}, val {len(set(items[va]))}, test {len(set(items[te]))}")

    # Class-reweighting by power of inverse frequency: 0 = none, 0.5 = sqrt, 1 = fully balanced
    def class_weights(power):
        w = np.minimum((counts.max() / np.maximum(counts[y[tr]], 1)) ** power, 20.0)
        return w / w.mean()

    candidates = {"logreg C=3.0 w=sqrt": fit_logreg(X[tr], y[tr], 3.0, class_weights(0.5))}
    print("fit logreg")
    for power in (0.0, 0.5, 0.75, 1.0):
        candidates[f"mlp512 w^{power}"] = fit_mlp(X[tr], y[tr], class_weights(power))
        print("fit", f"mlp512 w^{power}")

    results = {name: evaluate(probs(L, X[va]), y[va], items[va]) for name, L in candidates.items()}
    for name, r in results.items():
        print(f"{name:28s} val acc {r['accuracy']:.3f}  ±1 {r['within_one_decade']:.3f}  "
              f"macroR {r['macro_recall']:.3f}  nll {r['log_loss']:.3f}  bal-nll {r['balanced_log_loss']:.3f}")
    # Selected on decade-balanced log loss so the most-scraped decade can't dominate the tool's output
    best = min(results, key=lambda n: results[n]["balanced_log_loss"])
    layers = candidates[best]

    test = evaluate(probs(layers, X[te]), y[te], items[te])
    _, yl = listing_level(probs(layers, X[te]), y[te], items[te])
    majority = int(np.argmax(counts))
    test["majority_baseline_accuracy"] = float((yl == majority).mean())
    print(f"\nselected: {best}\ntest: {json.dumps({k: v for k, v in test.items() if k != 'confusion'}, indent=1)}")

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "head.npz", mean=mean.astype(np.float32), classes=np.array(DECADES),
             **{f"W{i}": W for i, (W, _) in enumerate(layers)},
             **{f"b{i}": b for i, (_, b) in enumerate(layers)})
    (OUT / "metrics.json").write_text(json.dumps({
        "selected": best, "decades": DECADES, "label_source": "eBay listing titles (weak labels)",
        "backbone": "openai/clip-vit-base-patch32 (frozen)",
        "train_images": int(tr.sum()), "train_listings": len(set(items[tr])),
        "val": results, "test": test}, indent=1))


if __name__ == "__main__":
    main()
