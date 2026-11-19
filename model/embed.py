"""Embed listing images with the frozen backbone, cached per listing.

data/embeddings/items/{item_id}.npy holds one row per image; a rerun only embeds listings without a
cache file, so new scrapes are incremental and interruptions lose at most one chunk. Labels are
re-extracted from titles at merge time, so label-extractor fixes apply without re-embedding.
Merged output: data/embeddings/{embeddings.npy, index.json}.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.backbone import SIZE, embed, load_backbone, load_image, preprocess  # noqa: E402
from scraper.label_extractor import extract  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "embeddings_b16"
ITEMS = OUT / "items"
CHUNK_IMAGES = 2048


def listings():
    out = []
    for meta_path in sorted(RAW.glob("*/metadata.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        label = extract(meta.get("title", ""))
        if not label or label.confidence == "low":
            continue
        imgs = sorted((meta_path.parent / "images").glob("*.jpg"), key=lambda p: int(p.stem))
        if imgs:
            out.append({"item_id": meta["item_id"], "start": label.start, "end": label.end, "images": imgs})
    return out


def migrate_legacy():
    """One-time split of the old monolithic embeddings.npy into per-listing files."""
    legacy_idx = OUT / "index.json"
    if not legacy_idx.exists() or any(ITEMS.glob("*.npy")):
        return
    idx = json.loads(legacy_idx.read_text())
    if "image" not in idx[0]:
        return
    X = np.load(OUT / "embeddings.npy")
    rows = {}
    for i, r in enumerate(idx):
        rows.setdefault(r["item_id"], []).append((r["image"], i))
    for item_id, pairs in rows.items():
        pairs.sort()
        np.save(ITEMS / f"{item_id}.npy", X[[i for _, i in pairs]])
    print(f"migrated {len(rows)} listings from legacy embeddings")


class Images(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            return preprocess(load_image(self.paths[i])), True
        except Exception:
            return torch.zeros(3, SIZE, SIZE), False


def main():
    ITEMS.mkdir(parents=True, exist_ok=True)
    migrate_legacy()
    items = listings()
    todo = [it for it in items if not (ITEMS / f"{it['item_id']}.npy").exists()]
    print(f"{len(items)} labeled listings, {len(todo)} need embedding "
          f"({sum(len(it['images']) for it in todo)} images)")

    if todo:
        torch.set_num_threads(8)
        model = load_backbone()
        chunks, cur, n = [], [], 0
        for it in todo:
            cur.append(it)
            n += len(it["images"])
            if n >= CHUNK_IMAGES:
                chunks.append(cur)
                cur, n = [], 0
        if cur:
            chunks.append(cur)

        for chunk in tqdm(chunks, desc="chunks"):
            paths = [p for it in chunk for p in it["images"]]
            loader = DataLoader(Images(paths), batch_size=64, num_workers=6)
            embs, ok = [], []
            for batch, good in loader:
                embs.append(embed(model, batch).numpy().astype(np.float16))
                ok.append(good.numpy())
            embs, ok = np.concatenate(embs), np.concatenate(ok)
            pos = 0
            for it in chunk:
                k = len(it["images"])
                e, g = embs[pos:pos + k], ok[pos:pos + k]
                pos += k
                tmp = ITEMS / f"{it['item_id']}.tmp.npy"
                np.save(tmp, e[g])
                tmp.replace(ITEMS / f"{it['item_id']}.npy")

    X, index = [], []
    for it in items:
        e = np.load(ITEMS / f"{it['item_id']}.npy")
        X.append(e)
        index += [{"item_id": it["item_id"], "start": it["start"], "end": it["end"]}] * len(e)
    np.save(OUT / "embeddings.npy", np.concatenate(X))
    (OUT / "index.json").write_text(json.dumps(index))
    print(f"merged {len(index)} image embeddings from {len(items)} listings")


if __name__ == "__main__":
    main()
