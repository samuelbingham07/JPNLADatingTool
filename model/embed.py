"""Embed every labeled listing image with the frozen backbone.

Saves a shard every 2048 images to data/embeddings/shards/; a rerun skips finished shards.
Merges into data/embeddings/{embeddings.npy, index.json} at the end.
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

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "embeddings"
SHARDS = OUT / "shards"
SHARD_SIZE = 2048


def collect():
    rows = []
    for meta_path in sorted(RAW.glob("*/metadata.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        label = meta.get("label")
        if not label:
            continue
        for img in sorted((meta_path.parent / "images").glob("*.jpg"), key=lambda p: int(p.stem)):
            rows.append({"item_id": meta["item_id"], "image": int(img.stem),
                         "start": label["start"], "end": label["end"], "path": str(img)})
    return rows


class Images(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        try:
            return preprocess(load_image(self.rows[i]["path"])), True
        except Exception:
            return torch.zeros(3, SIZE, SIZE), False


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = collect()
    print(f"{len(rows)} images from {len({r['item_id'] for r in rows})} labeled listings")

    torch.set_num_threads(8)
    model = load_backbone()
    SHARDS.mkdir(parents=True, exist_ok=True)
    chunks = range(0, len(rows), SHARD_SIZE)
    todo = [s for s in chunks if not (SHARDS / f"{s:07d}.npz").exists()]
    print(f"{len(chunks) - len(todo)}/{len(chunks)} shards already done")

    for start in tqdm(todo, desc="shards"):
        loader = DataLoader(Images(rows[start:start + SHARD_SIZE]), batch_size=64, num_workers=6)
        embs, ok = [], []
        for batch, good in loader:
            embs.append(embed(model, batch).numpy().astype(np.float16))
            ok.append(good.numpy())
        tmp = SHARDS / f"{start:07d}.tmp.npz"
        np.savez(tmp, emb=np.concatenate(embs), ok=np.concatenate(ok))
        tmp.replace(SHARDS / f"{start:07d}.npz")

    parts = [np.load(SHARDS / f"{s:07d}.npz") for s in chunks]
    embs = np.concatenate([p["emb"] for p in parts])
    ok = np.concatenate([p["ok"] for p in parts])

    keep = [{k: v for k, v in r.items() if k != "path"} for r, g in zip(rows, ok) if g]
    np.save(OUT / "embeddings.npy", embs[ok])
    (OUT / "index.json").write_text(json.dumps(keep))
    print(f"saved {ok.sum()} embeddings ({(~ok).sum()} unreadable images skipped)")


if __name__ == "__main__":
    main()
