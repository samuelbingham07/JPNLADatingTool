"""GPU (DirectML) embedding with the same ONNX vision model the browser runs.

Runs in .venv-dml (onnxruntime-directml + numpy + pillow; no torch). Same per-listing cache layout as
model/embed.py, under data/embeddings_<tag>/items/. Writes are atomic per listing so a rerun resumes.
Usage: .venv-dml/Scripts/python model/embed_dml.py <hf_repo> <tag> [--items list.txt]
"""
import json, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SIZE = 224
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)


def prep(path):
    try:
        img = Image.open(path)
        img.draft("RGB", (SIZE * 2, SIZE * 2))
        img = img.convert("RGB")
        w, h = img.size
        s = SIZE / min(w, h)
        img = img.resize((max(SIZE, round(w * s)), max(SIZE, round(h * s))), Image.BICUBIC)
        w, h = img.size
        l, t = (w - SIZE) // 2, (h - SIZE) // 2
        a = np.asarray(img.crop((l, t, l + SIZE, t + SIZE)), np.float32) / 255.0
        return ((a - MEAN) / STD).transpose(2, 0, 1), True
    except Exception:
        return np.zeros((3, SIZE, SIZE), np.float32), False


def session(model_path):
    so = ort.SessionOptions()
    return ort.InferenceSession(str(model_path), so, providers=["DmlExecutionProvider", "CPUExecutionProvider"])


def run(sess, batch):
    out = sess.run(["image_embeds"], {"pixel_values": batch})[0].astype(np.float32)
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def main():
    repo, tag = sys.argv[1], sys.argv[2]
    only = None
    if "--items" in sys.argv:
        only = set(Path(sys.argv[sys.argv.index("--items") + 1]).read_text().split())
    from huggingface_hub import hf_hub_download
    sess = session(hf_hub_download(repo, "onnx/vision_model.onnx"))
    items_dir = ROOT / "data" / f"embeddings_{tag}" / "items"
    items_dir.mkdir(parents=True, exist_ok=True)
    todo = []
    for meta_path in sorted((ROOT / "data" / "raw").glob("*/metadata.json")):
        iid = meta_path.parent.name
        if only is not None and iid not in only:
            continue
        if (items_dir / f"{iid}.npy").exists():
            continue
        imgs = sorted((meta_path.parent / "images").glob("*.jpg"), key=lambda p: int(p.stem))
        if imgs:
            todo.append((iid, imgs))
    total = sum(len(i) for _, i in todo)
    print(f"{len(todo)} listings / {total} images to embed", flush=True)
    pool = ThreadPoolExecutor(8)
    done, t0, B = 0, time.time(), 64
    pending = []  # (iid, n_images)
    buf_x, buf_ok = [], []
    def flush(force=False):
        nonlocal buf_x, buf_ok, pending
        if not pending or (not force and len(buf_x) < 512):
            return
        X = np.stack(buf_x)
        E = np.concatenate([run(sess, X[i:i + B]) for i in range(0, len(X), B)]).astype(np.float16)
        ok = np.array(buf_ok)
        pos = 0
        for iid, k in pending:
            tmp = items_dir / f"{iid}.tmp.npy"
            np.save(tmp, E[pos:pos + k][ok[pos:pos + k]])
            tmp.replace(items_dir / f"{iid}.npy")
            pos += k
        buf_x, buf_ok, pending = [], [], []
    for n, (iid, imgs) in enumerate(todo):
        for x, ok in pool.map(prep, imgs):
            buf_x.append(x); buf_ok.append(ok)
        pending.append((iid, len(imgs)))
        done += len(imgs)
        flush()
        if n % 200 == 0:
            r = done / (time.time() - t0)
            print(f"{done}/{total} images, {r:.0f} img/s, eta {(total - done) / max(r, 1e-9) / 60:.0f} min", flush=True)
    flush(force=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
