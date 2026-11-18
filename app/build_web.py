"""Assemble the free, fully in-browser version as a static Hugging Face Space in dist/web/.

The browser runs the same frozen CLIP encoder (transformers.js, fp16) and our trained head, and a JS
port of the constraint engine. Engine patterns, rules, and the RN table are exported from the Python
source here so there is one source of truth.
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from build_space import EXAMPLE_MAX, EXAMPLES  # noqa: E402  (app/ is on sys.path as the script dir)
from knowledge import rn_lookup  # noqa: E402
from model import constraints as C  # noqa: E402
from model import predict as P  # noqa: E402

SRC = ROOT / "app" / "web"
OUT = ROOT / "dist" / "web"

README = """---
title: Vintage Garment Dater
emoji: 🧵
colorFrom: yellow
colorTo: red
sdk: static
pinned: false
short_description: Date vintage clothing from photos of the garment and tags
---

Upload photos of a vintage garment and its tags. A trained decade classifier plus a rules engine built on
garment-labeling history (FTC care labels, RN numbers, union labels, country of origin) returns a date range,
a vintage verdict, and the evidence behind it. Everything runs in your browser.
"""


def export_head():
    z = np.load(P.HEAD)
    n = len([k for k in z.files if k.startswith("W")])
    arrays, layout, offset = [], [], 0
    for name in ["mean"] + [x for i in range(n) for x in (f"W{i}", f"b{i}")]:
        a = z[name].astype(np.float32)
        layout.append({"name": name, "shape": list(a.shape), "offset": offset})
        offset += a.size
        arrays.append(a.ravel())
    (OUT / "head.bin").write_bytes(np.concatenate(arrays).tobytes())
    (OUT / "head.json").write_text(json.dumps({"layers": n, "classes": [int(c) for c in z["classes"]],
                                               "tensors": layout}))


def export_engine():
    def rule(path):
        return C._union_rule(path[1], path[2]) if path[0] == "union" else C._rule(*path)

    cfg = {
        "years": [int(C.YEARS[0]), int(C.YEARS[-1])],
        "soft_weight": C.SOFT_WEIGHT,
        "confident_mass": C.CONFIDENT_MASS,
        "floor": P.FLOOR,
        "vintage_before": P.VINTAGE_BEFORE,
        "countries": [{"name": k, "pattern": v, "rule": C._rule("country_of_origin", k)}
                      for k, v in C.COUNTRY_ALIASES.items()],
        "anywhere_countries": [{"name": k, "pattern": v, "rule": C._rule("country_of_origin", k)}
                               for k, v in C.ANYWHERE_COUNTRIES.items()],
        "wpl_rule": C._rule("rn_number", "wpl_present"),
        "rn_present_rule": C._rule("rn_number", "rn_present"),
        "wpl_kept_after_1971": sorted(C.WPL_KEPT_AFTER_1971),
        "unions": [{"name": k, "pattern": v, "rule": C._union_rule(k)} for k, v in C.UNION_PATTERNS],
        "text_rules": [{"pattern": pat, "key": path[1], "label": label, "rule": C._rule(*path)}
                       for pat, path, label in C.TEXT_RULES],
        "observations": [{"label": k, "rule": rule(v)} for k, v in C.OBSERVATIONS.items()],
        "rn": {"ranges": rn_lookup._RN_RANGES, "gap": [rn_lookup._GAP_MIN, rn_lookup._GAP_MAX],
               "cache": rn_lookup._load_cache()},
    }
    (OUT / "engine.json").write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")


def copy_examples():
    from PIL import Image, ImageOps
    labels = []
    for i, (src, label) in enumerate(EXAMPLES.items()):
        dst = OUT / "examples" / f"{i}"
        dst.mkdir(parents=True)
        files = []
        for j, p in enumerate(sorted((ROOT / "examples" / src).glob("*.jp*g"))):
            img = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
            img.thumbnail((EXAMPLE_MAX, EXAMPLE_MAX))
            img.save(dst / f"{j}.jpg", quality=88)  # re-encoding without exif= drops GPS and all other metadata
            files.append(f"examples/{i}/{j}.jpg")
        # Examples ship the tag text read by the full Python OCR, so they reproduce the tested results exactly
        tag_text = P.ocr([Image.open(OUT / f) for f in files])
        labels.append({"label": label, "files": files, "tag_text": tag_text})
    (OUT / "examples" / "examples.json").write_text(json.dumps(labels, ensure_ascii=False, indent=1), encoding="utf-8")


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    for f in SRC.iterdir():
        shutil.copy(f, OUT / f.name)
    bundle = ROOT / "app" / "web_ocr" / "ocr.bundle.js"
    if not bundle.exists():
        sys.exit("Build the OCR bundle first: cd app/web_ocr && npm install && npm run build")
    shutil.copy(bundle, OUT / "ocr.bundle.js")
    (OUT / "ocr").mkdir()
    for name in ("ch_PP-OCRv4_det_infer.onnx", "ch_PP-OCRv4_rec_infer.onnx", "ppocr_keys_v1.txt"):
        shutil.copy(ROOT / "app" / "web_ocr" / "node_modules" / "@gutenye" / "ocr-models" / "assets" / name, OUT / "ocr" / name)
    export_head()
    export_engine()
    shutil.copy(ROOT / "model" / "weights" / "metrics.json", OUT / "metrics.json")
    copy_examples()
    (OUT / "README.md").write_text(README, encoding="utf-8")
    (OUT / ".gitattributes").write_text("*.bin filter=lfs diff=lfs merge=lfs -text\n")
    print(f"Static site assembled at {OUT}")
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and "examples" not in p.parts:
            print(f"  {p.relative_to(OUT)}  ({p.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
