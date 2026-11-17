"""Assemble a self-contained Hugging Face Space in dist/space/ (inference code, weights, knowledge base only)."""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "dist" / "space"

FILES = [
    ("app/app.py", "app.py"),
    ("model/__init__.py", "model/__init__.py"),
    ("model/backbone.py", "model/backbone.py"),
    ("model/constraints.py", "model/constraints.py"),
    ("model/predict.py", "model/predict.py"),
    ("model/weights/head.npz", "model/weights/head.npz"),
    ("model/weights/metrics.json", "model/weights/metrics.json"),
    ("knowledge/__init__.py", "knowledge/__init__.py"),
    ("knowledge/rn_lookup.py", "knowledge/rn_lookup.py"),
    ("knowledge/rn_cache.json", "knowledge/rn_cache.json"),
    ("knowledge/rules.json", "knowledge/rules.json"),
    ("knowledge/union_labels.json", "knowledge/union_labels.json"),
]

README = """---
title: Vintage Garment Dater
emoji: 🧵
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: {gradio}
python_version: "3.11"
app_file: app.py
pinned: false
short_description: Date vintage clothing from photos of the garment and tags
---

Upload photos of a vintage garment and its tags. A trained decade classifier plus a rules engine built on
garment-labeling history (FTC care labels, RN numbers, union labels, country of origin) returns a date range
and the evidence behind it.
"""

REQUIREMENTS = """--extra-index-url https://download.pytorch.org/whl/cpu
torch
transformers>=4.45
numpy
pandas
pillow
rapidocr_onnxruntime
"""


def main():
    import gradio
    if OUT.exists():
        shutil.rmtree(OUT)
    for src, dst in FILES:
        (OUT / dst).parent.mkdir(parents=True, exist_ok=True)
        s = ROOT / src
        (OUT / dst).write_bytes(s.read_bytes()) if s.exists() else (OUT / dst).touch()
    (OUT / "README.md").write_text(README.format(gradio=gradio.__version__), encoding="utf-8")
    (OUT / "requirements.txt").write_text(REQUIREMENTS)
    (OUT / ".gitattributes").write_text("*.npz filter=lfs diff=lfs merge=lfs -text\n")
    print(f"Space assembled at {OUT}")
    for p in sorted(OUT.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(OUT)}  ({p.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
