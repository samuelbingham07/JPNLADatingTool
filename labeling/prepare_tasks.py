"""
Convert scraped eBay listings into Label Studio import format.

Usage:
    python -m labeling.prepare_tasks                    # process all raw data
    python -m labeling.prepare_tasks --limit 200        # first N listings
    python -m labeling.prepare_tasks --labeled-only     # only pre-labeled listings
"""

import argparse
import json
from pathlib import Path

from labeling.pre_annotate import build_prediction

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
OUTPUT_FILE = Path(__file__).parent / "tasks.json"


def listing_to_task(metadata: dict) -> dict | None:
    images = metadata.get("images", [])
    if not images:
        return None

    label = metadata.get("label") or {}

    image_paths = [
        f"http://localhost:8081/{img.replace(chr(92), '/')}" for img in images
    ]

    data = {
        "item_id": metadata["item_id"],
        "title": metadata.get("title", ""),
        "label_raw": label.get("raw", ""),
        "label_confidence": label.get("confidence", ""),
        "label_start": label.get("start"),
        "label_end": label.get("end"),
        "url": metadata.get("url", ""),
    }
    for i in range(6):
        data[f"image_{i}"] = image_paths[i] if i < len(image_paths) else ""

    prediction = build_prediction(data, label if label else None)

    return {"data": data, "predictions": [prediction]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--labeled-only", action="store_true")
    args = parser.parse_args()

    tasks = []
    item_dirs = sorted(DATA_DIR.iterdir()) if DATA_DIR.exists() else []

    for item_dir in item_dirs:
        metadata_path = item_dir / "metadata.json"
        if not metadata_path.exists():
            continue

        metadata = json.loads(metadata_path.read_text())

        if args.labeled_only and not metadata.get("label"):
            continue

        task = listing_to_task(metadata)
        if task:
            tasks.append(task)

        if args.limit and len(tasks) >= args.limit:
            break

    OUTPUT_FILE.write_text(json.dumps(tasks, indent=2))
    print(f"Wrote {len(tasks)} tasks to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
