"""
Convert Label Studio export JSON into training-ready labeled dataset.

Usage:
    python -m labeling.export_annotations --input annotations_export.json
    python -m labeling.export_annotations --input annotations_export.json --skip-skipped
"""

import argparse
import json
from pathlib import Path

OUTPUT_FILE = Path(__file__).parent.parent / "data" / "labeled.json"


def parse_annotation(task: dict) -> dict | None:
    data = task.get("data", {})
    annotations = task.get("annotations", [])

    if not annotations:
        return None

    # Take the first completed annotation
    ann = next((a for a in annotations if not a.get("was_cancelled")), None)
    if not ann:
        return None

    results = ann.get("result", [])

    def get_value(name):
        for r in results:
            if r.get("from_name") == name:
                val = r.get("value", {})
                # Number widget
                if "number" in val:
                    return val["number"]
                # Choices widget
                if "choices" in val:
                    choices = val["choices"]
                    return choices[0] if len(choices) == 1 else choices
                # TextArea widget
                if "text" in val:
                    texts = val["text"]
                    return texts[0] if texts else ""
        return None

    confidence = get_value("confidence")
    if confidence == "skip":
        return None

    date_start = get_value("date_start")
    date_end = get_value("date_end")

    if date_start is None or date_end is None:
        return None

    # Collect all signal choices
    signals = {}
    for key in ["signals_regulatory", "signals_country", "signals_union",
                "signals_construction", "signals_hardware", "signals_fiber",
                "signals_brand"]:
        val = get_value(key)
        if val:
            signals[key] = val if isinstance(val, list) else [val]

    # Image type classifications — keyed by index
    image_types = {}
    for r in results:
        name = r.get("from_name", "")
        if name.startswith("img_type_"):
            idx = name.split("_")[-1]
            choices = r.get("value", {}).get("choices", [])
            if choices:
                image_types[int(idx)] = choices[0]

    return {
        "item_id": data.get("item_id"),
        "title": data.get("title"),
        "images": data.get("images", []),
        "image_types": image_types,
        "garment_type": get_value("garment_type"),
        "material": get_value("material"),
        "brand_visible": get_value("brand_visible"),
        "brand_name": get_value("brand_name") or "",
        "tag_visibility": get_value("tag_visibility"),
        "style_signals": get_value("style_signals"),
        "label": {
            "start": int(date_start),
            "end": int(date_end),
            "confidence": confidence,
            "primary_signal": get_value("primary_signal"),
            "signals": signals,
            "notes": get_value("notes") or "",
        },
        "title_label": {
            "raw": data.get("label_raw"),
            "start": data.get("label_start"),
            "end": data.get("label_end"),
            "confidence": data.get("label_confidence"),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Label Studio export JSON file")
    parser.add_argument("--skip-skipped", action="store_true",
                        help="Exclude tasks annotators marked as skip")
    args = parser.parse_args()

    export_data = json.loads(Path(args.input).read_text())
    labeled = []
    skipped = 0

    for task in export_data:
        result = parse_annotation(task)
        if result is None:
            skipped += 1
            continue
        labeled.append(result)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(labeled, indent=2))

    print(f"Exported {len(labeled)} labeled examples ({skipped} skipped)")
    print(f"Output: {OUTPUT_FILE}")

    # Quick stats
    if labeled:
        from collections import Counter
        conf_counts = Counter(e["label"]["confidence"] for e in labeled)
        sig_counts = Counter(e["label"]["primary_signal"] for e in labeled)
        print(f"\nConfidence: {dict(conf_counts)}")
        print(f"Top primary signals: {dict(sig_counts.most_common(5))}")


if __name__ == "__main__":
    main()
