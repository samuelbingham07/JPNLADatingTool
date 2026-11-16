# Label Studio Setup

## Start Label Studio

```
label-studio start
```

Opens at http://localhost:8080. Create an account (local only).

## Create a project

1. New Project → name it "JPNLADatingTool"
2. Labeling Setup → Code → paste contents of `config.xml`
3. Settings → Local Storage → point to `data/raw` so images load

## Import tasks

```
python -m labeling.prepare_tasks --labeled-only --limit 500
```

Then in Label Studio: Import → upload `labeling/tasks.json`.

## Label

Per task:
- Classify each image (tag, garment, detail, etc.)
- Set date range start + end
- Pick confidence
- Check visible signals
- Pick the primary deciding signal
- Skip if genuinely undatable

## Export and convert

In Label Studio: Export → JSON → save as `labeling/annotations_export.json`

```
python -m labeling.export_annotations --input labeling/annotations_export.json
```

Output: `data/labeled.json` — training-ready.

## Notes

- Always label title-extracted dates (`label_raw`) as a starting point, then confirm or correct from the images.
- The `primary_signal` field is critical — it tells the model which visual feature drove the prediction.
- If only the title gave you the date and images are uninformative, mark `primary_signal = title_only` and `confidence = low`.
