import html
import json
import sys
from pathlib import Path

import gradio as gr
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE if (HERE / "model").is_dir() else HERE.parent
sys.path.insert(0, str(ROOT))
from model import constraints as C  # noqa: E402
from model.predict import date_garment  # noqa: E402

METRICS = json.loads((ROOT / "model" / "weights" / "metrics.json").read_text())

CSS = """
.gradio-container { max-width: 1120px !important; margin: auto; }
#title h1 { font-family: 'Fraunces', Georgia, serif; font-weight: 600; letter-spacing: -0.01em; margin-bottom: 0; }
#title p { color: var(--body-text-color-subdued); margin-top: 4px; }
.result-card { border: 1px solid var(--border-color-primary); border-radius: 14px; padding: 22px 24px;
  background: var(--background-fill-secondary); }
.result-range { font-family: 'Fraunces', Georgia, serif; font-size: 3.2rem; line-height: 1; font-weight: 600; margin: 6px 0 10px; }
.result-meta { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; font-size: 0.92rem; }
.pill { border-radius: 999px; padding: 3px 10px; border: 1px solid var(--border-color-primary); }
.pill.rules { background: #8a5a2b; color: #fff; border-color: #8a5a2b; }
.pill.model { background: #2f4f4f; color: #fff; border-color: #2f4f4f; }
.pill.warn { background: #b4441f; color: #fff; border-color: #b4441f; }
.conf-track { height: 8px; border-radius: 4px; background: var(--border-color-primary); margin: 14px 0 4px; overflow: hidden; }
.conf-fill { height: 100%; background: #8a5a2b; }
.sig { border-top: 1px solid var(--border-color-primary); padding: 12px 0 2px; }
.sig-head { display: flex; justify-content: space-between; gap: 12px; font-weight: 600; }
.sig-span { font-variant-numeric: tabular-nums; white-space: nowrap; }
.sig-note { color: var(--body-text-color-subdued); font-size: 0.9rem; margin-top: 4px; }
.tagline { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--body-text-color-subdued); }
"""

HEAD = '<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600&display=swap" rel="stylesheet">'

LOW_CONF_ASK = ("Low confidence. To tighten it, add a clear, straight-on photo of the neck tag, "
                "the care/content tag (often a side seam), and any RN/WPL or union label.")


def _signal_html(s):
    kind = "Hard bound" if s["hard"] else "Soft anchor"
    flag = " · <b>deciding factor</b>" if s["decisive"] else ""
    src = "from tag text" if s["source"] == "tag text" else "you observed"
    ev = f" — <i>“{html.escape(s['evidence'])}”</i>" if s["source"] == "tag text" else ""
    return (f'<div class="sig"><div class="sig-head"><span>{html.escape(s["label"])}</span>'
            f'<span class="sig-span">{s["span"]}</span></div>'
            f'<div class="sig-note">{kind}, {s["confidence"]} confidence, {src}{ev}{flag}</div>'
            f'<div class="sig-note">{html.escape(s["notes"])}</div></div>')


def render(r):
    decided = ('<span class="pill rules">Decided by knowledge-base rules</span>' if r["decided_by"] == "rules"
               else '<span class="pill model">Decided by visual model</span>')
    top = r["model"]["top"]
    top_p = r["model"]["decades"][top]
    model_line = (f'<div class="sig"><div class="sig-head"><span>Visual model</span>'
                  f'<span class="sig-span">{"≤" if top == min(r["model"]["decades"]) else ""}{top}s · {top_p:.0%}</span></div>'
                  f'<div class="sig-note">Trained date classifier over the uploaded photos. '
                  f'On its own it would say {r["model"]["range"]}.</div></div>')
    sigs = "".join(_signal_html(s) for s in r["signals"][:3])
    more = len(r["signals"]) - 3
    extra = f'<div class="sig-note" style="margin-top:8px">+{more} more signal(s) below.</div>' if more > 0 else ""
    dropped = (f'<div class="sig-note" style="margin-top:8px">Conflicting rules ignored: {html.escape(", ".join(r["dropped"]))}</div>'
               if r["dropped"] else "")
    warn = f'<div class="sig-note" style="margin-top:10px"><span class="pill warn">More photos</span> {LOW_CONF_ASK}</div>' if r["needs_more"] else ""
    v = r["vintage_prob"]
    verdict = "Vintage" if v >= 0.8 else "Not vintage" if v <= 0.2 else "Borderline vintage"
    vintage = f'<span class="pill">{verdict} (pre-2006): {v:.0%}</span>'
    kind = {"tell": "Clear tell", "five": "5-year window", "ten": "10-year window",
            "fifteen": "15-year window", "low": "Wide range"}[r["kind"]]
    return f"""
<div class="result-card">
  <div class="tagline">Estimated production date · {kind}</div>
  <div class="result-range">{r['range']}</div>
  <div class="result-meta">{decided}<span class="pill">Confidence {r['confidence']:.0%}</span>{vintage}</div>
  <div class="conf-track"><div class="conf-fill" style="width:{r['confidence']*100:.0f}%"></div></div>
  {warn}
  <div class="tagline" style="margin-top:18px">What drove it</div>
  {sigs}{model_line}{extra}{dropped}
</div>"""


def run(files, tag_text, observations, use_typed_text):
    if not files:
        raise gr.Error("Upload at least one photo of the garment or its tags.")
    paths = [f if isinstance(f, str) else f.name for f in files]
    r = date_garment(paths, tag_text if use_typed_text and tag_text.strip() else None, observations)
    years = pd.DataFrame({"Year": C.YEARS, "Probability": r["posterior"]})
    years = years[years.Year >= 1925]
    all_sigs = pd.DataFrame([{"Signal": s["label"], "Range": s["span"], "Type": "hard" if s["hard"] else "soft",
                              "Confidence": s["confidence"], "Source": s["source"]} for s in r["signals"]]
                            or [{"Signal": "No tag-text or observed signals", "Range": "", "Type": "", "Confidence": "", "Source": ""}])
    return render(r), years, all_sigs, r["tag_text"], gr.update(value=True)


def about_md():
    t = METRICS["test"]
    recall = " · ".join(f"{k} {v:.0%}" for k, v in t["per_decade_recall"].items())
    return f"""
### How it works
1. **Visual model.** Each photo is embedded with a frozen open-source image encoder ({METRICS['backbone']}),
   then scored in 5-year bins (1940–2019) by a classifier **trained on {METRICS['train_listings']:,} dated vintage
   listings ({METRICS['train_images']:,} photos)**. Photo scores are combined into one distribution.
2. **Tag reading.** On-device OCR reads the tag text: country of origin, RN/WPL numbers, union names, fiber content,
   care copy, zipper and snap brands. You can correct the text or tick things OCR can't see, like a missing care label.
3. **Constraint engine.** Every match maps to a rule in a hand-built knowledge base (FTC care-label rule, Textile Fiber Act,
   RN registration ranges, union merger dates, dissolved countries, and more). *Hard bounds* clip impossible years and
   *soft anchors* reweight likely ones. Conflicting low-confidence rules are dropped first.
4. **Range selection.** The narrowest 5-, 10- or 15-year window holding enough probability, with the threshold calibrated
   on held-out listings so the range contains the true year about 80% of the time. If nothing qualifies, it returns a
   20-year range and asks for more tag photos.

No third-party vision or LLM API is involved. Inference runs on our own weights and rules.

### Held-out performance (visual model alone, {t['listings']:,} test listings)
| Metric | Value |
|---|---|
| Average range (photos alone) | **{t['mean_range_width_years']:.1f} years** |
| Range contains the true year | **{t['exact_year_hit']:.1%}** (listings titled with an exact year) |
| Top decade correct | **{t['decade_accuracy']:.1%}** |
| Within ±1 decade | **{t['within_one_decade']:.1%}** |
| Per-decade recall | {recall} |

Labels come from listing titles, so they are noisy and the numbers are conservative. The rules engine adds
precision on top whenever tag text is legible.

### Limitations
Built for **American-market garments, 1940s–2010s**. "Vintage" here means made before 2006.
Pre-1940 pieces rely mostly on the rules.
This is an estimate, not an appraisal or authentication.
"""


THEME = gr.themes.Soft(primary_hue="orange", neutral_hue="stone")

with gr.Blocks(title="Vintage Garment Dater") as demo:
    with gr.Column(elem_id="title"):
        gr.Markdown("# Vintage Garment Dater\nUpload photos of a vintage piece and its tags to get a date range "
                    "and the evidence behind it.")
    with gr.Row(equal_height=False):
        with gr.Column(scale=5):
            files = gr.File(label="Photos (garment + tags, up to 8)", file_count="multiple", file_types=["image"])
            gallery = gr.Gallery(show_label=False, columns=4, height=180, visible=False)
            obs = gr.CheckboxGroup(list(C.OBSERVATIONS), label="Anything you can see that a photo might miss? (optional)")
            with gr.Accordion("Tag text (auto-read by OCR, editable)", open=False):
                tag_text = gr.Textbox(lines=5, show_label=False, placeholder="Filled in automatically after dating. Edit and re-run to correct OCR.")
                use_typed = gr.Checkbox(label="Use this text instead of re-running OCR", value=False)
            go = gr.Button("Date this garment", variant="primary", size="lg")
        with gr.Column(scale=6):
            result = gr.HTML('<div class="result-card"><div class="tagline">Result</div>'
                             '<div class="sig-note">Your estimate shows up here. No garment handy? Try an example below.</div></div>')
            dist = gr.LinePlot(x="Year", y="Probability", height=220, label="Probability by year (after rules)")
            sig_table = gr.Dataframe(label="All signals found", interactive=False, wrap=True)
    examples_dir = HERE / "examples"
    if (examples_dir / "labels.json").exists():
        labels = json.loads((examples_dir / "labels.json").read_text(encoding="utf-8"))
        gr.Examples(
            examples=[[sorted(str(p) for p in (examples_dir / k).glob("*.jpg")), "", [], False] for k in labels],
            example_labels=list(labels.values()),
            inputs=[files, tag_text, obs, use_typed],
            outputs=[result, dist, sig_table, tag_text, use_typed],
            fn=run, run_on_click=True, cache_examples=False,
            label="Try one of my garments (click to date it)",
        )
    with gr.Accordion("About this tool", open=False):
        gr.Markdown(about_md())

    files.change(lambda f: gr.update(value=[x if isinstance(x, str) else x.name for x in (f or [])][:8], visible=bool(f)),
                 files, gallery)
    tag_text.input(lambda: gr.update(value=True), None, use_typed)
    files.change(lambda: gr.update(value=False), None, use_typed)
    go.click(run, [files, tag_text, obs, use_typed], [result, dist, sig_table, tag_text, use_typed])

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS, head=HEAD)
