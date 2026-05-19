# JPNLADatingTool — Claude Context

## What This Is

A proprietary AI system that dates vintage American clothing from images of garments and their tags. The shipped product is a **conversational agent** — users upload images, the agent asks follow-up questions, requests additional photos, and refines its estimate interactively. The model + constraint engine is the reasoning core underneath that agent.

This is a long-term build. The end goal is a personal tool that becomes a marketed product.

## What We Are Building

**The reasoning core:**
- A multi-input vision model trained on owned, labeled data — takes tag image(s), garment image(s), and OCR'd tag text as inputs
- A constraint engine that applies hard bounds and soft anchors from a structured knowledge base (`/knowledge`) to clip and weight the model's raw probability distribution
- Output: a predicted date range (5-year interval) + confidence + list of signals that drove the prediction

**The product layer (on top of the core):**
- Primary interaction: user uploads photo(s) → receives a 5-year date range immediately
- Secondary interaction: user can continue the conversation to ask follow-up questions, get deeper explanation, or provide more photos to refine the estimate
- Explains its reasoning in plain language, referencing specific tells
- Stores and learns from interactions over time

We are **not** building a wrapper around GPT-4o, Claude, or any other third-party vision API. The shipped product will run on our own trained weights.

## User Context

The user runs a vintage clothing business and has significant hands-on expertise dating garments by eye. They are the domain authority on this project. Do not explain vintage dating fundamentals unless asked. Treat their judgment on dating signals, label decisions, and edge cases as correct unless there is a specific technical reason to push back.

## How to Collaborate

- Be concise and direct. No preamble, no recap of what was just discussed.
- Treat this like a co-builder relationship — raise tradeoffs, flag risks, but default to moving forward.
- When proposing an approach, lead with the recommendation. Put reasoning after.
- Never suggest the "fast path" of using a vision LLM as the product. That ship has sailed.
- Do not add abstractions, error handling, or features beyond what the current phase requires.
- No comments in code unless the why is genuinely non-obvious.

## Prediction Spec

| Case | Output |
|------|--------|
| Normal | 5-year range (e.g., 1968–1973) + confidence score |
| Clear tell present | Exact year or tight range (e.g., post-1971 care label = no earlier than 1972) |
| Low confidence | Wide range + request for additional images |

## Conversational Agent Behavior

- **Primary flow:** user uploads photo(s) → agent returns a date range immediately. This is the core product loop. It must be fast and require no conversation.
- **Secondary flow:** user can then ask questions, provide more photos, or request explanation. The agent refines its estimate and answers naturally.
- Always surfaces the top 2-3 signals that drove the prediction, without the user having to ask
- Flags when a constraint engine rule was the deciding factor vs. the model

## Key Domain Anchors

These are hard constraints that override model predictions when triggered. Full ruleset lives in `/knowledge/rules.json`.

- **Care labels:** Mandatory in the US after July 3, 1971 (FTC ruling). No care label = pre-1972.
- **Country of origin:** Required after the Textile Fiber Products Identification Act (1960). China = post-1995 hard bound. Made in USA = strong prior for pre-2005.
- **Union labels:** ILGWU active until 1995 merger into UNITE. Specific designs map to narrow date ranges.
- **RN/WPL numbers:** Traceable to brand and registration date via FTC database — hard earliest-date anchor.
- **Fiber content:** Required after 1960.
- **Screenprinted collar tag:** Strong signal for post-2005.
- **Dissolved countries on labels:** West Germany / East Germany (→ pre-1990), USSR (→ pre-1991), Yugoslavia (→ pre-1992).

*This list will grow. The knowledge base in `/knowledge` is the authoritative source.*

## Project Structure

```
/knowledge      — structured signal database (rules.json, union_labels.json, brand_tags.json)
/scraper        — data collection (eBay and other sources)
/data           — raw and labeled image datasets
/labeling       — Label Studio config and labeling guides
/model          — architecture, training scripts, experiment configs
/agent          — conversational agent layer (dialogue management, state, follow-up logic)
/app            — user-facing product (frontend + inference API)
```

## Current Phase

**Phase 1 — Data pipeline.** Everything is blocked on labeled data. Priority order:
1. eBay scraper (dated listings with tag photos)
2. Knowledge base completion (`brand_tags.json`, `union_labels.json`)
3. Label Studio setup

Model and agent architecture decisions come after the first 2,000 labeled examples exist.

## Stack

- Python 3.11+
- PyTorch (model training)
- Weights & Biases (experiment tracking)
- Label Studio (labeling)
- FastAPI (inference API)
- PostgreSQL (metadata + interaction history)
- Local GPU for training
