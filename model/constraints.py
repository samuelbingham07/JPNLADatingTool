"""Constraint engine: turns tag text + user observations into knowledge-base rules, then
clips/reweights a year-level probability distribution and picks a decade-split range."""

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from knowledge.rn_lookup import lookup as rn_lookup  # noqa: E402

KB = ROOT / "knowledge"
RULES = json.loads((KB / "rules.json").read_text(encoding="utf-8"))
UNIONS = json.loads((KB / "union_labels.json").read_text(encoding="utf-8"))

YEARS = np.arange(1900, 2026)
SOFT_WEIGHT = {"high": 4.0, "medium": 2.5, "low": 1.5}
CONFIDENT_MASS = 0.55


@dataclass
class Signal:
    key: str
    label: str
    rule: dict
    evidence: str
    source: str  # "tag text" | "you"

    @property
    def is_hard(self):
        return self.rule["type"] == "hard_bound"

    @property
    def span(self):
        r = self.rule
        if self.is_hard:
            return r.get("earliest"), r.get("latest")
        return r.get("likely_start"), r.get("likely_end")

    def describe_span(self):
        lo, hi = self.span
        if lo and hi:
            return f"{lo}–{hi}"
        return f"no earlier than {lo}" if lo else f"no later than {hi}"


def _rule(*path):
    node = RULES
    for p in path:
        node = node[p]
    return node


COUNTRY_ALIASES = {
    "China": r"china|p\.?r\.?c\.?",
    "Hong Kong": r"hong\s*kong|h\.?k\.?",
    "Korea": r"(?:south\s+)?korea",
    "Taiwan": r"taiwan(?:\s+r\.?o\.?c\.?)?",
    "USA": r"u\.?\s?s\.?\s?a\.?|united\s+states(?:\s+of\s+america)?|america",
    "Mexico": r"mexico|m[eé]xico",
    "El Salvador": r"el\s+salvador",
    "Honduras": r"honduras",
    "Guatemala": r"guatemala",
    "Bangladesh": r"bangladesh",
    "Japan": r"japan",
    "Philippines": r"(?:the\s+)?philippines",
    "India": r"india",
    "Sri_Lanka": r"sri\s+lanka",
    "Vietnam": r"viet\s*nam",
    "Cambodia": r"cambodia",
    "Indonesia": r"indonesia",
    "Romania": r"romania",
}
ANYWHERE_COUNTRIES = {
    "West Germany": r"west\s+germany|w\.\s*germany",
    "East Germany": r"east\s+germany|\bgdr\b|\bddr\b",
    "USSR": r"u\.?s\.?s\.?r\.?|soviet\s+union|\bcccp\b",
    "Yugoslavia": r"yugoslavia|jugoslavija",
    "Czechoslovakia": r"czechoslovakia",
}

TEXT_RULES = [
    (r"\b\d{1,3}\s?%", ("label_text_patterns", "percent_symbol_fiber"), "Percent fiber content"),
    (r"polyester|poly\s*/|\bpoly\b", ("label_text_patterns", "polyester_blend"), "Polyester listed"),
    (r"dacron", ("label_text_patterns", "dacron_label"), "Dacron trade name"),
    (r"\bnylon\b", ("label_text_patterns", "nylon_label"), "Nylon listed"),
    (r"qiana", ("label_text_patterns", "qiana"), "Qiana nylon"),
    (r"spandex|lycra", ("label_text_patterns", "spandex_lycra"), "Spandex/Lycra listed"),
    (r"sanforized[\s-]*plus", ("label_text_patterns", "sanforized_plus"), "Sanforized-Plus"),
    (r"sanforized(?![\s-]*plus)", ("label_text_patterns", "sanforized"), "Sanforized"),
    (r"wash\s*['’]?\s*(?:n|and|&)\s*['’]?\s*wear", ("label_text_patterns", "wash_n_wear"), "Wash 'n Wear copy"),
    (r"drip[\s-]?dry", ("label_text_patterns", "drip_dry"), "Drip Dry copy"),
    (r"assembled\s+in\s+(?:the\s+)?u\.?s\.?a", ("label_text_patterns", "assembled_in_usa"), "Assembled in USA"),
    (r"woolmark", ("woolmark_symbol", "woolmark_present"), "Woolmark"),
    (r"wool\s*blend\s*mark|woolblend", ("woolmark_symbol", "woolblend_mark"), "Woolblend mark"),
    (r"machine\s+wash|tumble\s+dry|dry\s+clean|wash\s+(?:warm|cold|separately)|do\s+not\s+bleach",
     ("care_label", "present"), "Care instructions present"),
    (r"hand\s+wash\s+cold", ("care_label_text_patterns", "hand_wash_cold"), "'Hand wash cold'"),
    (r"\bdsa[\s-]?\d{2,3}", ("military_contract_label", "dsa_contract_prefix"), "DSA contract number"),
    (r"\bdla[\s-]?\d{2,3}", ("military_contract_label", "dla_contract_prefix"), "DLA contract number"),
    (r"\btalon\b.*lightning|lightning.*\btalon\b", ("zipper_hardware", "talon_lightning"), "Talon Lightning zipper"),
    (r"\btalon\b", ("zipper_hardware", "talon"), "Talon zipper"),
    (r"\bconmar\b", ("zipper_hardware", "conmar"), "Conmar zipper"),
    (r"\bykk\b", ("zipper_hardware", "ykk"), "YKK zipper"),
    (r"\bgripper\b|\bscovill\b", ("snap_hardware", "gripper_scovill"), "Gripper/Scovill hardware"),
]

UNION_PATTERNS = [
    ("ILGWU", r"i\.?\s?l\.?\s?g\.?\s?w\.?\s?u|ladies['’]?\s+garment\s+workers"),
    ("ACTWU", r"a\.?c\.?t\.?w\.?u|amalgamated\s+clothing\s+(?:and|&)\s+textile"),
    ("ACWA", r"\ba\.?c\.?w\.?a\b|amalgamated\s+clothing\s+workers"),
    ("TWU", r"textile\s+workers\s+union"),
    ("UNITE_HERE", r"unite\s+here"),
    ("UNITE", r"\bunite\b(?!\s+here)"),
]

# Observations OCR can't make (absence of a label, construction, print method) — the user ticks these.
OBSERVATIONS = {
    "No care label anywhere": ("care_label", "absent"),
    "No fiber content listed": ("fiber_content_label", "absent"),
    "Care label uses symbols only, no words": ("care_label_text_patterns", "symbols_only_no_words"),
    "Brand screenprinted on collar (no sewn tag)": ("tag_construction", "screenprinted_collar"),
    "Single stitch hem / sleeves": ("construction_details", "single_stitch_hem"),
    "Selvedge denim": ("construction_details", "selvedge_denim"),
    "Chain stitch outseam": ("construction_details", "chain_stitch_outseam"),
    "Paper tag / paper inner label": ("tag_construction", "paper_tag"),
    "Brand/size only, no country or fiber info": ("label_text_patterns", "size_without_state_country"),
    "ILGWU red-white-blue label": ("union", "ILGWU", "1974_1995_red_white_blue"),
    "ILGWU scalloped circle w/ needle & thread": ("union", "ILGWU", "1955_1963_scallop_needle"),
    "ILGWU dark-circle label (not red-white-blue)": ("union", "ILGWU", "1963_1974_dark_circle"),
}


# Brands that kept printing their WPL number long after RNs replaced them in 1971
WPL_KEPT_AFTER_1971 = {423}  # Levi Strauss & Co.


def _union_rule(name, sub=None):
    u = UNIONS[name]
    return u["sub_designs"][sub] if sub else u["existence_bound"]


def extract_signals(text: str, observations=()) -> list[Signal]:
    t = " " + re.sub(r"\s+", " ", text.lower()) + " "
    out = []

    def add(key, label, rule, evidence, source="tag text"):
        if rule.get("earliest") is None and rule.get("latest") is None and rule["type"] == "hard_bound":
            return
        if all(s.key != key for s in out):
            out.append(Signal(key, label, rule, evidence, source))

    for country, pat in COUNTRY_ALIASES.items():
        m = re.search(rf"(?:made|hecho|fabriqu[eé]|manufactured)\s+(?:in|en)\s+(?:the\s+)?(?:{pat})\b", t)
        if m and not re.search(r"assembled|imported\s+(?:fabric|materials)", t[max(0, m.start() - 40):m.end() + 40]):
            add(f"country:{country}", f"Made in {country.replace('_', ' ')}",
                _rule("country_of_origin", country), m.group(0).strip())
    for country, pat in ANYWHERE_COUNTRIES.items():
        m = re.search(pat, t)
        if m:
            add(f"country:{country}", f"{country} on label", _rule("country_of_origin", country), m.group(0).strip())

    m = re.search(r"\bw\.?\s?p\.?\s?l\.?\s*(?:no\.?|#)?\s*(\d{3,6})", t)
    if m and int(m.group(1)) not in WPL_KEPT_AFTER_1971:
        add("wpl", f"WPL {m.group(1)}", _rule("rn_number", "wpl_present"), m.group(0).strip())
    m = re.search(r"\br\.?\s?n\.?\s*(?:no\.?|#)?\s*(\d{3,6})\b", t)
    if m:
        rn = int(m.group(1))
        info = rn_lookup(rn)
        if info.get("earliest_year"):
            who = f" ({info['company']})" if info.get("company") else ""
            rule = {"type": "hard_bound", "earliest": info["earliest_year"], "confidence": "high",
                    "notes": f"RN {rn}{who} registered ~{info['earliest_year']}; a garment can't predate its RN. "
                             f"{info.get('note', '')}".strip()}
            add("rn", f"RN {rn}{who}", rule, m.group(0).strip())
        else:
            add("rn", f"RN {rn}", _rule("rn_number", "rn_present"), m.group(0).strip())

    for name, pat in UNION_PATTERNS:
        m = re.search(pat, t)
        if m:
            add(f"union:{name}", f"{name} union label", _union_rule(name), m.group(0).strip())

    for pat, path, label in TEXT_RULES:
        m = re.search(pat, t)
        if m:
            add(path[1], label, _rule(*path), m.group(0).strip())

    if "talon_lightning" in {s.key for s in out}:
        out = [s for s in out if s.key != "talon"]
    if "sanforized_plus" in {s.key for s in out}:
        out = [s for s in out if s.key != "sanforized"]

    for obs in observations:
        path = OBSERVATIONS[obs]
        rule = _union_rule(path[1], path[2]) if path[0] == "union" else _rule(*path)
        add(f"obs:{obs}", obs, rule, "you observed this", source="you")
    return out


def _bounds(signals):
    lo, hi = YEARS[0], YEARS[-1]
    for s in signals:
        if s.is_hard:
            e, l = s.span
            lo, hi = max(lo, e or lo), min(hi, l or hi)
    return lo, hi


def apply(prior: np.ndarray, signals: list[Signal]):
    """prior: probability per year in YEARS. Returns (posterior, active_signals, dropped_signals, (lo, hi))."""
    hard = [s for s in signals if s.is_hard]
    dropped = []
    lo, hi = _bounds(hard)
    if lo > hi:
        for level in ("low", "medium"):
            dropped += [s for s in hard if s.rule["confidence"] == level]
            hard = [s for s in hard if s.rule["confidence"] != level]
            lo, hi = _bounds(hard)
            if lo <= hi:
                break
        if lo > hi:
            dropped += hard
            hard = []
            lo, hi = YEARS[0], YEARS[-1]

    post = prior * ((YEARS >= lo) & (YEARS <= hi))
    soft = [s for s in signals if not s.is_hard]
    for s in soft:
        a, b = s.span
        post = post * np.where((YEARS >= (a or 0)) & (YEARS <= (b or 9999)), SOFT_WEIGHT[s.rule["confidence"]], 1.0)
    if post.sum() == 0:
        post = ((YEARS >= lo) & (YEARS <= hi)).astype(float)
    return post / post.sum(), hard + soft, dropped, (int(lo), int(hi))


def _candidates(lo, hi):
    for d in range(1900, 2030, 10):
        for a, b, kind in ((d, d + 4, "early"), (d + 3, d + 7, "mid"), (d + 6, d + 9, "late"),
                           (d, d + 9, "full"), (d, d + 19, "wide")):
            a, b = max(a, lo), min(b, hi)
            if a <= b:
                yield a, b, kind


def pick_range(post: np.ndarray, bounds):
    lo, hi = bounds
    if hi - lo <= 5:
        return {"start": lo, "end": hi, "confidence": 1.0, "kind": "tell"}
    cum = np.concatenate([[0], np.cumsum(post)])

    def mass(a, b):
        return cum[b - YEARS[0] + 1] - cum[a - YEARS[0]]

    scored = [(a, b, kind, mass(a, b)) for a, b, kind in _candidates(lo, hi)]
    for kinds in (("early", "mid", "late"), ("full",)):
        good = [c for c in scored if c[2] in kinds and c[3] >= CONFIDENT_MASS]
        if good:
            a, b, kind, m = max(good, key=lambda c: c[3])
            return {"start": a, "end": b, "confidence": float(m), "kind": kind}
    if hi - lo <= 15:
        return {"start": lo, "end": hi, "confidence": 1.0, "kind": "tell"}
    a, b, kind, m = max((c for c in scored if c[2] == "wide"), key=lambda c: c[3])
    return {"start": a, "end": b, "confidence": float(m), "kind": "low"}
