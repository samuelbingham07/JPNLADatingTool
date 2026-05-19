import re
from dataclasses import dataclass


@dataclass
class DateLabel:
    start: int
    end: int
    confidence: str  # "high", "medium", "low"
    raw: str         # the matched string from the title


_DECADE_MAP = {
    "20s": (1920, 1929), "1920s": (1920, 1929),
    "30s": (1930, 1939), "1930s": (1930, 1939),
    "40s": (1940, 1949), "1940s": (1940, 1949),
    "50s": (1950, 1959), "1950s": (1950, 1959),
    "60s": (1960, 1969), "1960s": (1960, 1969),
    "70s": (1970, 1979), "1970s": (1970, 1979),
    "80s": (1980, 1989), "1980s": (1980, 1989),
    "90s": (1990, 1999), "1990s": (1990, 1999),
    "00s": (2000, 2009), "2000s": (2000, 2009),
    "y2k": (1998, 2004),
}

_EARLY_MID_LATE = {
    "early": (0, 4),
    "mid":   (3, 7),
    "late":  (6, 9),
}

_YEAR_RANGE_RE = re.compile(
    r"\b(1[89]\d{2}|20[012]\d)\s*[-–to]+\s*(1[89]\d{2}|20[012]\d)\b"
)
_CIRCA_RE = re.compile(
    r"\b(?:circa|c\.?|ca\.?)\s*(1[89]\d{2}|20[012]\d)\b", re.IGNORECASE
)
_SINGLE_YEAR_RE = re.compile(r"\b(1[89]\d{2}|20[012]\d)\b")
_DECADE_RE = re.compile(
    r"\b(early|mid|late)[-\s]?((?:19|20)?\d0s)\b|\b((?:19|20)?\d0s)\b",
    re.IGNORECASE,
)


def extract(title: str) -> DateLabel | None:
    title_lower = title.lower()

    # Explicit year range: "1965-1970"
    m = _YEAR_RANGE_RE.search(title)
    if m:
        start, end = int(m.group(1)), int(m.group(2))
        if 1900 <= start <= end <= 2025:
            return DateLabel(start, end, "high", m.group(0))

    # Circa year: "circa 1968"
    m = _CIRCA_RE.search(title)
    if m:
        year = int(m.group(1))
        return DateLabel(year - 2, year + 2, "medium", m.group(0))

    # Qualified decade: "early 60s", "late 1970s"
    m = _DECADE_RE.search(title_lower)
    if m:
        qualifier = m.group(1)
        decade_str = m.group(2) or m.group(3)
        decade_str = decade_str.lower().strip()

        if len(decade_str) == 3:  # "60s" → "1960s"
            prefix = "19" if decade_str[0] in "234567890" else "20"
            decade_str = prefix + decade_str

        if decade_str in _DECADE_MAP:
            base_start, base_end = _DECADE_MAP[decade_str]
            if qualifier and qualifier.lower() in _EARLY_MID_LATE:
                offset_start, offset_end = _EARLY_MID_LATE[qualifier.lower()]
                return DateLabel(
                    base_start + offset_start,
                    base_start + offset_end,
                    "medium",
                    m.group(0),
                )
            return DateLabel(base_start, base_end, "medium", m.group(0))

    # Single year: "1967 denim jacket"
    m = _SINGLE_YEAR_RE.search(title)
    if m:
        year = int(m.group(1))
        if 1900 <= year <= 2025:
            return DateLabel(year, year, "medium", m.group(0))

    return None
