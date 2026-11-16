"""
RN/WPL number lookup and earliest-year estimation.

Two tiers:
  1. Local cache (knowledge/rn_cache.json) — grows over time as lookups are done
  2. Range-based estimation — derives earliest registration year from the RN number itself

The FTC database (rn.ftc.gov) requires login for searches, so individual lookups
require manual entry or a logged-in session. Use `add_to_cache()` to record
results as you look them up manually.

Usage:
    from knowledge.rn_lookup import lookup

    result = lookup(13765)
    # -> {"rn": 13765, "company": "Fruit of the Loom", "earliest_year": 1959,
    #     "source": "cache"}

    result = lookup(54321)
    # -> {"rn": 54321, "earliest_year": 1974, "source": "range_estimate",
    #     "year_range": [1973, 1975]}
"""

import json
from pathlib import Path

_CACHE_FILE = Path(__file__).parent / "rn_cache.json"

# ---------------------------------------------------------------------------
# Range table — derived from:
#   - RN 00101-04086: issued 1952-1959 (Fur Products Labeling Act era)
#   - RN 13670+: new series starting ~1959, avg ~2635/year
#   Sources: FTC FAQ, Vintage Fashion Guild research, freshpickvintage.blogspot.com
# ---------------------------------------------------------------------------
_RN_RANGES = [
    # (rn_min, rn_max_exclusive, year_start, year_end)
    # Pre-1959 block
    (101,    4087,   1952, 1959),
    # Post-1959 block — computed from 13670 base, ~2635/year
    # Each entry covers one year; last entry is open-ended
    (13670,  16305,  1959, 1960),
    (16305,  18940,  1960, 1961),
    (18940,  21575,  1961, 1962),
    (21575,  24210,  1962, 1963),
    (24210,  26845,  1963, 1964),
    (26845,  29480,  1964, 1965),
    (29480,  32115,  1965, 1966),
    (32115,  34750,  1966, 1967),
    (34750,  37385,  1967, 1968),
    (37385,  40020,  1968, 1969),
    (40020,  42655,  1969, 1970),
    (42655,  45290,  1970, 1971),
    (45290,  47925,  1971, 1972),
    (47925,  50560,  1972, 1973),
    (50560,  53195,  1973, 1974),
    (53195,  55830,  1974, 1975),
    (55830,  58465,  1975, 1976),
    (58465,  61100,  1976, 1977),
    (61100,  63735,  1977, 1978),
    (63735,  66370,  1978, 1979),
    (66370,  69005,  1979, 1980),
    (69005,  71640,  1980, 1981),
    (71640,  74275,  1981, 1982),
    (74275,  76910,  1982, 1983),
    (76910,  79545,  1983, 1984),
    (79545,  82180,  1984, 1985),
    (82180,  84815,  1985, 1986),
    (84815,  87450,  1986, 1987),
    (87450,  90085,  1987, 1988),
    (90085,  92720,  1988, 1989),
    (92720,  95355,  1989, 1990),
    (95355,  97990,  1990, 1991),
    (97990,  100625, 1991, 1992),
    (100625, 103260, 1992, 1993),
    (103260, 105895, 1993, 1994),
    (105895, 108530, 1994, 1995),
    (108530, 111165, 1995, 1996),
    (111165, 113800, 1996, 1997),
    (113800, 116435, 1997, 1998),
    (116435, 119070, 1998, 1999),
    (119070, 121705, 1999, 2000),
    (121705, 999999, 2000, 2030),  # open-ended
]

# Gap between the two series (4087–13669) — numbers not issued under this system.
# An RN in this gap indicates a data anomaly; treat as unknown.
_GAP_MIN = 4087
_GAP_MAX = 13669


def _load_cache() -> dict:
    if _CACHE_FILE.exists():
        return json.loads(_CACHE_FILE.read_text())
    return {}


def _save_cache(cache: dict) -> None:
    _CACHE_FILE.write_text(json.dumps(cache, indent=2, sort_keys=True))


def _estimate_from_range(rn: int) -> dict:
    if rn < 101:
        return {"earliest_year": None, "source": "range_estimate",
                "note": "RN below known range"}
    if _GAP_MIN <= rn <= _GAP_MAX:
        return {"earliest_year": None, "source": "range_estimate",
                "note": "RN in unissued gap between pre/post-1959 series"}

    for lo, hi, y_start, y_end in _RN_RANGES:
        if lo <= rn < hi:
            return {
                "earliest_year": y_start,
                "year_range": [y_start, y_end],
                "source": "range_estimate",
                "note": f"Estimated from RN range {lo}–{hi-1} ≈ {y_start}–{y_end}",
            }

    return {"earliest_year": None, "source": "range_estimate",
            "note": "RN outside known ranges"}


def lookup(rn: int) -> dict:
    """
    Return dating info for an RN number.

    Returns dict with keys:
        rn            : int
        company       : str | None
        earliest_year : int | None  — hard lower bound for garment manufacture
        source        : "cache" | "range_estimate"
        note          : str (optional)
    """
    cache = _load_cache()
    key = str(rn)

    if key in cache:
        entry = cache[key].copy()
        entry["source"] = "cache"
        return {"rn": rn, **entry}

    result = _estimate_from_range(rn)
    return {"rn": rn, **result}


def add_to_cache(rn: int, company: str, earliest_year: int,
                 notes: str = "") -> None:
    """
    Manually record a verified RN lookup result.
    Call this after looking up a number in the FTC database.
    """
    cache = _load_cache()
    cache[str(rn)] = {
        "company": company,
        "earliest_year": earliest_year,
        "notes": notes,
    }
    _save_cache(cache)
    print(f"Cached RN {rn}: {company} ({earliest_year})")


# ---------------------------------------------------------------------------
# Pre-seeded cache for well-known RN numbers
# ---------------------------------------------------------------------------
_KNOWN = {
    13765: ("Fruit of the Loom / Screen Stars", 1959),
    15763: ("Hanes", 1960),
}


def _seed_cache() -> None:
    cache = _load_cache()
    changed = False
    for rn, (company, year) in _KNOWN.items():
        if str(rn) not in cache:
            cache[str(rn)] = {"company": company, "earliest_year": year, "notes": "pre-seeded"}
            changed = True
    if changed:
        _save_cache(cache)


_seed_cache()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m knowledge.rn_lookup <RN_NUMBER>")
        print("       python -m knowledge.rn_lookup add <RN> <COMPANY> <YEAR>")
        sys.exit(1)

    if sys.argv[1] == "add":
        add_to_cache(int(sys.argv[2]), sys.argv[3], int(sys.argv[4]))
    else:
        result = lookup(int(sys.argv[1]))
        print(json.dumps(result, indent=2))
