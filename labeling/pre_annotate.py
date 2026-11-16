"""
Infer likely annotation values from listing title and metadata.
Used to pre-fill Label Studio tasks so annotators only need to correct.
"""

import re


def _lower(title: str) -> str:
    return title.lower()


def infer_garment_type(title: str) -> str | None:
    t = _lower(title)
    if any(x in t for x in ["t-shirt", "tshirt", "t shirt", "tee shirt", "tee"]):
        return "tshirt"
    if any(x in t for x in ["hoodie", "hooded sweatshirt", "hooded sweater"]):
        return "hoodie"
    if any(x in t for x in ["sweatshirt", "crewneck", "crew neck", "reverse weave"]):
        return "sweatshirt_crewneck"
    if any(x in t for x in ["flannel", "chamois"]):
        return "flannel_shirt"
    if any(x in t for x in ["western shirt", "pearl snap", "snap shirt", "cowboy shirt"]):
        return "western_shirt"
    if any(x in t for x in ["polo shirt", "polo"]) and "ralph" not in t:
        return "polo_shirt"
    if any(x in t for x in ["dress shirt", "oxford shirt", "button down", "button-down"]):
        return "dress_shirt"
    if any(x in t for x in ["denim jacket", "jean jacket", "trucker jacket", "chore jacket"]):
        return "denim_jacket"
    if any(x in t for x in ["chore coat", "chore jacket", "work coat"]):
        return "chore_coat"
    if any(x in t for x in ["work jacket", "work shirt", "work wear", "workwear"]):
        return "work_jacket"
    if any(x in t for x in ["varsity", "letterman"]):
        return "varsity_jacket"
    if any(x in t for x in ["coat", "parka", "anorak", "puffer", "down jacket", "windbreaker", "jacket"]):
        return "outerwear_coat"
    if any(x in t for x in ["jeans", "denim pant", "501", "505", "lee rider", "wrangler"]) and "jacket" not in t:
        return "jeans"
    if any(x in t for x in ["trouser", "pants", "slacks", "chino", "khaki", "corduroy pant"]):
        return "trousers_pants"
    if any(x in t for x in ["overalls", "bib overall", "bib-overall"]):
        return "overalls"
    if any(x in t for x in ["coverall", "cover-all", "jumpsuit"]):
        return "coveralls"
    if any(x in t for x in ["dress", "skirt", "blouse"]):
        return "dress_skirt"
    if any(x in t for x in ["blazer", "sport coat", "sportcoat", "suit jacket"]):
        return "blazer_sportcoat"
    if any(x in t for x in ["sweater", "cardigan", "pullover", "knit"]):
        return "sweater_knit"
    return None


def infer_materials(title: str) -> list[str]:
    t = _lower(title)
    materials = []
    if any(x in t for x in ["denim", "jean", "501", "505"]):
        materials.append("denim")
    if any(x in t for x in ["flannel"]):
        materials.append("flannel")
    if any(x in t for x in ["wool", "woolrich", "pendleton", "worsted"]):
        materials.append("wool")
    if any(x in t for x in ["corduroy", "cord"]):
        materials.append("corduroy")
    if any(x in t for x in ["nylon", "windbreaker", "anorak"]):
        materials.append("nylon")
    if any(x in t for x in ["canvas", "duck cloth", "chore"]):
        materials.append("canvas_duck")
    if any(x in t for x in ["leather", "suede"]):
        materials.append("leather")
    if any(x in t for x in ["fleece", "polar fleece"]):
        materials.append("fleece")
    if any(x in t for x in ["satin", "rayon", "silky", "bowling shirt", "gabardine"]):
        materials.append("satin_rayon")
    if any(x in t for x in ["polyester", "poly", "double knit", "qiana"]):
        materials.append("polyester")
    if not materials:
        materials.append("cotton")  # safe default for most vintage
    return materials


def infer_country(title: str) -> str | None:
    t = _lower(title)
    if any(x in t for x in ["made in usa", "made in the usa", "usa made", "made in u.s.a"]):
        return "usa"
    if "made in japan" in t:
        return "japan"
    if "made in hong kong" in t:
        return "hong_kong"
    if "made in taiwan" in t:
        return "taiwan"
    if "made in korea" in t:
        return "korea"
    if "made in mexico" in t:
        return "mexico"
    if "made in china" in t:
        return "china"
    return None


def infer_brand(title: str) -> tuple[str, str]:
    """Returns (brand_visible_choice, brand_name)."""
    t = _lower(title)
    brands = [
        "levi's", "levis", "wrangler", "lee", "champion", "russell athletic",
        "nike", "adidas", "carhartt", "dickies", "pendleton", "woolrich",
        "hanes", "fruit of the loom", "screen stars", "jerzees", "gildan",
        "north face", "patagonia", "columbia", "eddie bauer", "ralph lauren",
        "polo", "lacoste", "gap", "tommy hilfiger", "nautica",
        "big mac", "walls", "key industries", "oshkosh",
        "h bar c", "rockmount", "panhandle slim",
    ]
    for brand in brands:
        if brand in t:
            # Capitalise nicely
            name = brand.title().replace("'S", "'s")
            return "yes_readable", name
    return "partial", ""


def build_prediction(task_data: dict, label: dict | None) -> dict:
    """Build a Label Studio prediction result from task data."""
    title = task_data.get("title", "")
    result = []

    def add(from_name, type_, value_dict):
        result.append({
            "from_name": from_name,
            "to_name": "listing_meta",
            "type": type_,
            "value": value_dict,
        })

    # Garment type
    gt = infer_garment_type(title)
    if gt:
        add("garment_type", "choices", {"choices": [gt]})

    # Material
    mats = infer_materials(title)
    if mats:
        add("material", "choices", {"choices": mats})

    # Country
    country = infer_country(title)
    if country:
        add("signals_country", "choices", {"choices": [country]})

    # Brand
    brand_vis, brand_name = infer_brand(title)
    add("brand_visible", "choices", {"choices": [brand_vis]})
    if brand_name:
        add("brand_name", "textarea", {"text": [brand_name]})

    # Tag visibility default
    add("tag_visibility", "choices", {"choices": ["partial_some_text"]})

    # Date from label extractor
    if label:
        if label.get("start"):
            add("date_start", "number", {"number": label["start"]})
        if label.get("end"):
            add("date_end", "number", {"number": label["end"]})
        conf = label.get("confidence", "medium")
        add("confidence", "choices", {"choices": [conf]})

    return {"result": result, "score": 0.5}
