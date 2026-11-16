import os
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"

import json

# eBay API keys: set EBAY_APP_ID / EBAY_CERT_ID env vars, or put them in
# scraper/ebay_keys.json (gitignored) as {"EBAY_APP_ID": "...", "EBAY_CERT_ID": "..."}
_KEYS_FILE = Path(__file__).parent / "ebay_keys.json"
_keys = json.loads(_KEYS_FILE.read_text()) if _KEYS_FILE.exists() else {}
EBAY_APP_ID  = os.environ.get("EBAY_APP_ID",  _keys.get("EBAY_APP_ID", ""))
EBAY_CERT_ID = os.environ.get("EBAY_CERT_ID", _keys.get("EBAY_CERT_ID", ""))

# Search queries — designed to surface dated sold listings with tag photos.
# Lead with decade/era terms so the title label extractor gets clean hits.
SEARCH_QUERIES = [
    # Denim / jeans
    "vintage 1960s workwear jacket usa made",
    "vintage 1970s denim jacket made in usa",
    "vintage levi's 501 big e",
    "vintage wrangler blue bell denim",
    "vintage 1960s lee riders denim",
    "vintage 1950s denim jeans usa selvedge",
    "vintage 1980s denim jacket made in usa",
    # Flannel / shirts
    "vintage 1980s flannel shirt made in usa",
    "vintage 1970s flannel shirt union made",
    "vintage 1950s western shirt pearl snap",
    "vintage 1950s rockabilly western shirt",
    "vintage 1960s pendleton wool shirt",
    "vintage 1970s pendleton wool plaid shirt",
    "vintage 1960s chamois flannel shirt usa",
    # T-shirts / sweatshirts
    "vintage 1960s t-shirt single stitch",
    "vintage 1980s screen stars t-shirt",
    "vintage 1970s champion reverse weave",
    "vintage 1980s sweatshirt champion usa",
    "vintage 1970s russell athletic",
    "vintage 1980s fruit of the loom t-shirt",
    "vintage 1990s single stitch t-shirt usa",
    "vintage 1980s hanes beefy t shirt usa",
    "vintage 1970s raglan baseball t-shirt",
    # Workwear
    "vintage 1970s chore coat usa",
    "vintage 1960s military field jacket",
    "vintage carhartt detroit jacket made usa",
    "vintage 1970s dickies workwear usa",
    "vintage 1950s big mac jcpenney workwear",
    "vintage 1960s oshkosh bib overalls usa",
    "vintage 1970s walls workwear usa",
    "vintage 1960s key industries overalls",
    "vintage 1940s work coat union made usa",
    # Outerwear
    "vintage 1950s wool coat union made",
    "vintage 1970s woolrich hunting jacket",
    "vintage 1960s wool mackinaw coat usa",
    "vintage 1980s north face jacket made usa",
    "vintage 1970s columbia sportswear jacket",
    # Sportswear / athletic
    "vintage 1980s nike made in usa",
    "vintage 1980s adidas tracksuit",
    "vintage 1970s gym shorts usa",
    "vintage 1980s baseball jersey usa",
    "vintage 1970s nylon windbreaker usa",
    # Trousers / pants
    "vintage 1970s corduroy pants usa made",
    "vintage 1960s chino trousers usa",
    "vintage 1980s parachute pants",
    # Knit / sweaters
    "vintage 1960s wool sweater union made",
    "vintage 1970s acrylic sweater usa",
    # Union label specific
    "vintage 1960s dress blouse ilgwu union label",
    "vintage 1950s jacket acwa union label",
    "vintage 1970s women's dress union made",

    # More denim / jeans
    "vintage 1960s levi's big e jeans",
    "vintage 1970s levi's orange tab jeans",
    "vintage 1950s wrangler cowboy denim",
    "vintage 1980s lee jeans made in usa",
    "vintage 1960s sanforized denim jeans",
    "vintage 1970s bell bottom jeans usa",
    "vintage 1980s jordache jeans",
    "vintage 1970s landlubber jeans",

    # More shirts / tops
    "vintage 1960s bowling shirt usa",
    "vintage 1950s gabardine shirt",
    "vintage 1970s disco shirt usa",
    "vintage 1960s madras shirt usa",
    "vintage 1980s polo shirt made in usa",
    "vintage 1970s henley shirt single stitch",
    "vintage 1960s button down oxford shirt usa",
    "vintage 1950s short sleeve shirt union made",
    "vintage 1980s rugby shirt usa",
    "vintage 1970s thermal waffle knit shirt usa",

    # More sweatshirts / fleece
    "vintage 1980s crewneck sweatshirt single stitch",
    "vintage 1970s hooded sweatshirt usa made",
    "vintage 1960s college sweatshirt usa",
    "vintage 1980s raglan sweatshirt made usa",
    "vintage 1990s champion hoodie made usa",
    "vintage 1980s russell athletic sweatshirt",
    "vintage 1970s reverse weave crewneck",

    # More jackets / outerwear
    "vintage 1960s varsity letterman jacket",
    "vintage 1970s satin baseball jacket usa",
    "vintage 1980s nylon jacket made in usa",
    "vintage 1960s corduroy jacket usa",
    "vintage 1970s denim trucker jacket",
    "vintage 1950s leather jacket usa",
    "vintage 1980s ski jacket usa",
    "vintage 1960s hunting jacket wool usa",
    "vintage 1970s coaches jacket nylon usa",
    "vintage 1960s zip up jacket usa union",

    # More workwear
    "vintage 1950s hickory stripe overalls",
    "vintage 1960s painter pants usa",
    "vintage 1970s mechanic coveralls usa",
    "vintage 1940s denim chore coat usa",
    "vintage 1950s duck canvas work jacket",
    "vintage 1960s railroad work pants",
    "vintage 1970s big smith overalls",
    "vintage 1950s pointer brand overalls",

    # Trousers / pants
    "vintage 1970s polyester pants usa",
    "vintage 1960s chino khaki pants usa",
    "vintage 1980s pleated trousers made usa",
    "vintage 1970s wide leg trousers usa",
    "vintage 1960s wool trousers union made",
    "vintage 1950s gabardine slacks usa",
    "vintage 1980s corduroy trousers usa",

    # Sportswear / athletic
    "vintage 1980s adidas sweatshirt usa",
    "vintage 1970s gym shorts made usa",
    "vintage 1980s nike swoosh t-shirt",
    "vintage 1980s starter jacket",
    "vintage 1970s athletic shorts usa single stitch",
    "vintage 1980s ringer t-shirt usa",
    "vintage 1970s baseball jersey raglan usa",

    # Women's / other
    "vintage 1960s wiggle dress union label",
    "vintage 1950s house dress ilgwu",
    "vintage 1970s polyester blouse usa",
    "vintage 1960s shift dress union made",
    "vintage 1980s women's blazer made usa",
    "vintage 1970s women's jeans usa made",

    # Specific brands
    "vintage pendleton board shirt 1960s",
    "vintage 1980s patagonia fleece made usa",
    "vintage 1970s woolrich wool jacket",
    "vintage 1960s mcgregor jacket usa",
    "vintage 1970s jantzen sweater usa",
    "vintage 1960s arrow shirt union made",
    "vintage 1970s van heusen shirt usa",
    "vintage 1950s h bar c western shirt",
    "vintage 1960s rockmount ranch wear",
    "vintage 1970s panhandle slim western",
]

MAX_PAGES_PER_QUERY = 10   # 100 results/page = 1000 listings per query
DELAY_BETWEEN_REQUESTS = 1  # seconds between Shopping API calls
MAX_IMAGES_PER_LISTING = 12
