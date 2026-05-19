from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"

# Search queries to run. These are designed to surface dated sold listings with tag photos.
# Add or remove queries as you learn which terms yield better-labeled results.
SEARCH_QUERIES = [
    "vintage american workwear jacket",
    "vintage denim jacket usa made",
    "vintage flannel shirt made in usa",
    "vintage t-shirt single stitch",
    "vintage western shirt pearl snap",
    "vintage chore coat usa",
    "vintage sweatshirt raglan",
    "vintage military field jacket",
    "vintage levi's denim",
    "vintage wrangler usa",
]

MAX_PAGES_PER_QUERY = 10   # 50 results per page = 500 listings per query
DELAY_BETWEEN_REQUESTS = 2  # seconds — be respectful
MAX_IMAGES_PER_LISTING = 8  # eBay listings often have many images; cap to save disk space
