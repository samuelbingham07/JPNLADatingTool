"""
eBay sold-listing scraper.

Usage:
    python -m scraper.ebay_scraper
    python -m scraper.ebay_scraper --query "vintage levi's" --pages 5
"""

import argparse
import asyncio
import json
import re
import time
from pathlib import Path
from urllib.parse import quote_plus

import httpx
from playwright.async_api import async_playwright, Page
from tqdm import tqdm

from scraper.config import DATA_DIR, DELAY_BETWEEN_REQUESTS, MAX_IMAGES_PER_LISTING, SEARCH_QUERIES, MAX_PAGES_PER_QUERY
from scraper.label_extractor import extract as extract_label


EBAY_SEARCH_URL = (
    "https://www.ebay.com/sch/i.html"
    "?_nkw={query}"
    "&LH_Sold=1"       # sold listings only
    "&LH_Complete=1"   # completed listings
    "&_sop=13"         # sort by most recently sold
    "&_pgn={page}"
    "&_ipg=50"         # 50 results per page
)


async def get_listing_urls(page: Page, query: str, max_pages: int) -> list[dict]:
    listings = []

    for page_num in range(1, max_pages + 1):
        url = EBAY_SEARCH_URL.format(query=quote_plus(query), page=page_num)
        await page.goto(url, wait_until="domcontentloaded")
        await asyncio.sleep(DELAY_BETWEEN_REQUESTS)

        items = await page.query_selector_all(".s-item__wrapper")
        if not items:
            break

        for item in items:
            title_el = await item.query_selector(".s-item__title")
            link_el = await item.query_selector(".s-item__link")
            price_el = await item.query_selector(".s-item__price")

            if not title_el or not link_el:
                continue

            title = await title_el.inner_text()
            href = await link_el.get_attribute("href")
            price = await price_el.inner_text() if price_el else ""

            if not href or "itm" not in href:
                continue

            item_id_match = re.search(r"/itm/(\d+)", href)
            if not item_id_match:
                continue

            label = extract_label(title)
            listings.append({
                "item_id": item_id_match.group(1),
                "title": title,
                "price": price,
                "url": href.split("?")[0],
                "label": {
                    "start": label.start,
                    "end": label.end,
                    "confidence": label.confidence,
                    "raw": label.raw,
                } if label else None,
            })

        if len(items) < 10:
            break

    return listings


async def scrape_listing(page: Page, listing: dict, client: httpx.AsyncClient) -> bool:
    item_dir = DATA_DIR / listing["item_id"]
    metadata_path = item_dir / "metadata.json"

    if metadata_path.exists():
        return False  # already scraped

    await page.goto(listing["url"], wait_until="domcontentloaded")
    await asyncio.sleep(DELAY_BETWEEN_REQUESTS)

    # Collect image URLs from the listing gallery
    img_elements = await page.query_selector_all("img[src*='i.ebayimg.com']")
    image_urls = []
    seen = set()

    for el in img_elements:
        src = await el.get_attribute("src") or await el.get_attribute("data-src") or ""
        # Upgrade to full-size eBay image URL
        src = re.sub(r"s-l\d+", "s-l1600", src)
        if src and src not in seen:
            seen.add(src)
            image_urls.append(src)

    image_urls = image_urls[:MAX_IMAGES_PER_LISTING]

    if not image_urls:
        return False

    item_dir.mkdir(parents=True, exist_ok=True)
    images_dir = item_dir / "images"
    images_dir.mkdir(exist_ok=True)

    saved_images = []
    for idx, img_url in enumerate(image_urls):
        try:
            resp = await client.get(img_url, timeout=15)
            if resp.status_code == 200:
                img_path = images_dir / f"{idx}.jpg"
                img_path.write_bytes(resp.content)
                saved_images.append(str(img_path.relative_to(DATA_DIR)))
        except Exception:
            continue

    if not saved_images:
        item_dir.rmdir()
        return False

    metadata = {**listing, "images": saved_images, "scraped_at": time.time()}
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return True


async def run(queries: list[str], max_pages: int):
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        )
        page = await context.new_page()

        async with httpx.AsyncClient(follow_redirects=True) as client:
            for query in queries:
                print(f"\nSearching: {query}")
                listings = await get_listing_urls(page, query, max_pages)

                labeled = [l for l in listings if l["label"] is not None]
                print(f"  Found {len(listings)} listings, {len(labeled)} with date labels")

                for listing in tqdm(labeled, desc="  Scraping"):
                    await scrape_listing(page, listing, client)

        await browser.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default=None, help="Single search query (overrides config)")
    parser.add_argument("--pages", type=int, default=MAX_PAGES_PER_QUERY)
    args = parser.parse_args()

    queries = [args.query] if args.query else SEARCH_QUERIES
    asyncio.run(run(queries, args.pages))


if __name__ == "__main__":
    main()
