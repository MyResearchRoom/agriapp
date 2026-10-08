import logging
import os

import httpx


logger = logging.getLogger("product_search")
PRODUCT_SEARCH_URL = "https://api.openwebninja.com/realtime-product-search/search"


def _first_value(item: dict, *keys: str) -> str | None:
    for key in keys:
        value = item.get(key)
        if isinstance(value, (str, int, float)) and str(value).strip():
            return str(value).strip()
    return None


async def search_products(query: str, max_results: int = 5) -> list[dict]:
    """Return only products and fields supplied by OpenWeb Ninja."""
    if not query or not query.strip():
        return []
    api_key = os.getenv("OPENWEBNINJA_API_KEY", "").strip()
    if not api_key:
        logger.info("OpenWeb Ninja product search is not configured.")
        return []

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            response = await client.get(
                PRODUCT_SEARCH_URL,
                params={"query": query.strip(), "limit": min(max_results, 10)},
                headers={"x-api-key": api_key},
            )
            if response.status_code != 200:
                logger.warning(
                    "OpenWeb Ninja product search returned HTTP %s.",
                    response.status_code,
                )
                return []
            payload = response.json()
    except Exception as error:
        logger.warning("OpenWeb Ninja product search failed: %s", error)
        return []

    if isinstance(payload, dict):
        candidates = payload.get("products") or payload.get("shopping_results") or payload.get("results")
    else:
        candidates = payload
    if not isinstance(candidates, list):
        return []

    products = []
    for item in candidates[:max(0, min(max_results, 10))]:
        if not isinstance(item, dict):
            continue
        title = _first_value(item, "title", "name", "product_title")
        product_url = _first_value(item, "product_url", "link", "url")
        if not title or not product_url or not product_url.startswith(("https://", "http://")):
            continue
        products.append(
            {
                "title": title,
                "price": _first_value(item, "price", "extracted_price"),
                "image_url": _first_value(item, "image_url", "thumbnail", "image"),
                "retailer": _first_value(item, "retailer", "source", "merchant"),
                "product_url": product_url,
            }
        )
    return products
