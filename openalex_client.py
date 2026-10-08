import httpx
import logging
import os

OPENALEX_BASE_URL = "https://api.openalex.org/works"
OPENALEX_CONTACT_EMAIL = "s.gurme@wesolutize.com"  # Courtesy contact for OpenAlex; authentication uses optional OPENALEX_API_KEY.
logger = logging.getLogger("openalex")


def _reconstruct_abstract(inverted_index: dict | None) -> str:
    """OpenAlex stores abstracts as a word->positions inverted index, not plain text. Rebuild it."""
    if not inverted_index or not isinstance(inverted_index, dict):
        return ""
    position_map = {}
    for word, positions in inverted_index.items():
        if isinstance(positions, list):
            for pos in positions:
                position_map[pos] = word
    if not position_map:
        return ""
    return " ".join(position_map[i] for i in sorted(position_map.keys()))


async def search_openalex(query: str, top_k: int = 2) -> list[dict]:
    """
    Search OpenAlex for relevant open-access papers. Returns [] on any failure —
    this must NEVER raise, and must NEVER be allowed to consume more than ~5 seconds
    total (including any retry), since it sits in the critical path of every /advisory
    request and the mobile app has its own timeout waiting on this whole call.
    """
    if not query or not query.strip():
        return []

    query = query.strip().rstrip("?").strip()
    if not query:
        return []
    search_parameter = (
        "search.exact" if "*" in query or "?" in query else "search"
    )
    params = {
        search_parameter: query,
        "per_page": top_k,
        "mailto": OPENALEX_CONTACT_EMAIL,  # always include this exact email on every request
        "filter": "has_abstract:true",     # only return results we can actually use as grounding text
    }
    api_key = os.getenv("OPENALEX_API_KEY")
    if api_key and api_key.strip():
        params["api_key"] = api_key.strip()
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(OPENALEX_BASE_URL, params=params)
            if resp.status_code != 200:
                try:
                    error_body = resp.json()
                except ValueError:
                    error_body = {}
                detail = (
                    error_body.get("message")
                    or error_body.get("error")
                    or resp.text[:300]
                )
                logger.warning(
                    "OpenAlex returned HTTP %s: %s",
                    resp.status_code,
                    detail,
                )
                return []
            data = resp.json()
    except (httpx.TimeoutException, httpx.RequestError) as error:
        logger.warning(
            "OpenAlex request failed; skipping web grounding for this request: %s",
            error,
        )
        return []

    results = []
    for work in data.get("results", []):
        abstract = _reconstruct_abstract(work.get("abstract_inverted_index"))
        if not abstract:
            continue  # skip anything without usable abstract text, same rule as before
        results.append({
            "title": work.get("title") or "",
            "abstract": abstract,
            "url": work.get("id") or "",  # OpenAlex work ID, resolves as a URL
            "year": work.get("publication_year"),
        })
    return results