"""Review CABI PlantwisePlus access without scraping or copying its content."""

from datetime import date
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parent
KNOWLEDGE_BASE_DIR = BACKEND_DIR.parent / "knowledge_base"
REVIEW_DIR = KNOWLEDGE_BASE_DIR / "cabi"
PORTAL_URL = "https://plantwiseplusknowledgebank.org/"
TERMS_URL = "https://www.cabi.org/terms-and-conditions/"
PERMISSIONS_EMAIL = "permissions@cabi.org"
TARGET_TOPICS = (
    "mulberry leaf rust",
    "mulberry leaf spot",
    "mulberry powdery mildew",
    "rice diseases detected by the JK-TK model",
    "wheat diseases detected by the JK-TK model",
    "cotton diseases detected by the JK-TK model",
)


def main() -> None:
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REVIEW_DIR / "ACCESS_REVIEW.txt"
    topics = "\n".join(f"- {topic}" for topic in TARGET_TOPICS)
    report = f"""CABI PlantwisePlus Knowledge Bank source review
Reviewed: {date.today().isoformat()}
Portal: {PORTAL_URL}
Terms: {TERMS_URL}

Topics to search manually:
{topics}

Access and reuse decision:
- No public factsheet API or bulk-download permission was confirmed.
- CABI's published website terms prohibit systematic downloading/archiving,
  automated scraping/crawling, and use of CABI portal content with AI tools,
  including indexing, analysis, and generating outputs, without prior agreement
  and licensing. These restrictions directly cover adding factsheets to this
  project's AI/RAG knowledge base.
- No CABI factsheet text or PDF was downloaded or copied. This script is
  intentionally discovery-only; rate limiting would not make prohibited use
  permissible.
- Before adding any CABI material, request written permission and any required
  rights-holder licences for local storage, indexing, retrieval, and AI-generated
  responses. Contact {PERMISSIONS_EMAIL}, identify each factsheet by title and
  URL, and explain the intended AgriAI/RAG use. Re-check the terms and any
  item-specific licence after permission is obtained.

Manual workflow: open the portal, search the topics above, and record only
candidate titles/URLs here for permission review. Do not save factsheet files
into the knowledge base unless the required written permissions cover this use.
"""
    report_path.write_text(report, encoding="utf-8")
    print("CABI discovery stopped before content retrieval.")
    print("CABI's published terms prohibit systematic collection and AI/RAG use without prior permission.")
    print(f"No factsheets downloaded. Review topics and permissions notes: {report_path}")
    print(f"Portal: {PORTAL_URL}")
    print(f"Terms: {TERMS_URL}")
    print(f"Request written permission from {PERMISSIONS_EMAIL} before indexing content.")


if __name__ == "__main__":
    main()
