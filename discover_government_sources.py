"""Download official scheme guidelines and cache AGMARKNET reports for review."""

import json
import hashlib
import logging
import os
import re
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
import pymupdf
from dotenv import load_dotenv


BACKEND_DIR = Path(__file__).resolve().parent
KNOWLEDGE_BASE_DIR = BACKEND_DIR.parent / "knowledge_base"
SCHEMES_DIR = KNOWLEDGE_BASE_DIR / "schemes"
AGMARKNET_DIR = KNOWLEDGE_BASE_DIR / "agmarknet"
AGMARKNET_RESOURCE_ID = "9ef84268-d588-465a-a308-a864a43d0070"
AGMARKNET_URL = f"https://api.data.gov.in/resource/{AGMARKNET_RESOURCE_ID}"
TARGET_COMMODITIES = ("Rice", "Wheat", "Cotton")
AGMARKNET_LOOKBACK_DAYS = 90
SCHEME_DOCUMENTS = (
    (
        "PM-KISAN_Revised_Operational_Guidelines_English.pdf",
        "https://pmkisan.gov.in/Documents/RevisedPM-KISANOperationalGuidelines(English).pdf",
        "PM-KISAN revised operational guidelines",
    ),
    (
        "PMFBY_Operational_Guidelines_2023.pdf",
        "https://pmfby.amnex.co.in/pmfby/pdf/operational_guidelines_pmfby.pdf",
        "PMFBY 2023 operational guidelines",
    ),
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("government_sources")

def _download_verified_pdf(
    client: httpx.Client,
    filename: str,
    url: str,
    expected_title: str,
) -> bool:
    destination = SCHEMES_DIR / filename
    if destination.exists():
        print(f"Existing scheme document left unchanged: {destination}")
        return True

    try:
        response = client.get(url)
        response.raise_for_status()
        if not response.content.startswith(b"%PDF-"):
            raise ValueError(
                f"official URL did not return a PDF (content-type "
                f"{response.headers.get('content-type', 'unknown')})"
            )
        document = pymupdf.open(stream=response.content, filetype="pdf")
        try:
            if not document.page_count:
                raise ValueError("PDF contains no pages")
            first_page = document[0].get_text("text")
            all_text = " ".join(
                document[page_number].get_text("text")
                for page_number in range(min(document.page_count, 3))
            ).casefold()
            if expected_title.split()[0].casefold() not in all_text:
                raise ValueError("PDF title/content did not match the expected scheme")
            if not first_page.strip():
                raise ValueError("PDF has no extractable text on its first page")
        finally:
            document.close()

        content_hash = hashlib.sha256(response.content).digest()
        duplicate = next(
            (
                existing
                for existing in KNOWLEDGE_BASE_DIR.rglob("*.pdf")
                if existing.is_file()
                and hashlib.sha256(existing.read_bytes()).digest() == content_hash
            ),
            None,
        )
        if duplicate is not None:
            print(
                f"{expected_title} is byte-identical to an existing knowledge-base "
                f"document; leaving it unchanged: {duplicate}"
            )
            return True

        temporary_path = destination.with_suffix(".pdf.download")
        temporary_path.write_bytes(response.content)
        temporary_path.replace(destination)
        print(f"Downloaded and validated {expected_title}: {destination}")
        return True
    except Exception as error:
        logger.error("Could not download %s from the official portal: %s", expected_title, error)
        return False

def _report_pdf_text(
    title: str,
    source_url: str,
    accessed_at: datetime,
    records: list[dict[str, Any]],
) -> str:
    header = [
        title,
        f"Source: AGMARKNET via data.gov.in ({source_url})",
        f"Retrieved (UTC): {accessed_at.isoformat()}",
        f"Records returned: {len(records)}",
        "These are historical reference records, not live quotes or a price forecast.",
        "",
    ]
    lines = header
    for index, record in enumerate(records, start=1):
        lines.append(f"Record {index}")
        for key, value in record.items():
            safe_value = re.sub(r"\s+", " ", str(value)).strip()
            lines.append(f"{key}: {safe_value}")
        lines.append("")
    return "\n".join(lines)


def _arrival_date(record: dict[str, Any]) -> date | None:
    value = record.get("arrival_date") or record.get("Arrival_Date")
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    for date_format in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(normalized, date_format).date()
        except ValueError:
            continue
    return None


def _write_text_pdf(destination: Path, text: str) -> None:
    document = pymupdf.open()
    page = document.new_page()
    font_size = 9
    margin = 42
    y = margin
    for line in text.splitlines():
        wrapped = [line[index : index + 105] for index in range(0, len(line), 105)] or [""]
        for part in wrapped:
            if y > page.rect.height - margin:
                page = document.new_page()
                y = margin
            page.insert_text((margin, y), part, fontsize=font_size)
            y += 13
    temporary_path = destination.with_suffix(".pdf.download")
    document.save(temporary_path)
    document.close()
    temporary_path.replace(destination)


def _download_agmarknet_reports() -> None:
    api_key = os.getenv("DATA_GOV_IN_API_KEY", "").strip()
    if not api_key:
        print(
            "AGMARKNET skipped: set DATA_GOV_IN_API_KEY in rag_backend/.env "
            "to download cached historical reports."
        )
        return

    AGMARKNET_DIR.mkdir(parents=True, exist_ok=True)
    retrieved_at = datetime.now(timezone.utc)
    day_stamp = retrieved_at.date().isoformat()
    success_count = 0
    with httpx.Client(timeout=30.0) as client:
        for commodity in TARGET_COMMODITIES:
            params = {
                "api-key": api_key,
                "format": "json",
                "limit": 100,
                "filters[commodity]": commodity,
            }
            try:
                response = client.get(AGMARKNET_URL, params=params)
                response.raise_for_status()
                payload = response.json()
                records = payload.get("records") if isinstance(payload, dict) else None
                if not isinstance(records, list) or not records:
                    print(f"AGMARKNET returned no records for {commodity}; no report saved.")
                    continue
                cutoff = retrieved_at.date() - timedelta(days=AGMARKNET_LOOKBACK_DAYS)
                valid_records = [
                    record
                    for record in records
                    if isinstance(record, dict)
                    and (arrival_date := _arrival_date(record)) is not None
                    and cutoff <= arrival_date <= retrieved_at.date()
                ]
                if not valid_records:
                    print(
                        f"AGMARKNET returned no dated records from the last "
                        f"{AGMARKNET_LOOKBACK_DAYS} days for {commodity}; no report saved."
                    )
                    continue

                raw_path = AGMARKNET_DIR / f"AGMARKNET_{commodity}_{day_stamp}.json"
                pdf_path = AGMARKNET_DIR / f"AGMARKNET_{commodity}_{day_stamp}.pdf"
                raw_path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                report = _report_pdf_text(
                    f"AGMARKNET Daily Mandi Price Report — {commodity}",
                    AGMARKNET_URL,
                    retrieved_at,
                    valid_records,
                )
                _write_text_pdf(pdf_path, report)
                print(
                    f"Saved {len(valid_records)} {commodity} records for manual review: "
                    f"{pdf_path}"
                )
                success_count += 1
            except Exception as error:
                logger.error("AGMARKNET request failed for %s: %s", commodity, error)
    print(
        f"AGMARKNET report summary: {success_count}/{len(TARGET_COMMODITIES)} "
        "commodity reports saved."
    )


def _report_npss_access() -> None:
    print(
        "NPSS skipped: the public portal at https://npss.dac.gov.in/app/ is an "
        "interactive application whose data API requires an authenticated session. "
        "No public API or bulk-download documentation was verified; no endpoint "
        "has been guessed or queried."
    )


def main() -> int:
    load_dotenv(BACKEND_DIR / ".env")
    SCHEMES_DIR.mkdir(parents=True, exist_ok=True)
    AGMARKNET_DIR.mkdir(parents=True, exist_ok=True)

    with httpx.Client(
        timeout=45.0,
        follow_redirects=True,
        headers={"User-Agent": "AgriAI-KnowledgeBase-SourceReview/1.0"},
    ) as client:
        scheme_results = [
            _download_verified_pdf(client, filename, url, title)
            for filename, url, title in SCHEME_DOCUMENTS
        ]

    _download_agmarknet_reports()
    _report_npss_access()
    if not all(scheme_results):
        print("One or more scheme documents could not be verified or downloaded.")
        return 1
    print(
        "Review all downloaded/generated PDFs and their source/date before running "
        "ingest_knowledge_base.py. No FAISS files were modified."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
