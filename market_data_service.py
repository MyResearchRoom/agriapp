import logging
import os
import asyncio
from datetime import datetime, timezone
from typing import Any

import httpx


logger = logging.getLogger("market_data")
MANDI_PRICES_URL = "https://mandi-api.onrender.com/v1/prices"
FAOSTAT_BASE_URL = "https://fenixservices.fao.org/faostat/api/v1/en"
COMTRADE_URL = "https://comtradeapi.un.org/public/v1/preview/C/A/HS"

# Standard HS trade codes; only commodities with an explicit code are queried.
HS_COMMODITY_CODES = {
    "wheat": "1001",
    "rice": "1006",
    "paddy": "1006",
    "maize": "1005",
    "corn": "1005",
    "soybean": "1201",
    "soybeans": "1201",
    "cotton": "5201",
    "sugar": "1701",
    "coffee": "0901",
    "tea": "0902",
    "potato": "0701",
    "potatoes": "0701",
    "onion": "0703",
    "onions": "0703",
}


def _commodity_code(commodity: str) -> str | None:
    normalized = commodity.casefold().strip()
    return HS_COMMODITY_CODES.get(normalized)


async def get_mandi_price(
    commodity: str,
    state: str | None = None,
    market: str | None = None,
) -> dict | None:
    commodity = commodity.strip()
    if not commodity:
        return None
    requested_state = (state or "Maharashtra").strip()
    params = {"state": requested_state, "commodity": commodity}
    if market and market.strip():
        params["market"] = market.strip()
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(MANDI_PRICES_URL, params=params)
            response.raise_for_status()
            payload = response.json()
        records = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(records, list) or not records:
            return None
        normalized_records = [
            {
                key: record.get(key)
                for key in (
                    "state",
                    "district",
                    "market",
                    "commodity",
                    "variety",
                    "grade",
                    "arrival_date",
                    "min_price",
                    "max_price",
                    "modal_price",
                )
            }
            for record in records
            if isinstance(record, dict)
        ]
        normalized_records.sort(
            key=lambda record: str(record.get("arrival_date") or ""),
            reverse=True,
        )
        if not normalized_records:
            return None
        metadata = payload.get("meta")
        metadata = metadata if isinstance(metadata, dict) else {}
        return {
            "state": metadata.get("state", requested_state),
            "commodity": metadata.get("commodity", commodity),
            "market": market.strip() if market and market.strip() else None,
            "price_unit": "INR per quintal",
            "records": normalized_records[:10],
            "record_count": len(normalized_records),
        }
    except Exception as error:
        logger.warning("Mandi price lookup failed: %s", error)
        return None


async def get_global_price_trend(commodity: str) -> dict | None:
    if not commodity.strip():
        return None
    try:
        async with httpx.AsyncClient(timeout=6.0) as client:
            definitions = await client.get(
                f"{FAOSTAT_BASE_URL}/definitions/domain/PP"
            )
            definitions.raise_for_status()
            definition_data = definitions.json()
            definition_rows = (
                definition_data.get("data")
                if isinstance(definition_data, dict)
                else None
            )
            if not isinstance(definition_rows, list):
                return None
            match = next(
                (
                    item
                    for item in definition_rows
                    if isinstance(item, dict)
                    and commodity.casefold()
                    in str(item.get("label") or item.get("description") or "").casefold()
                ),
                None,
            )
            if not match:
                return None
            item_code = match.get("code") or match.get("item_code")
            if not item_code:
                return None
            data_response = await client.get(
                f"{FAOSTAT_BASE_URL}/data/PP",
                params={"item_code": item_code, "limit": 10},
            )
            data_response.raise_for_status()
            data_payload = data_response.json()
        records = data_payload.get("data") if isinstance(data_payload, dict) else None
        if not isinstance(records, list) or not records:
            return None
        return {"records": records[:10], "item": match.get("label", commodity)}
    except Exception as error:
        logger.warning("FAOSTAT price lookup failed: %s", error)
        return None


async def get_trade_data(commodity: str) -> dict | None:
    code = _commodity_code(commodity)
    if not code:
        return None
    subscription_key = os.getenv("UN_COMTRADE_SUBSCRIPTION_KEY", "").strip()
    base_params = {
        "reporterCode": 356,
        "cmdCode": code,
        "partnerCode": 0,
        "period": str(datetime.now(timezone.utc).year - 2),
    }
    headers = (
        {"Ocp-Apim-Subscription-Key": subscription_key}
        if subscription_key
        else {}
    )
    try:
        async with httpx.AsyncClient(timeout=6.0) as client:
            async def fetch_flow(flow_code: str) -> tuple[str, Any]:
                response = await client.get(
                    COMTRADE_URL,
                    params={**base_params, "flowCode": flow_code},
                    headers=headers,
                )
                response.raise_for_status()
                return flow_code, response.json()

            responses = await asyncio.gather(
                fetch_flow("X"),
                fetch_flow("M"),
                return_exceptions=True,
            )
        flow_records = {}
        for response in responses:
            if isinstance(response, Exception):
                logger.warning("UN Comtrade flow lookup failed: %s", response)
                continue
            flow, payload = response
            records = payload.get("data") if isinstance(payload, dict) else None
            if isinstance(records, list) and records:
                flow_records["exports" if flow == "X" else "imports"] = records[:10]
        if not flow_records:
            return None
        return {"flows": flow_records, "commodity_code": code}
    except Exception as error:
        logger.warning("UN Comtrade lookup failed: %s", error)
        return None
