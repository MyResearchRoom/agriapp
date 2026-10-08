"""Surface-soil property lookup using the public ISRIC SoilGrids API."""

import logging
import asyncio
from datetime import datetime, timezone
import math
from typing import Any

import httpx


logger = logging.getLogger("soil")
SOILGRIDS_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"
_cache: dict[tuple[float, float, str], dict[str, Any]] = {}
_missing_cache: set[tuple[float, float, str]] = set()

PROPERTY_UNITS = {
    "phh2o": ("ph", 10),
    "soc": ("organic_carbon_g_kg", 10),
    "nitrogen": ("nitrogen_g_kg", 100),
    "sand": ("sand_percent", 10),
    "silt": ("silt_percent", 10),
    "clay": ("clay_percent", 10),
    "bdod": ("bulk_density_kg_dm3", 100),
}

NEARBY_SAMPLE_OFFSETS_KM = (
    (5, 0),
    (-5, 0),
    (0, 5),
    (0, -5),
    (3.54, 3.54),
    (3.54, -3.54),
    (-3.54, 3.54),
    (-3.54, -3.54),
)


def _location_key(lat: float, lon: float) -> tuple[float, float, str]:
    return (
        round(lat, 5),
        round(lon, 5),
        datetime.now(timezone.utc).date().isoformat(),
    )


def _parse_properties(payload: dict[str, Any]) -> dict[str, Any] | None:
    layers = payload.get("properties", {}).get("layers", [])
    if not isinstance(layers, list):
        return None
    values: dict[str, Any] = {}
    for layer in layers:
        if not isinstance(layer, dict):
            continue
        spec = PROPERTY_UNITS.get(layer.get("name"))
        if not spec:
            continue
        output_name, divisor = spec
        depths = layer.get("depths", [])
        if not isinstance(depths, list):
            continue
        surface = next(
            (item for item in depths if isinstance(item, dict) and item.get("label") == "0-5cm"),
            depths[0] if depths else None,
        )
        raw = surface.get("values", {}).get("mean") if isinstance(surface, dict) else None
        if isinstance(raw, (int, float)) and math.isfinite(raw) and raw >= 0:
            values[output_name] = round(raw / divisor, 2)
    return values or None


async def _request_properties(
    client: httpx.AsyncClient,
    lat: float,
    lon: float,
) -> dict[str, Any] | None:
    params = [
        ("lon", lon),
        ("lat", lat),
        *[("property", name) for name in PROPERTY_UNITS],
        ("depth", "0-5cm"),
        ("value", "mean"),
    ]
    response = await client.get(SOILGRIDS_URL, params=params)
    response.raise_for_status()
    return _parse_properties(response.json())


def _offset_coordinate(
    lat: float,
    lon: float,
    east_km: float,
    north_km: float,
) -> tuple[float, float]:
    latitude = lat + north_km / 111.32
    longitude = lon + east_km / (111.32 * max(math.cos(math.radians(lat)), 0.01))
    return latitude, longitude


async def get_soil_properties(lat: float, lon: float) -> dict[str, Any] | None:
    key = _location_key(lat, lon)
    if key in _cache:
        return _cache[key]
    if key in _missing_cache:
        return None

    try:
        async with httpx.AsyncClient(timeout=25.0) as client:
            values = await _request_properties(client, lat, lon)
            sampled_lat, sampled_lon = lat, lon
            approximate = False
            if not values:
                candidates = [
                    (*_offset_coordinate(lat, lon, east, north), east, north)
                    for east, north in NEARBY_SAMPLE_OFFSETS_KM
                ]
                results = await asyncio.gather(
                    *(
                        _request_properties(client, candidate_lat, candidate_lon)
                        for candidate_lat, candidate_lon, _, _ in candidates
                    ),
                    return_exceptions=True,
                )
                available = []
                for candidate, result in zip(candidates, results):
                    candidate_lat, candidate_lon, east, north = candidate
                    if isinstance(result, Exception):
                        logger.info("Nearby SoilGrids sample failed: %s", result)
                        continue
                    if result:
                        distance = math.hypot(east, north)
                        available.append(
                            (distance, candidate_lat, candidate_lon, result)
                        )
                if available:
                    _, sampled_lat, sampled_lon, values = min(
                        available,
                        key=lambda sample: sample[0],
                    )
                    approximate = True

        if not values:
            _missing_cache.add(key)
            return None

        values["depth"] = "0-5 cm"
        values["sample_latitude"] = round(sampled_lat, 5)
        values["sample_longitude"] = round(sampled_lon, 5)
        values["approximate"] = approximate
        if approximate:
            values["sample_distance_km"] = round(
                math.hypot(
                    (sampled_lat - lat) * 111.32,
                    (sampled_lon - lon)
                    * 111.32
                    * math.cos(math.radians(lat)),
                ),
                1,
            )
        _cache[key] = values
        if len(_cache) > 256:
            _cache.pop(next(iter(_cache)))
        return values
    except Exception as error:
        logger.warning("SoilGrids lookup failed (%s): %s", type(error).__name__, error)
        return None
