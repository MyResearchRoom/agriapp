"""Public NASA GIBS satellite imagery and approximate NDVI observations."""

import asyncio
import base64
import io
import logging
import math
import re
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta, timezone
from typing import Any

import httpx
from PIL import Image

logger = logging.getLogger("satellite")

GIBS_BASE_URL = "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best"
CAPABILITIES_URL = f"{GIBS_BASE_URL}/1.0.0/WMTSCapabilities.xml"
NDVI_LAYER = "MODIS_Terra_L3_NDVI_16Day"
TRUE_COLOR_LAYER = "MODIS_Terra_CorrectedReflectance_TrueColor"
NDVI_LEGEND_URL = "https://gibs.earthdata.nasa.gov/legends/MODIS_L3_NDVI_H.svg"
TILE_ZOOM = 9
CACHE_LIMIT = 256
CACHE_TTL_SECONDS = 24 * 60 * 60

_cache: dict[tuple[Any, ...], tuple[float, Any]] = {}
_cache_lock = asyncio.Lock()
_capability_dates: list[date] = []
_capability_fetched_at = 0.0
_capability_lock = asyncio.Lock()
_ndvi_palette: list[tuple[int, int, int]] = []
_palette_lock = asyncio.Lock()


def is_configured() -> bool:
    """NASA GIBS public imagery does not require account credentials."""
    return True


def _tile_for_coordinates(lat: float, lon: float) -> tuple[int, int, int, int] | None:
    if not -85.05112878 <= lat <= 85.05112878:
        return None
    scale = 2**TILE_ZOOM
    world_x = (lon + 180) / 360 * scale
    radians = math.radians(lat)
    world_y = (1 - math.asinh(math.tan(radians)) / math.pi) / 2 * scale
    tile_x = min(max(int(world_x), 0), scale - 1)
    tile_y = min(max(int(world_y), 0), scale - 1)
    pixel_x = min(max(int((world_x - tile_x) * 256), 0), 255)
    pixel_y = min(max(int((world_y - tile_y) * 256), 0), 255)
    return tile_x, tile_y, pixel_x, pixel_y


async def _cached(key: tuple[Any, ...]) -> Any:
    async with _cache_lock:
        cached = _cache.get(key)
        if cached is None:
            return None
        expires_at, value = cached
        if time.monotonic() >= expires_at:
            del _cache[key]
            return None
        return value


async def _store(key: tuple[Any, ...], value: Any) -> None:
    async with _cache_lock:
        if len(_cache) >= CACHE_LIMIT:
            expired = next(
                (item for item, (expires_at, _) in _cache.items()
                 if time.monotonic() >= expires_at),
                None,
            )
            del _cache[expired if expired is not None else next(iter(_cache))]
        _cache[key] = (time.monotonic() + CACHE_TTL_SECONDS, value)


async def _get_location_tile(
    lat: float,
    lon: float,
    layer: str,
    observation_date: date | None = None,
) -> bytes | None:
    tile_coordinates = _tile_for_coordinates(lat, lon)
    if tile_coordinates is None:
        return None
    tile_x, tile_y, _, _ = tile_coordinates
    day_key = observation_date.isoformat() if observation_date else "default"
    cache_key = (
        "location-tile",
        round(lat, 5),
        round(lon, 5),
        layer,
        day_key,
    )
    cached = await _cached(cache_key)
    if isinstance(cached, bytes):
        return cached

    if observation_date:
        path = (
            f"{layer}/default/{day_key}/GoogleMapsCompatible_Level{TILE_ZOOM}/"
            f"{TILE_ZOOM}/{tile_y}/{tile_x}"
        )
    else:
        path = (
            f"{layer}/default/GoogleMapsCompatible_Level{TILE_ZOOM}/"
            f"{TILE_ZOOM}/{tile_y}/{tile_x}"
        )
    extension = "jpg" if layer == TRUE_COLOR_LAYER else "png"
    url = f"{GIBS_BASE_URL}/{path}.{extension}"
    try:
        async with httpx.AsyncClient(timeout=25.0) as client:
            response = await client.get(url)
            response.raise_for_status()
        if not response.content or not response.headers.get("content-type", "").startswith("image/"):
            return None
        with Image.open(io.BytesIO(response.content)) as image:
            rgba = image.convert("RGBA")
            if rgba.size != (256, 256) or rgba.getchannel("A").getbbox() is None:
                return None
            output = io.BytesIO()
            rgba.save(output, format="PNG")
        image_bytes = output.getvalue()
        await _store(cache_key, image_bytes)
        return image_bytes
    except Exception as error:
        logger.info(
            "NASA GIBS image unavailable for %s at %.5f, %.5f (%s): %s",
            day_key,
            lat,
            lon,
            type(error).__name__,
            error,
        )
        return None


async def get_true_color_image(lat: float, lon: float) -> bytes | None:
    return await _get_location_tile(lat, lon, TRUE_COLOR_LAYER)


async def get_ndvi_image(lat: float, lon: float) -> bytes | None:
    return await _get_location_tile(lat, lon, NDVI_LAYER)


async def _available_ndvi_dates() -> list[date]:
    global _capability_dates, _capability_fetched_at
    if time.monotonic() - _capability_fetched_at < CACHE_TTL_SECONDS:
        return _capability_dates

    async with _capability_lock:
        if time.monotonic() - _capability_fetched_at < CACHE_TTL_SECONDS:
            return _capability_dates
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(CAPABILITIES_URL)
                response.raise_for_status()
            root = ET.fromstring(response.content)
            ns = {
                "wmts": "http://www.opengis.net/wmts/1.0",
                "ows": "http://www.opengis.net/ows/1.1",
            }
            layer = next(
                (
                    item
                    for item in root.findall(".//wmts:Layer", ns)
                    if item.findtext("ows:Identifier", namespaces=ns) == NDVI_LAYER
                ),
                None,
            )
            if layer is None:
                return []
            dates: set[date] = set()
            for value in layer.findall("wmts:Dimension/wmts:Value", ns):
                parts = (value.text or "").split("/")
                if len(parts) != 3 or parts[2] != "P16D":
                    continue
                start = date.fromisoformat(parts[0])
                end = date.fromisoformat(parts[1])
                current = start
                while current <= end:
                    dates.add(current)
                    current += timedelta(days=16)
            _capability_dates = sorted(dates)
            _capability_fetched_at = time.monotonic()
            return _capability_dates
        except Exception as error:
            logger.info(
                "NASA GIBS NDVI dates could not be loaded (%s): %s",
                type(error).__name__,
                error,
            )
            return []


async def _legend_palette() -> list[tuple[int, int, int]]:
    global _ndvi_palette
    if _ndvi_palette:
        return _ndvi_palette
    async with _palette_lock:
        if _ndvi_palette:
            return _ndvi_palette
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.get(NDVI_LEGEND_URL)
                response.raise_for_status()
            match = re.search(
                rb"data:image/png;base64,\s*([^\"']+)",
                response.content,
            )
            if match is None:
                return []
            image_bytes = base64.b64decode(match.group(1).strip())
            with Image.open(io.BytesIO(image_bytes)) as image:
                legend = image.convert("RGBA")
                row = legend.height // 2
                _ndvi_palette = [
                    legend.getpixel((x, row))[:3]
                    for x in range(legend.width)
                ]
            return _ndvi_palette
        except Exception as error:
            logger.info(
                "NASA GIBS NDVI legend could not be loaded (%s): %s",
                type(error).__name__,
                error,
            )
            return []


async def _read_ndvi_pixel(
    lat: float,
    lon: float,
    observation_date: date,
    palette: list[tuple[int, int, int]],
) -> float | None:
    tile_coordinates = _tile_for_coordinates(lat, lon)
    if tile_coordinates is None:
        return None
    _, _, pixel_x, pixel_y = tile_coordinates
    image_bytes = await _get_location_tile(
        lat,
        lon,
        NDVI_LAYER,
        observation_date,
    )
    if image_bytes is None:
        return None
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            pixel = image.convert("RGBA").getpixel((pixel_x, pixel_y))
        if pixel[3] == 0:
            return None
        nearest_index = min(
            range(len(palette)),
            key=lambda index: sum(
                (pixel[channel] - palette[index][channel]) ** 2
                for channel in range(3)
            ),
        )
        color_error = sum(
            (pixel[channel] - palette[nearest_index][channel]) ** 2
            for channel in range(3)
        )
        if color_error > 1800:
            return None
        return round(nearest_index / max(len(palette) - 1, 1), 3)
    except Exception as error:
        logger.info(
            "Could not decode NASA GIBS NDVI pixel on %s (%s): %s",
            observation_date,
            type(error).__name__,
            error,
        )
        return None


async def get_ndvi_trend(lat: float, lon: float, days: int = 90) -> list[dict[str, Any]]:
    days = min(max(int(days), 5), 365)
    cache_key = (
        "trend",
        round(lat, 5),
        round(lon, 5),
        datetime.now(timezone.utc).date().isoformat(),
        days,
    )
    cached = await _cached(cache_key)
    if isinstance(cached, list):
        return cached
    if _tile_for_coordinates(lat, lon) is None:
        return []

    cutoff = datetime.now(timezone.utc).date() - timedelta(days=days)
    today = datetime.now(timezone.utc).date()
    dates = [
        observation_date
        for observation_date in await _available_ndvi_dates()
        if cutoff <= observation_date <= today
    ]
    palette = await _legend_palette()
    if not dates or not palette:
        return []

    results = await asyncio.gather(
        *(
            _read_ndvi_pixel(lat, lon, observation_date, palette)
            for observation_date in dates
        )
    )
    series = [
        {
            "date": observation_date.isoformat(),
            "ndvi": ndvi,
        }
        for observation_date, ndvi in zip(dates, results)
        if ndvi is not None
    ]
    await _store(cache_key, series)
    return series
