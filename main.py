import logging
from typing import Union, List, Optional, Literal
from fastapi import FastAPI, HTTPException, File, UploadFile, Query, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image
import io
import re
import urllib.request
import advisory
import chat_service
import product_service
import satellite_service
import soil_service
import market_data_service
import vision_models
import species_model
import indic_f5_tts

app = FastAPI(
    title="AgriAI Hybrid OpenAlex & Local RAG Advisory Service",
    description="Hybrid OpenAlex literature retrieval & Local FAISS knowledge base grounded Groq Llama advisory service for sericulture",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Google News RSS proxy (no API key) ─────────────────────────────
_NEWS_QUERIES = {
    "all": 'India+(drought+OR+flood+OR+"heavy+rain"+OR+cyclone)+(farmers+OR+agriculture)',
    "rain": 'India+("heavy+rain"+OR+rainfall+OR+monsoon)+(farmers+OR+agriculture+OR+crops)',
    "drought": 'India+(drought+OR+"water+stress"+OR+"dry+spell")+(farmers+OR+agriculture)',
    "flood": 'India+(flood+OR+flooding+OR+inundation)+(farmers+OR+agriculture+OR+crops)',
    "cyclone": 'India+(cyclone+OR+"tropical+storm"+OR+hurricane)+(farmers+OR+agriculture)',
}


def _extract_tag(block: str, tag: str) -> str:
    pattern = (
        rf"<{tag}(?:\s[^>]*)?><!\[CDATA\[([\s\S]*?)\]\]></{tag}>|"
        rf"<{tag}(?:\s[^>]*)?>([\s\S]*?)</{tag}>"
    )
    m = re.search(pattern, block, flags=re.IGNORECASE)
    if not m:
        return ""
    return (m.group(1) or m.group(2) or "").strip()


def _strip_html(s: str) -> str:
    s = re.sub(r"<[^>]+>", "", s)
    return (
        s.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&#39;", "'")
        .strip()
    )


@app.get("/news/rss")
def news_rss_proxy(filter: str = Query(default="all")):
    """Proxy Google News RSS so the mobile app never has to parse XML or hit CORS."""
    key = filter if filter in _NEWS_QUERIES else "all"
    q = _NEWS_QUERIES[key]
    url = f"https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "AgriAI/1.0"})
        with urllib.request.urlopen(req, timeout=12) as resp:
            xml = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Upstream news fetch failed: {e}")

    items = []
    parts = re.split(r"<item[\s>]", xml, flags=re.IGNORECASE)[1:]
    for i, part in enumerate(parts[:12]):
        block = part.split("</item>")[0] if "</item>" in part.lower() else part
        title = _strip_html(_extract_tag(block, "title"))
        link = _strip_html(_extract_tag(block, "link"))
        pub = _strip_html(_extract_tag(block, "pubDate"))
        source = _strip_html(_extract_tag(block, "source")) or "Google News"
        if not title:
            continue
        items.append(
            {
                "id": f"{key}-{i}-{link[:40]}",
                "title": title,
                "link": link,
                "published": pub,
                "source": source,
                "filterHint": key,
            }
        )
    return {"items": items, "filter": key}


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query string", example="Mulberry Leaf Rust treatment")
    top_k: int = Field(default=3, ge=1, le=50, description="Number of top results to return")


class SearchResponseItem(BaseModel):
    chunk_text: str
    source_file: str
    source_folder: str
    page_number: int
    chunk_index: int
    similarity_score: float


class AdvisoryRequest(BaseModel):
    crop: str = Field(..., description="Crop name", example="Mulberry")
    disease: str = Field(..., description="Detected disease name", example="Leaf Rust")
    confidence: float = Field(..., description="Vision classification confidence percentage", example=94.3)
    language: Literal["en", "hi", "mr"] = Field(
        default="en",
        description="Language code for advisory content: 'en', 'hi', or 'mr'",
    )


class VoiceChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=1600)


class VoiceChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    language: Literal["en", "hi", "mr"] = "en"
    history: List[VoiceChatTurn] = Field(default_factory=list, max_length=12)


class VoiceChatResponse(BaseModel):
    reply: str
    audio_b64: Optional[str] = None
    audio_format: Optional[str] = None


class VoiceTtsRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1600)
    language: Literal["en", "hi", "mr"] = "hi"


ChatRequest = VoiceChatRequest
ChatResponse = VoiceChatResponse


class ProductItem(BaseModel):
    title: str
    price: Optional[str] = None
    image_url: Optional[str] = None
    retailer: Optional[str] = None
    product_url: str


class ProductResponse(BaseModel):
    products: List[ProductItem] = Field(default_factory=list)


class SourceItem(BaseModel):
    type: str = "web"
    title: str
    source_file: Optional[str] = None
    source_folder: Optional[str] = None
    page_number: Optional[int] = None
    url: Optional[str] = None
    year: Optional[Union[int, str]] = None
    paper_id: Optional[str] = None
    venue: Optional[str] = None
    authors: List[str] = Field(default_factory=list)
    source_label: Optional[str] = None


class AdvisoryResponse(BaseModel):
    summary: Union[str, List[str]]
    symptoms: Union[str, List[str]]
    immediate_steps: Union[str, List[str]]
    recovery: Union[str, List[str]]
    prevention: Union[str, List[str]]
    warning: Union[str, List[str]]
    sources: List[SourceItem] = Field(default_factory=list)
    low_confidence_grounding: bool = False


class VisionClassifyResponse(BaseModel):
    disease: str
    confidence: float
    source_model: str
    detected_crop: str
    species_confidence: float
    crop_mismatch_warning: Optional[str] = None


@app.post("/vision/classify", response_model=VisionClassifyResponse)
async def classify_leaf_image(image: UploadFile = File(...)):
    """
    Two-stage pipeline for every non-Mulberry crop:
      1. BioCLIP 2 (species_model) identifies which crop species the photo shows.
      2. JK-TK/PlantDiseaseDetection (vision_models) detects the disease.
    Both results are returned together, with a simple cross-check warning if
    the disease model's predicted label doesn't match the detected species
    (e.g. species says "cotton" but the disease label starts with "Potato").
    """
    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read uploaded image: {e}")

    try:
        species_result = species_model.identify_crop_species(pil_image)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Species identification failed: {e}")

    try:
        disease_result = vision_models.classify_leaf(pil_image)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Disease classification failed: {e}")

    crop_mismatch_warning = None
    detected_crop = species_result["crop"]
    disease_label_lower = disease_result["disease"].lower()
    if detected_crop.lower() not in disease_label_lower:
        crop_mismatch_warning = (
            f"Species model detected '{detected_crop}', but the disease label "
            f"'{disease_result['disease']}' doesn't clearly match it — treat this "
            f"result with extra caution."
        )
        print(f"[Vision Pipeline] WARNING: {crop_mismatch_warning}")

    return {
        "disease": disease_result["disease"],
        "confidence": disease_result["confidence"],
        "source_model": disease_result["source_model"],
        "detected_crop": detected_crop,
        "species_confidence": species_result["confidence"],
        "crop_mismatch_warning": crop_mismatch_warning,
    }


@app.get("/health")
def health_check():
    """Health check endpoint for the Hybrid RAG service."""
    return {
        "status": "ok",
        "retrieval_mode": "hybrid_local_openalex",
        "generation_provider": "groq_llama",
        "vision_pipeline": ["imageomics/bioclip-2 (species ID)", "JK-TK/PlantDiseaseDetection (disease)"],
    }


@app.post("/search", response_model=list[SearchResponseItem])
def search_chunks(request: SearchRequest):
    """Legacy local vector search endpoint, kept for diagnostics."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    import retrieval

    results = retrieval.search(query=request.query, top_k=request.top_k)
    return results


@app.post("/advisory", response_model=AdvisoryResponse)
async def generate_advisory(request: AdvisoryRequest):
    """Generate Groq Llama advisory grounded by OpenAlex literature and local knowledge base."""
    try:
        advisory_data = await advisory.get_grounded_advisory(
            crop=request.crop,
            disease=request.disease,
            confidence=request.confidence,
            language=request.language,
        )
        return advisory_data
    except advisory.AdvisoryGenerationError as e:
        return JSONResponse(
            status_code=503,
            content={"error": "advisory_generation_failed", "detail": str(e)}
        )
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"error": "advisory_generation_failed", "detail": str(e)}
        )


@app.post("/voice/chat", response_model=VoiceChatResponse)
async def voice_chat(request: VoiceChatRequest):
    """Generate a multilingual voice reply grounded in AgriAI RAG sources."""
    try:
        reply = await chat_service.answer_farmer(
            message=request.message.strip(),
            language=request.language,
            history=[
                turn.model_dump()
                for turn in request.history
            ],
        )
        if request.language in {"hi", "mr"}:
            reply = indic_f5_tts.clean_text_for_speech(reply)
        payload = {"reply": reply}
        if request.language in {"hi", "mr"}:
            try:
                payload["audio_b64"] = indic_f5_tts.generate_audio_base64(reply, request.language)
                payload["audio_format"] = "wav"
            except Exception as exc:  # pragma: no cover - degrade gracefully if model not ready
                logger = logging.getLogger("voice_chat")
                logger.warning("IndicF5 TTS generation failed for %s: %s", request.language, exc)
        return payload
    except chat_service.ChatGenerationError as error:
        return JSONResponse(
            status_code=503,
            content={"error": "voice_chat_generation_failed", "detail": str(error)},
        )


@app.post("/voice/tts")
async def voice_tts(request: VoiceTtsRequest):
    """Generate speech audio for Hindi/Marathi replies using IndicF5."""
    if request.language not in {"hi", "mr"}:
        raise HTTPException(status_code=400, detail="IndicF5 TTS is enabled only for Hindi and Marathi.")
    try:
        audio_b64 = indic_f5_tts.generate_audio_base64(request.text, request.language)
        return {"audio_b64": audio_b64, "audio_format": "wav"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"IndicF5 TTS failed: {exc}") from exc


@app.post("/chat/message", response_model=ChatResponse)
async def chat_message(request: ChatRequest):
    try:
        reply = await chat_service.get_assistant_reply(
            message=request.message.strip(),
            history=[turn.model_dump() for turn in request.history],
            language=request.language,
        )
        return {"reply": reply}
    except Exception as error:
        raise HTTPException(status_code=503, detail=f"Chat reply failed: {error}") from error


@app.get("/products", response_model=ProductResponse)
async def get_products(
    query: str = Query(..., min_length=1, max_length=200),
):
    products = await product_service.search_products(query)
    return {"products": products}


@app.get("/satellite/true-color")
async def satellite_true_color(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
):
    image = await satellite_service.get_true_color_image(lat, lon)
    if image is None:
        raise HTTPException(
            status_code=404,
            detail="No clear satellite image available right now.",
        )
    return Response(content=image, media_type="image/png")


@app.get("/satellite/status")
async def satellite_status():
    configured = satellite_service.is_configured()
    return {
        "configured": configured,
        "provider": "NASA GIBS",
        "message": None,
    }


@app.get("/satellite/ndvi")
async def satellite_ndvi(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
):
    image = await satellite_service.get_ndvi_image(lat, lon)
    if image is None:
        raise HTTPException(
            status_code=404,
            detail="No clear satellite image available right now.",
        )
    return Response(content=image, media_type="image/png")


@app.get("/satellite/ndvi-trend")
async def satellite_ndvi_trend(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    days: int = Query(90, ge=5, le=365),
):
    return await satellite_service.get_ndvi_trend(lat, lon, days)


@app.get("/soil/properties")
async def soil_properties(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
):
    properties = await soil_service.get_soil_properties(lat, lon)
    if properties is None:
        raise HTTPException(
            status_code=404,
            detail="Soil data is unavailable for this location right now.",
        )
    return properties


@app.get("/market/prices")
async def market_prices(
    state: str = Query("Maharashtra", min_length=2, max_length=80),
    commodity: str = Query(..., min_length=2, max_length=80),
    market: Optional[str] = Query(None, min_length=2, max_length=120),
):
    prices = await market_data_service.get_mandi_price(commodity, state, market)
    if prices is None:
        raise HTTPException(
            status_code=503,
            detail="Mandi prices are temporarily unavailable for this location and crop.",
        )
    return prices


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)