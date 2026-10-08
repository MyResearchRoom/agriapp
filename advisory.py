import asyncio
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from groq import AsyncGroq

from groq_service import GroqGenerationError, generate_with_model_cascade
import retrieval
from openalex_client import search_openalex
from domain_prompt import DOMAIN_SYSTEM_PROMPT


# Load environment variables from .env in rag_backend or project root
BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR.parent / ".env")

logger = logging.getLogger("advisory")


REQUIRED_KEYS = {
    "summary",
    "symptoms",
    "immediate_steps",
    "recovery",
    "prevention",
    "warning",
}

NON_DISEASE_CLASSES = {
    "disease free leaves",
    "mulberry tree",
}

LOW_CONFIDENCE_GROUNDING_THRESHOLD = 0.45
SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "mr": "Marathi",
}


class AdvisoryGenerationError(Exception):
    """Raised when all models fail to produce a grounded advisory."""


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def build_grounded_prompt(
    crop: str,
    disease: str,
    confidence: float,
    retrieved_chunks: list[dict],
    web_papers: list[dict],
    language: str = "en",
) -> str:
    """Build a grounded advisory prompt in the requested response language."""

    response_language = SUPPORTED_LANGUAGES.get(language, SUPPORTED_LANGUAGES["en"])

    context_blocks = []

    # 1. Local context literature
    for index, chunk in enumerate(retrieved_chunks, 1):
        text = chunk.get("chunk_text", "").strip()
        source_file = chunk.get("source_file", "Knowledge Base")
        source_folder = chunk.get("source_folder", "info")
        page = chunk.get("page_number", 1)

        if text:
            context_blocks.append(
                f"[Local Context {index} — from your knowledge base, "
                f"file: '{source_file}', folder: '{source_folder}', "
                f"page: {page}]: {text}"
            )

    # 2. Web context research
    for index, paper in enumerate(web_papers, 1):
        title = paper.get("title", "")
        abstract = paper.get("abstract", "").strip()

        if title and abstract:
            context_blocks.append(
                f"[Web Context {index} — from published research, "
                f"title: '{title}']: {abstract}"
            )

    context_str = (
        "\n\n".join(context_blocks)
        if context_blocks
        else "[No specific RAG context literature retrieved]"
    )

    disease_clean = disease.strip().lower()

    if disease_clean in NON_DISEASE_CLASSES:
        case_instructions = """
This is a healthy leaf / non-disease condition.
Do not fabricate a plant disease or recommend unnecessary pesticides.
Mention healthy leaf maintenance, optimal harvesting for silkworms,
and routine field management.
"""
        case_label = f"Condition: {disease}"

    else:
        case_instructions = """
Preserve the detected disease label and focus on symptom recognition,
immediate field steps, recovery, and prevention.
"""
        case_label = f"Detected Disease: {disease}"

    return f"""{DOMAIN_SYSTEM_PROMPT}

---
ADVISORY TASK INSTRUCTIONS:

You are an expert agricultural advisory system grounded in verified
sericulture and agricultural research literature.

Crop: {crop} | {case_label} | Vision Confidence: {confidence}%

VERIFIED AGRONOMIC CONTEXT LITERATURE
(HYBRID RAG GROUNDING):

{context_str}

INSTRUCTIONS:

{case_instructions}

Write every advisory text value in {response_language}. Do not switch to English,
even if the source material is in English. Keep the JSON keys exactly as listed.
Preserve crop and disease names, scientific names, chemical names, units, and
numerical dosages as written when translating.

Return only valid JSON with these exact keys:

summary
symptoms
immediate_steps
recovery
prevention
warning

The JSON keys themselves must remain exactly as specified.

Treat Local Context literature as the primary and most authoritative
grounding for treatment dosages, chemical names, and safety waiting
periods.

Treat Web Context abstracts as supplementary literature.

Derive treatment dosages, chemical names, and safety waiting periods
strictly from the provided RAG Context Literature.

Do not hallucinate chemical remedies not listed in the retrieved context.

Do not change or replace the predicted condition.

State confidence as exactly {confidence}%.
"""


# ---------------------------------------------------------------------------
# Advisory generation
# ---------------------------------------------------------------------------


async def get_grounded_advisory(
    crop: str,
    disease: str,
    confidence: float,
    language: str = "en",
) -> dict:
    """
    Retrieve hybrid grounding context and generate the advisory in the
    requested response language.
    """

    if language not in SUPPORTED_LANGUAGES:
        raise AdvisoryGenerationError(
            f"Unsupported advisory language '{language}'. "
            f"Choose one of: {', '.join(SUPPORTED_LANGUAGES)}."
        )

    load_dotenv(BASE_DIR / ".env", override=True)
    load_dotenv(BASE_DIR.parent / ".env", override=True)

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key or api_key.strip() == "":
        raise AdvisoryGenerationError(
            "GROQ_API_KEY is not configured in environment."
        )

    groq_client = AsyncGroq(api_key=api_key.strip())

    # -----------------------------------------------------------------------
    # 1. Retrieve hybrid grounding context
    # -----------------------------------------------------------------------

    query_disease = disease.replace("_", " ").replace("milky", "mildew")
    query = f"{crop} {query_disease} treatment symptoms prevention"

    local_res, web_res = await asyncio.gather(
        asyncio.to_thread(retrieval.search, query, 3),
        search_openalex(query, top_k=2),
        return_exceptions=True,
    )

    retrieved_chunks = (
        local_res if isinstance(local_res, list) else []
    )

    web_papers = (
        web_res if isinstance(web_res, list) else []
    )

    if isinstance(local_res, Exception):
        logger.warning(
            "Local retrieval search encountered exception: %s",
            local_res,
        )

    if isinstance(web_res, Exception):
        logger.warning(
            "OpenAlex search encountered exception: %s",
            web_res,
        )

    # -----------------------------------------------------------------------
    # 2. Grounding score
    # -----------------------------------------------------------------------

    top_score = (
        retrieved_chunks[0].get("similarity_score", 0.0)
        if retrieved_chunks
        else 0.0
    )

    low_confidence_grounding = (
        top_score < LOW_CONFIDENCE_GROUNDING_THRESHOLD
    )

    # -----------------------------------------------------------------------
    # 3. Generate in the selected app language
    # -----------------------------------------------------------------------

    prompt = build_grounded_prompt(
        crop,
        disease,
        confidence,
        retrieved_chunks,
        web_papers,
        language,
    )
    language_instruction = (
        f"Write every advisory text value in {SUPPORTED_LANGUAGES[language]}. "
        "Do not switch to English."
    )
    logger.info("Advisory language instruction sent to Groq: %s", language_instruction)

    # -----------------------------------------------------------------------
    # 4. Groq model cascade
    # -----------------------------------------------------------------------

    def validate_response(response, model_name):
        raw_text = response.choices[0].message.content
        if not raw_text or not raw_text.strip():
            raise ValueError(
                f"Empty response received from model '{model_name}'."
            )

        data = json.loads(raw_text)
        if not isinstance(data, dict):
            raise ValueError(
                f"Model '{model_name}' did not return a JSON object."
            )

        missing_keys = REQUIRED_KEYS - set(data.keys())
        if missing_keys:
            raise ValueError(
                f"Model '{model_name}' output missing required keys: "
                f"{missing_keys}"
            )
        return data

    try:
        data, model_name = await generate_with_model_cascade(
            groq_client,
            [{"role": "user", "content": prompt}],
            logger,
            validator=validate_response,
            temperature=0.2,
            response_format={"type": "json_object"},
            timeout=12.0,
        )
    except GroqGenerationError as error:
        raise AdvisoryGenerationError(str(error)) from error

    # ----------------------------------------------------------------
    # 6. Build sources
    # ----------------------------------------------------------------

    sources = []
    seen_sources = set()

    # Local sources
    for chunk in retrieved_chunks:
        sf = chunk.get("source_file")
        sf_folder = chunk.get("source_folder")
        pg = chunk.get("page_number")

        key = ("local", sf, sf_folder, pg)

        if key not in seen_sources and sf:
            seen_sources.add(key)

            sources.append(
                {
                    "type": "local",
                    "title": sf,
                    "source_file": sf,
                    "source_folder": sf_folder,
                    "page_number": pg,
                }
            )

    # Web sources
    for paper in web_papers:
        title = paper.get("title")
        url = paper.get("url")
        year = paper.get("year")

        key = ("web", title, url)

        if key not in seen_sources and title:
            seen_sources.add(key)

            sources.append(
                {
                    "type": "web",
                    "title": title,
                    "url": url,
                    "year": year,
                }
            )

    data["sources"] = sources
    data["low_confidence_grounding"] = low_confidence_grounding

    logger.info(
        "Successfully generated grounded advisory using model '%s'. "
        "Language: %s. Total sources: %d",
        model_name,
        language,
        len(sources),
    )

    return data