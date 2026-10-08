import logging
from threading import Lock

import numpy as np


logger = logging.getLogger("topic_guardrail")

IN_SCOPE_TOPICS = [
    "crop diseases and pests",
    "plant leaf symptoms and treatment",
    "fertilizer and pesticide recommendations",
    "farming practices and crop management",
    "weather and its effect on crops",
    "mandi and market prices for crops",
    "agricultural import and export",
    "government farming schemes and crop insurance",
    "soil health and soil properties",
    "satellite imagery and crop monitoring",
    "sericulture and mulberry cultivation",
    "livestock feed safety related to farming",
    "farm equipment and agricultural financing",
]

MULTILINGUAL_AGRICULTURE_TERMS = (
    "खेती", "किसान", "फसल", "पौधा", "पत्त", "पत्ता", "पत्ते", "रोग", "बीमारी",
    "कीट", "दवा", "खाद", "उर्वरक", "छिड़काव", "सिंचाई", "मौसम", "मंडी",
    "भाव", "कीमत", "योजना", "बीमा", "मिट्टी", "खेत", "धान", "गेहूं",
    "कपास", "शहतूत", "रेशम", "पशु", "चारा", "कृषि",
    "शेती", "शेतकरी", "पीक", "पिके", "झाड", "पान", "पाने", "रोग", "कीड",
    "औषध", "खत", "फवारणी", "सिंचन", "हवामान", "बाजार", "किंमत", "विमा",
    "माती", "शेत", "तांदूळ", "गहू", "कापूस", "तुती", "रेशीम", "पशुधन",
    "आजार", "पाण्य",
    "कृषी",
)

RELEVANCE_THRESHOLD = 0.35
_topic_vectors = None
_topic_vectors_lock = Lock()

OFF_TOPIC_REPLY = {
    "en": (
        "I'm AgriAI, built to help with farming, crop diseases, and "
        "agriculture-related questions. Could you ask me something related to that?"
    ),
    "hi": (
        "मैं AgriAI हूं, खेती, फसल रोगों और कृषि से जुड़े सवालों में मदद के लिए "
        "बना हूं। कृपया उससे जुड़ा कोई सवाल पूछें।"
    ),
    "mr": (
        "मी AgriAI आहे, शेती, पीक रोग आणि शेतीशी संबंधित प्रश्नांसाठी मदत "
        "करण्यासाठी बनवले आहे. कृपया त्याच्याशी संबंधित प्रश्न विचारा."
    ),
}


def _get_topic_vectors() -> np.ndarray:
    global _topic_vectors
    if _topic_vectors is None:
        with _topic_vectors_lock:
            if _topic_vectors is None:
                import retrieval

                model = retrieval.retriever.model
                _topic_vectors = model.encode(
                    IN_SCOPE_TOPICS,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                )
    return _topic_vectors


def is_agriculture_related(message: str) -> bool:
    if not message or not message.strip():
        return False

    normalized = message.casefold()
    if any(term in normalized for term in MULTILINGUAL_AGRICULTURE_TERMS):
        return True

    import retrieval

    model = retrieval.retriever.model
    message_vector = model.encode(
        [message.strip()[:4000]],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]
    similarities = _get_topic_vectors() @ message_vector
    score = float(np.max(similarities))
    logger.debug("Agriculture topic relevance score: %.3f", score)
    return score >= RELEVANCE_THRESHOLD
