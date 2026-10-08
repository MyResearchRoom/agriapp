"""
Stage 1 crop-species gatekeeper — runs BEFORE JK-TK/PlantDiseaseDetection.
Uses BioCLIP 2 (zero-shot, imageomics/bioclip-2) to identify which crop
species a leaf photo actually shows, so the disease-detection stage isn't
guessing/confusing crops 

Mulberry never reaches this file at all — it's handled entirely by the
separate on-device EfficientNet-B0 model (src/services/mulberryModel.ts).

BioCLIP 2 was independently verified as real, MIT-licensed, and field-
validated before being wired in here.
"""
import torch
import open_clip
from PIL import Image

MODEL_ID = "hf-hub:imageomics/bioclip-2"

# Crop species this app currently supports outside of Mulberry.
# Keep this in sync with the active (non-Mulberry) entries in
# src/screens/PlantSelectScreen.tsx whenever you add/remove a crop.
CROP_LABELS = ["rice", "wheat", "cotton"]

_model = None
_preprocess = None
_tokenizer = None
_text_features = None


def _load_model():
    """Lazy-loads BioCLIP 2 once, on first use, and pre-computes the text
    embeddings for CROP_LABELS once (they never change per-request)."""
    global _model, _preprocess, _tokenizer, _text_features

    if _model is None:
        print(f"[Species Model] Loading {MODEL_ID} ...")
        _model, _, _preprocess = open_clip.create_model_and_transforms(MODEL_ID)
        _tokenizer = open_clip.get_tokenizer(MODEL_ID)
        _model.eval()

        prompts = [f"a photo of a {label} plant leaf" for label in CROP_LABELS]
        tokens = _tokenizer(prompts)
        with torch.no_grad():
            _text_features = _model.encode_text(tokens)
            _text_features = _text_features / _text_features.norm(dim=-1, keepdim=True)
        print("[Species Model] BioCLIP 2 loaded successfully.")


def identify_crop_species(image: Image.Image) -> dict:
    """
    Returns {"crop": str, "confidence": float (0.0-1.0)} — whichever of
    CROP_LABELS BioCLIP 2 matched most strongly against the given photo.
    """
    _load_model()

    image_input = _preprocess(image).unsqueeze(0)
    with torch.no_grad():
        image_features = _model.encode_image(image_input)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ _text_features.T).softmax(dim=-1)[0]

    best_idx = int(torch.argmax(similarity).item())
    crop = CROP_LABELS[best_idx]
    confidence = float(similarity[best_idx].item())

    print(f"[Species Model] Detected crop species: {crop} ({confidence:.4f})")
    return {"crop": crop, "confidence": confidence}
