"""
Runs JK-TK/PlantDiseaseDetection (YOLOv11x, MIT License) for all non-Mulberry
crops. Mulberry uses the separate on-device EfficientNet-B0 model instead
(see src/services/mulberryModel.ts) — this module is never called for Mulberry.

This model was independently verified as real, correctly-licensed, and
downloadable before being wired in here — do not swap it without
re-verifying license + real download link first.
"""
import os
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
import glob
from PIL import Image
from ultralytics import YOLO
from huggingface_hub import snapshot_download

YOLO_MODEL_ID = "JK-TK/PlantDiseaseDetection"

_yolo_model = None


def _load_model():
    """Lazy-loads the model once, on first use, and reuses it on every
    subsequent call."""
    global _yolo_model

    if _yolo_model is None:
        print(f"[Vision Models] Downloading {YOLO_MODEL_ID} weights from Hugging Face Hub...")
        local_dir = snapshot_download(repo_id=YOLO_MODEL_ID)
        pt_files = glob.glob(f"{local_dir}/**/*.pt", recursive=True)
        if not pt_files:
            raise RuntimeError(f"No .pt weights file found in downloaded repo {YOLO_MODEL_ID}")
        weights_path = pt_files[0]
        print(f"[Vision Models] Loading YOLO weights from {weights_path}")
        _yolo_model = YOLO(weights_path)
        print("[Vision Models] YOLOv11x model loaded successfully.")


def classify_leaf(image: Image.Image) -> dict:
    """
    Runs the YOLO model on the given PIL image and returns:
    {"disease": str, "confidence": float (0.0-1.0), "source_model": str}
    """
    _load_model()

    results = _yolo_model.predict(image, verbose=False)
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        raise RuntimeError("YOLOv11x detected nothing in this image — could not classify.")

    boxes = results[0].boxes
    confidences = boxes.conf.tolist()
    class_ids = boxes.cls.tolist()
    best_i = max(range(len(confidences)), key=lambda i: confidences[i])
    label = results[0].names[int(class_ids[best_i])]
    confidence = float(confidences[best_i])

    print(f"[Vision Models] YOLOv11x prediction: {label} ({confidence:.4f})")

    return {
        "disease": label,
        "confidence": confidence,
        "source_model": YOLO_MODEL_ID,
    }