import base64
import io
import logging
import re
import shutil
import wave
from pathlib import Path
from typing import Any

import numpy as np
from huggingface_hub import HfApi, hf_hub_download

logger = logging.getLogger("indic_tts")

MODEL_ID = "ai4bharat/IndicF5"
REFERENCE_TEXT = {
    "hi": "नमस्ते, मैं कृषि सहायता के लिए तैयार हूँ।",
    "mr": "नमस्कार, मी कृषी सहाय्यास तयार आहे.",
}

_model_cache: Any = None


def clean_text_for_speech(text: str) -> str:
    """Remove Markdown syntax that IndicF5 would otherwise pronounce literally."""
    return re.sub(r"\s+", " ", re.sub(r"[*_~`#>]", "", text)).strip()


def _load_model():
    global _model_cache
    if _model_cache is not None:
        return _model_cache
    try:
        from transformers import AutoModel
    except ModuleNotFoundError as exc:  # pragma: no cover - import-time guard
        raise RuntimeError(
            "transformers is required for IndicF5 TTS. Install the python dependencies first."
        ) from exc

    _model_cache = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
    return _model_cache


def _candidate_reference_paths(language: str) -> list[Path]:
    project_root = Path(__file__).resolve().parent.parent
    base_dirs = [
        Path(__file__).resolve().parent,
        project_root,
    ]
    names = [
        f"{language}_reference.wav",
        f"{language}_ref.wav",
        f"{language}_prompt.wav",
        f"ref_{language}.wav",
        f"reference_{language}.wav",
    ]
    paths: list[Path] = []
    for base_dir in base_dirs:
        for search_path in [base_dir, base_dir / "assets", base_dir / "voice_prompts", base_dir / "sample_voices"]:
            for name in names:
                paths.append(search_path / name)
        paths.extend(list(base_dir.rglob("*.wav")))
    seen: set[Path] = set()
    ordered: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        ordered.append(resolved)
    return ordered


def _download_reference_audio(language: str) -> Path:
    workspace_dir = Path(__file__).resolve().parent / "voice_prompts"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    try:
        api = HfApi()
        files = api.list_repo_files(repo_id=MODEL_ID, repo_type="model")
        candidate_files = [
            file_name
            for file_name in files
            if file_name.lower().endswith(".wav")
            and ("prompt" in file_name.lower() or "ref" in file_name.lower())
            and language.lower() in file_name.lower()
        ]
        if not candidate_files:
            candidate_files = [
                file_name
                for file_name in files
                if file_name.lower().endswith(".wav")
                and ("prompt" in file_name.lower() or "ref" in file_name.lower())
            ]
        if not candidate_files:
            raise FileNotFoundError("No reference WAV files were found in the IndicF5 model repo.")
        downloaded_path = hf_hub_download(repo_id=MODEL_ID, filename=candidate_files[0], repo_type="model")
        destination = workspace_dir / Path(downloaded_path).name
        shutil.copy2(downloaded_path, destination)
        return destination
    except Exception as exc:  # pragma: no cover - model repo may be inaccessible at runtime
        raise FileNotFoundError(
            f"No reference audio prompt was found for {language}. Add a Hindi/Marathi WAV prompt under the project to enable IndicF5 TTS."
        ) from exc


def _find_reference_audio(language: str) -> Path:
    language = language.lower()
    for path in _candidate_reference_paths(language):
        if path.is_file() and path.stat().st_size > 0:
            return path
    return _download_reference_audio(language)


def _nativize_audio(audio: Any) -> np.ndarray:
    arr = np.asarray(audio)
    if arr.size == 0:
        raise ValueError("IndicF5 returned empty audio.")
    if arr.ndim == 2:
        arr = arr[0]
    arr = arr.astype(np.float32)
    arr = np.asarray(arr)
    if np.max(np.abs(arr)) > 1.0:
        arr = arr / max(float(np.max(np.abs(arr))), 1e-6)
    arr = np.clip(arr, -1.0, 1.0)
    return arr


def _audio_to_wav_bytes(audio: Any) -> bytes:
    normalized = _nativize_audio(audio)
    pcm = np.int16(normalized * 32767.0)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(pcm.tobytes())
    return buffer.getvalue()


def generate_audio_base64(text: str, language: str) -> str:
    language = (language or "hi").lower()
    if language not in {"hi", "mr"}:
        raise ValueError(f"IndicF5 is configured for Hindi and Marathi only. Got: {language}")

    cleaned = clean_text_for_speech(text or "")
    if not cleaned:
        raise ValueError("Text to speak is required.")

    model = _load_model()
    ref_audio_path = str(_find_reference_audio(language))
    ref_text = REFERENCE_TEXT[language]
    logger.info("Generating IndicF5 TTS audio for %s language using reference %s", language, ref_audio_path)

    generated = model(
        cleaned,
        ref_audio_path=ref_audio_path,
        ref_text=ref_text,
    )
    wav_bytes = _audio_to_wav_bytes(generated)
    return base64.b64encode(wav_bytes).decode("ascii")
