"""
tools/multimodal.py
-------------------
Multimodal input processing:
  - Image analysis via Gemini Vision
  - Audio transcription via faster-whisper
"""

from __future__ import annotations

import base64
import io
import time
import tempfile
from pathlib import Path

import structlog

from config.settings import settings
from models.schemas import AudioTranscription, ImageAnalysis

log = structlog.get_logger()


# ── Image Analysis (Gemini Vision) ────────────────────────────────────────────

async def analyse_image(image_bytes: bytes, mime_type: str = "image/jpeg") -> ImageAnalysis:
    """
    Send an image to Gemini Vision and extract:
    - Natural language description
    - Any text visible in the image (OCR)
    - Named entities / key subjects
    """
    from google import genai
    from google.genai import types

    t0 = time.perf_counter()
    client = genai.Client(api_key=settings.gemini_api_key.get_secret_value())

    prompt = (
        "Analyse this image thoroughly. Return a JSON object with these exact keys:\n"
        '{"description": "...", "extracted_text": "...", "detected_entities": ["..."]}\n'
        "extracted_text: any text visible in the image (empty string if none).\n"
        "detected_entities: list of key subjects, people, logos, brands, or topics.\n"
        "Return ONLY the JSON object, no markdown."
    )

    image_part = types.Part.from_bytes(
        data=image_bytes,
        mime_type=mime_type,
    )

    try:
        import json
        response = await client.aio.models.generate_content(
            model=settings.gemini_vision_model,
            contents=[prompt, image_part],
        )
        raw = response.text.strip().lstrip("```json").rstrip("```")
        data = json.loads(raw)
        analysis = ImageAnalysis(
            description=data.get("description", ""),
            extracted_text=data.get("extracted_text") or None,
            detected_entities=data.get("detected_entities", []),
            model_used=settings.gemini_vision_model,
        )
    except Exception as exc:
        log.warning("image_analysis_failed", error=str(exc))
        analysis = ImageAnalysis(
            description="Image could not be analysed.",
            model_used=settings.gemini_vision_model,
        )

    latency_ms = (time.perf_counter() - t0) * 1000
    log.info("image_analysed", latency_ms=round(latency_ms, 1), entities=len(analysis.detected_entities))
    return analysis


# ── Audio Transcription (faster-whisper) ──────────────────────────────────────

_whisper_model = None


def _get_whisper():
    """Lazy-load the faster-whisper model (cached after first call)."""
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        log.info("loading_whisper", size=settings.whisper_model_size, device=settings.whisper_device)
        _whisper_model = WhisperModel(
            settings.whisper_model_size,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
        )
    return _whisper_model


async def transcribe_audio(audio_bytes: bytes, filename: str = "audio.wav") -> AudioTranscription:
    """
    Transcribe audio bytes using faster-whisper.
    Writes to a temp file (whisper needs a file path), then cleans up.
    """
    import asyncio

    t0 = time.perf_counter()
    suffix = Path(filename).suffix or ".wav"

    def _sync_transcribe() -> AudioTranscription:
        model = _get_whisper()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            segments, info = model.transcribe(tmp_path, beam_size=5)
            text = " ".join(seg.text.strip() for seg in segments)
            return AudioTranscription(
                transcript=text,
                language=info.language,
                duration_s=round(info.duration, 2),
                model_used=f"faster-whisper-{settings.whisper_model_size}",
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(None, _sync_transcribe)
    except Exception as exc:
        log.warning("transcription_failed", error=str(exc))
        result = AudioTranscription(transcript="", model_used="faster-whisper")

    latency_ms = (time.perf_counter() - t0) * 1000
    log.info("audio_transcribed", language=result.language, duration_s=result.duration_s,
             latency_ms=round(latency_ms, 1))
    return result
