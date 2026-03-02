"""
Voice WebSocket endpoint — /ws/voice

Flow per turn:
  1. Receive JSON { audio_b64, lang }  from browser
  2. Vertex AI Speech-to-Text  →  transcript text
  3. Send { type: "transcript", text }  to browser immediately
  4. ADK Runner.run_async() streaming:
       - forward typing events
       - split partial reply on sentence boundaries
       - TTS-synthesize each complete sentence → send audio_chunk
  5. After final event: synthesize any leftover text
  6. Send { type: "message", content, agent }  (full text)
  7. Send { type: "audio_done" }

Sessions reuse the same ADK session_service and Runner as the text chat.
"""

import re
import uuid
import base64
import asyncio
import logging
from typing import Optional

from fastapi import WebSocket, WebSocketDisconnect
from google.genai import types as genai_types

from orchestrator.adk_runner import get_runner, session_service
from services.conversation_logger import conversation_logger
import config

logger = logging.getLogger(__name__)

# ── Language → TTS voice mapping ─────────────────────────────────────────────
# Neural2 voices give the best quality; Standard used as fallback.
_VOICE_MAP: dict[str, str] = {
    "es": "es-ES-Neural2-A",
    "en": "en-US-Neural2-F",
    "fr": "fr-FR-Neural2-A",
    "de": "de-DE-Neural2-F",
    "it": "it-IT-Neural2-A",
    "pt": "pt-PT-Neural2-A",
    "ca": "ca-ES-Standard-A",
}

# ── Sentence splitter ─────────────────────────────────────────────────────────
_SENTENCE_END = re.compile(r'(?<=[.!?])\s+')


def _split_sentences(text: str) -> tuple[list[str], str]:
    """
    Split text into complete sentences + remainder.
    Returns (complete_sentences, leftover).
    """
    parts = _SENTENCE_END.split(text)
    if len(parts) <= 1:
        return [], text
    return parts[:-1], parts[-1]


# ── STT client (lazy singleton) ───────────────────────────────────────────────
_stt_client = None


def _get_stt_client():
    global _stt_client
    if _stt_client is None:
        from google.cloud import speech
        _stt_client = speech.SpeechClient()
    return _stt_client


# ── TTS client (lazy singleton) ───────────────────────────────────────────────
_tts_client = None


def _get_tts_client():
    global _tts_client
    if _tts_client is None:
        from google.cloud import texttospeech
        _tts_client = texttospeech.TextToSpeechClient()
    return _tts_client


# ── STT: transcribe audio bytes ───────────────────────────────────────────────

def _transcribe(audio_bytes: bytes, lang: str) -> str:
    """
    Send audio to Cloud Speech-to-Text and return the transcript.
    Accepts WebM/Opus as produced by MediaRecorder in Chrome/Firefox/Safari.
    """
    from google.cloud import speech

    client = _get_stt_client()
    audio = speech.RecognitionAudio(content=audio_bytes)

    # BCP-47 language code: "es" → "es-ES", "en" → "en-US", etc.
    lang_map = {
        "es": "es-ES", "en": "en-US", "fr": "fr-FR",
        "de": "de-DE", "it": "it-IT", "pt": "pt-PT", "ca": "ca-ES",
    }
    language_code = lang_map.get(lang, f"{lang}-{lang.upper()}")

    config_stt = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
        sample_rate_hertz=48000,
        language_code=language_code,
        enable_automatic_punctuation=True,
        model="latest_short",
    )

    try:
        response = client.recognize(config=config_stt, audio=audio)
        parts = [r.alternatives[0].transcript for r in response.results if r.alternatives]
        return " ".join(parts).strip()
    except Exception as exc:
        logger.warning("STT recognition failed: %s", exc)
        return ""


# ── TTS: synthesize text → audio bytes ───────────────────────────────────────

def _synthesize(text: str, lang: str) -> bytes:
    """
    Synthesize text with Cloud Text-to-Speech.
    Returns raw audio bytes (MP3 or OGG_OPUS per config).
    """
    from google.cloud import texttospeech

    client = _get_tts_client()

    # Determine voice name: explicit config override > per-language default
    voice_name = config.TTS_VOICE_NAME or _VOICE_MAP.get(lang, "en-US-Neural2-F")

    # Derive language code from voice name (first 5 chars, e.g. "es-ES")
    voice_lang = voice_name[:5] if len(voice_name) >= 5 else f"{lang}-{lang.upper()}"

    encoding_map = {
        "MP3": texttospeech.AudioEncoding.MP3,
        "OGG_OPUS": texttospeech.AudioEncoding.OGG_OPUS,
        "LINEAR16": texttospeech.AudioEncoding.LINEAR16,
    }
    audio_encoding = encoding_map.get(
        config.TTS_AUDIO_ENCODING.upper(), texttospeech.AudioEncoding.MP3
    )

    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code=voice_lang,
        name=voice_name,
    )
    audio_config = texttospeech.AudioConfig(audio_encoding=audio_encoding)

    try:
        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config,
        )
        return response.audio_content
    except Exception as exc:
        logger.warning("TTS synthesis failed for lang=%s voice=%s: %s", lang, voice_name, exc)
        return b""


# ── Main WebSocket handler ────────────────────────────────────────────────────

async def voice_websocket_endpoint(
    websocket: WebSocket,
    lang: str = "en",
    user_id: str = "",
):
    await websocket.accept()

    # Immediate ping to prevent Railway's idle-timeout from closing the connection
    await websocket.send_json({"type": "ping"})

    # Keepalive task
    async def _keepalive():
        while True:
            await asyncio.sleep(10)
            try:
                await websocket.send_json({"type": "ping"})
            except Exception:
                break

    _ping_task = asyncio.create_task(_keepalive())

    supported_lang = lang if lang in _VOICE_MAP else "en"

    if not user_id:
        user_id = f"anon_{uuid.uuid4().hex[:12]}"

    # ── Lazy session state ────────────────────────────────────────────────────
    session_id: Optional[str] = None
    conv_id: Optional[str] = None

    async def _ensure_session():
        nonlocal session_id, conv_id
        if session_id is not None:
            return
        adk_session = await session_service.create_session(
            app_name="stayforlong",
            user_id=user_id,
            state={"lang": supported_lang, "channel": "voice"},
        )
        session_id = adk_session.id
        conv_id = str(uuid.uuid4())
        await conversation_logger.log_conversation_start(
            conv_id, user_id, session_id, supported_lang, channel="voice"
        )
        await websocket.send_json({
            "type":       "session_init",
            "session_id": session_id,
            "user_id":    user_id,
            "lang":       supported_lang,
        })

    try:
        # ── Main loop ─────────────────────────────────────────────────────────
        while True:
            data = await websocket.receive_json()

            msg_type = data.get("type", "")

            # Ignore pings from client
            if msg_type == "ping":
                continue

            audio_b64: str = data.get("audio_b64", "")
            if not audio_b64:
                continue

            # ── 1. Decode audio ───────────────────────────────────────────────
            try:
                audio_bytes = base64.b64decode(audio_b64)
            except Exception as exc:
                logger.warning("Failed to decode audio_b64: %s", exc)
                await websocket.send_json({
                    "type": "error",
                    "content": "Invalid audio data.",
                })
                continue

            # ── 2. STT ────────────────────────────────────────────────────────
            try:
                transcript = await asyncio.to_thread(
                    _transcribe, audio_bytes, supported_lang
                )
            except Exception as exc:
                logger.error("STT error: %s", exc, exc_info=True)
                await websocket.send_json({
                    "type": "error",
                    "content": "Speech recognition failed. Please try again.",
                })
                continue

            if not transcript:
                await websocket.send_json({
                    "type": "transcript",
                    "text": "",
                    "error": "Could not understand audio.",
                })
                continue

            # ── 3. Send transcript immediately ────────────────────────────────
            await websocket.send_json({"type": "transcript", "text": transcript})

            # ── 4. Materialise session on first real message ──────────────────
            try:
                await _ensure_session()
            except Exception as exc:
                logger.error("Voice session creation failed: %s", exc, exc_info=True)
                await websocket.send_json({
                    "type": "error",
                    "content": "Could not start session. Please try again.",
                })
                continue

            await conversation_logger.log_message(
                conv_id, "user", transcript, channel="voice"
            )
            await websocket.send_json({"type": "typing", "agent": "Stayforlong"})

            # ── 5. ADK streaming + sentence-level TTS pipeline ────────────────
            try:
                content = genai_types.Content(
                    role="user",
                    parts=[genai_types.Part(text=transcript)],
                )

                reply_text = ""
                reply_agent = "Stayforlong"
                pending_text = ""   # accumulates partial reply for sentence splitting
                seq = 0             # audio chunk sequence number

                async def _send_tts_chunk(sentence: str, chunk_seq: int):
                    """Synthesize one sentence and send it as an audio_chunk."""
                    audio_bytes_out = await asyncio.to_thread(
                        _synthesize, sentence, supported_lang
                    )
                    if audio_bytes_out:
                        await websocket.send_json({
                            "type":      "audio_chunk",
                            "audio_b64": base64.b64encode(audio_bytes_out).decode(),
                            "seq":       chunk_seq,
                        })

                async for event in get_runner().run_async(
                    user_id=user_id,
                    session_id=session_id,
                    new_message=content,
                ):
                    if event.author and not event.is_final_response():
                        await websocket.send_json(
                            {"type": "typing", "agent": event.author}
                        )

                    if event.is_final_response():
                        if event.content and event.content.parts:
                            for part in event.content.parts:
                                if hasattr(part, "text") and part.text:
                                    new_text = part.text
                                    reply_text += new_text
                                    pending_text += new_text

                                    # Synthesize complete sentences eagerly
                                    sentences, pending_text = _split_sentences(pending_text)
                                    for sentence in sentences:
                                        sentence = sentence.strip()
                                        if sentence:
                                            await _send_tts_chunk(sentence, seq)
                                            seq += 1

                        if event.author:
                            reply_agent = event.author

                # Synthesize any remaining text after final event
                if pending_text.strip():
                    await _send_tts_chunk(pending_text.strip(), seq)

            except Exception as exc:
                logger.error("Error in voice ADK run_async: %s", exc, exc_info=True)
                await websocket.send_json({
                    "type": "error",
                    "content": "An internal error occurred. Please try again.",
                })
                continue

            # ── 6. Send full text reply ───────────────────────────────────────
            if reply_text:
                await websocket.send_json({
                    "type":    "message",
                    "content": reply_text,
                    "agent":   reply_agent,
                })
                await conversation_logger.log_message(
                    conv_id, "assistant", reply_text, reply_agent, channel="voice"
                )

            # ── 7. Signal end of audio stream ─────────────────────────────────
            await websocket.send_json({"type": "audio_done"})

    except WebSocketDisconnect:
        logger.info("Voice session %s (user %s) disconnected.", session_id, user_id)
        if conv_id:
            await conversation_logger.log_conversation_end(conv_id)
    except Exception as exc:
        logger.error("Voice WebSocket error: %s", exc, exc_info=True)
        if conv_id:
            await conversation_logger.log_conversation_end(conv_id)
    finally:
        _ping_task.cancel()
