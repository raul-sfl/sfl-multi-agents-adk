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
from langdetect import detect as _langdetect, LangDetectException
from langdetect import DetectorFactory as _DetectorFactory
_DetectorFactory.seed = 42  # deterministic results

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

# ── Language name map (must match what agent instructions expect) ────────────────
_LANG_NAMES: dict[str, str] = {
    "es": "Spanish", "en": "English", "fr": "French",
    "de": "German",  "it": "Italian", "pt": "Portuguese", "ca": "Catalan",
}

# ── BCP-47 language codes (shared by STT and TTS) ────────────────────────────
_LANG_BCP47: dict[str, str] = {
    "es": "es-ES", "en": "en-US", "fr": "fr-FR",
    "de": "de-DE", "it": "it-IT", "pt": "pt-PT", "ca": "ca-ES",
}

# ── STT model overrides (default: latest_short; ca not supported by latest_short) ──
_STT_MODEL: dict[str, str] = {
    "ca": "default",
}


def _detect_lang_from_text(text: str, fallback: str) -> str:
    """Detect language code from transcript text. Falls back to `fallback` on failure."""
    if not text.strip():
        return fallback
    try:
        detected = _langdetect(text)[:2].lower()
        return detected if detected in _LANG_BCP47 else fallback
    except LangDetectException:
        return fallback


# ── Sentence / clause splitter ───────────────────────────────────────────────
_SENTENCE_END = re.compile(r'(?<=[.!?])\s+')
_CLAUSE_END   = re.compile(r'(?<=[,;])\s+')
_MIN_CLAUSE_CHARS = 60  # only split on clause boundary when chunk is at least this long

# ── Markdown stripper (for TTS) ───────────────────────────────────────────────
_MD_BOLD_ITALIC = re.compile(r'\*{1,3}(.*?)\*{1,3}')
_MD_HEADING = re.compile(r'^\s*#{1,6}\s+', re.MULTILINE)
_MD_LINK = re.compile(r'\[([^\]]+)\]\([^)]+\)')
_MD_CODE = re.compile(r'`{1,3}[^`]*`{1,3}')
_MD_BULLET = re.compile(r'^\s*[-*+]\s+', re.MULTILINE)
_MD_NUMBERED = re.compile(r'^\s*\d+\.\s+', re.MULTILINE)
_MD_BLOCKQUOTE = re.compile(r'^\s*>\s+', re.MULTILINE)
_MD_HR = re.compile(r'^\s*[-*_]{3,}\s*$', re.MULTILINE)


def _strip_markdown(text: str) -> str:
    """Remove markdown formatting so TTS reads plain text."""
    text = _MD_BOLD_ITALIC.sub(r'\1', text)
    text = _MD_HEADING.sub('', text)
    text = _MD_LINK.sub(r'\1', text)
    text = _MD_CODE.sub('', text)
    text = _MD_BULLET.sub('', text)
    text = _MD_NUMBERED.sub('', text)
    text = _MD_BLOCKQUOTE.sub('', text)
    text = _MD_HR.sub('', text)
    return text.strip()


def _split_sentences(text: str) -> tuple[list[str], str]:
    """
    Split text into speakable chunks + remainder.
    Always splits on sentence boundaries (.!?).
    Also splits on clause boundaries (,;) when the accumulated text is long enough.
    Returns (complete_chunks, leftover).
    """
    parts = _SENTENCE_END.split(text)
    if len(parts) > 1:
        return parts[:-1], parts[-1]
    # No sentence boundary yet — try clause split if text is long enough
    if len(text) >= _MIN_CLAUSE_CHARS:
        clauses = _CLAUSE_END.split(text)
        if len(clauses) > 1:
            return clauses[:-1], clauses[-1]
    return [], text


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

def _transcribe(audio_bytes: bytes, lang: str) -> tuple[str, str]:
    """
    Send audio to Cloud Speech-to-Text and return (transcript, detected_lang).
    Uses alternative_language_codes so the API auto-detects the actual language spoken.
    Accepts WebM/Opus as produced by MediaRecorder in Chrome/Firefox/Safari.
    """
    from google.cloud import speech

    client = _get_stt_client()
    audio = speech.RecognitionAudio(content=audio_bytes)

    language_code = _LANG_BCP47.get(lang, f"{lang}-{lang.upper()}")
    alt_codes = [v for k, v in _LANG_BCP47.items() if k != lang]

    config_stt = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
        sample_rate_hertz=48000,
        language_code=language_code,
        alternative_language_codes=alt_codes,
        enable_automatic_punctuation=True,
        model=_STT_MODEL.get(lang, "latest_short"),
    )

    try:
        response = client.recognize(config=config_stt, audio=audio)
        parts = [r.alternatives[0].transcript for r in response.results if r.alternatives]
        transcript = " ".join(parts).strip()
        # Detect language from transcript text (more reliable than STT's language_code field,
        # which often returns the primary language even when an alternative was spoken).
        detected_lang = _detect_lang_from_text(transcript, lang)
        return transcript, detected_lang
    except Exception as exc:
        logger.warning("STT recognition failed: %s", exc)
        return "", lang


# ── TTS voice adapter ────────────────────────────────────────────────────────

def _adapt_voice_for_lang(base_voice: str, lang: str) -> str:
    """
    Adapt a voice name to the target language by substituting the lang-REGION prefix.
    Example: base="en-US-Chirp3-HD-Aoede", lang="es" → "es-ES-Chirp3-HD-Aoede"
    Falls back to _VOICE_MAP if lang unknown, base_voice too short, or format invalid.
    """
    lang_prefix = _LANG_BCP47.get(lang)
    if not lang_prefix or len(base_voice) < 7 or base_voice[5:6] != "-":
        return _VOICE_MAP.get(lang) or base_voice
    suffix = base_voice[6:]  # e.g. "Chirp3-HD-Aoede", "Neural2-A"
    return f"{lang_prefix}-{suffix}"


# ── TTS: synthesize text → audio bytes ───────────────────────────────────────

def _synthesize(text: str, lang: str) -> bytes:
    """
    Synthesize text with Cloud Text-to-Speech.
    Returns raw audio bytes (MP3 or OGG_OPUS per config).
    """
    from google.cloud import texttospeech

    client = _get_tts_client()

    # Determine voice name: adapt .env template to client language, or use per-lang map
    if config.TTS_VOICE_NAME:
        voice_name = _adapt_voice_for_lang(config.TTS_VOICE_NAME, lang)
    else:
        voice_name = _VOICE_MAP.get(lang, "en-US-Neural2-F")

    # Derive language code from voice name (first 5 chars, e.g. "es-ES")
    voice_lang = voice_name[:5] if len(voice_name) >= 5 else f"{lang}-{lang.upper()}"

    encoding_map = {
        "MP3": texttospeech.AudioEncoding.MP3,
        "OGG_OPUS": texttospeech.AudioEncoding.OGG_OPUS,
        "LINEAR16": texttospeech.AudioEncoding.LINEAR16,
    }
    # Chirp3-HD and Chirp-HD voices don't support MP3 — fall back to OGG_OPUS
    is_chirp = "Chirp" in voice_name
    default_encoding = "OGG_OPUS" if is_chirp else "MP3"
    audio_encoding = encoding_map.get(
        config.TTS_AUDIO_ENCODING.upper(), encoding_map[default_encoding]
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
    lang_name = _LANG_NAMES.get(supported_lang, "English")

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
            state={"lang": supported_lang, "lang_name": lang_name, "channel": "voice"},
        )
        session_id = adk_session.id
        conv_id = await conversation_logger.find_reusable_conversation_id(
            user_id=user_id,
            channel="voice",
        )
        if not conv_id:
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

    # Active pipeline task (cancelable on barge-in)
    _pipeline_task: Optional[asyncio.Task] = None

    async def _cancel_pipeline():
        nonlocal _pipeline_task
        if _pipeline_task and not _pipeline_task.done():
            _pipeline_task.cancel()
            try:
                await _pipeline_task
            except (asyncio.CancelledError, Exception):
                pass
        _pipeline_task = None

    async def _run_pipeline(audio_bytes: bytes, turn_lang: str):
        """Full STT → ADK → TTS pipeline for one voice turn."""
        nonlocal _pipeline_task

        # ── 2. STT ────────────────────────────────────────────────────────
        try:
            transcript, detected_lang = await asyncio.to_thread(
                _transcribe, audio_bytes, turn_lang
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("STT error: %s", exc, exc_info=True)
            await websocket.send_json({
                "type": "error",
                "content": "Speech recognition failed. Please try again.",
            })
            return

        if not transcript:
            await websocket.send_json({
                "type": "transcript",
                "text": "",
                "error": "Could not understand audio.",
            })
            return

        # ── 3. Send transcript immediately ────────────────────────────────
        await websocket.send_json({"type": "transcript", "text": transcript})

        # ── 4. Materialise session on first real message ──────────────────
        try:
            await _ensure_session()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("Voice session creation failed: %s", exc, exc_info=True)
            await websocket.send_json({
                "type": "error",
                "content": "Could not start session. Please try again.",
            })
            return

        await conversation_logger.log_message(
            conv_id, "user", transcript, channel="voice"
        )
        await websocket.send_json({"type": "typing", "agent": "Stayforlong"})

        # ── 5. ADK streaming + sentence-level TTS pipeline ────────────────
        # Inject per-turn language directive so agents respond in the detected
        # language, overriding the session-level lang_name (fixed at connection).
        if detected_lang != supported_lang:
            detected_lang_name = _LANG_NAMES.get(detected_lang, "English")
            message_text = f"[Respond in {detected_lang_name}]\n{transcript}"
        else:
            message_text = transcript
        content = genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=message_text)],
        )

        reply_text = ""
        reply_agent = "Stayforlong"
        pending_text = ""

        tts_queue: asyncio.Queue = asyncio.Queue()
        _SENTINEL = object()

        async def _tts_sender():
            """Consume synthesized audio from tts_queue and forward to browser."""
            s = 0
            while True:
                item = await tts_queue.get()
                if item is _SENTINEL:
                    break
                text_chunk, = item
                audio_bytes_out = await asyncio.to_thread(
                    _synthesize, text_chunk, detected_lang
                )
                if audio_bytes_out:
                    await websocket.send_json({
                        "type":      "audio_chunk",
                        "audio_b64": base64.b64encode(audio_bytes_out).decode(),
                        "seq":       s,
                    })
                    s += 1

        sender_task = asyncio.create_task(_tts_sender())

        try:
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

                                chunks, pending_text = _split_sentences(pending_text)
                                for chunk in chunks:
                                    chunk = _strip_markdown(chunk.strip())
                                    if chunk:
                                        await tts_queue.put((chunk,))

                    if event.author:
                        reply_agent = event.author

            if pending_text.strip():
                await tts_queue.put((_strip_markdown(pending_text.strip()),))

        except asyncio.CancelledError:
            sender_task.cancel()
            try:
                await sender_task
            except (asyncio.CancelledError, Exception):
                pass
            await websocket.send_json({"type": "interrupted"})
            raise
        except Exception as exc:
            logger.error("Error in voice ADK run_async: %s", exc, exc_info=True)
            await websocket.send_json({
                "type":    "error",
                "content": "An internal error occurred. Please try again.",
            })
        finally:
            if not sender_task.done():
                await tts_queue.put(_SENTINEL)
                try:
                    await sender_task
                except (asyncio.CancelledError, Exception):
                    pass

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

    try:
        # ── Main loop ─────────────────────────────────────────────────────────
        while True:
            data = await websocket.receive_json()

            msg_type = data.get("type", "")

            # Ignore pings from client
            if msg_type == "ping":
                continue

            # ── Barge-in: cancel active pipeline ──────────────────────────────
            if msg_type == "interrupt":
                await _cancel_pipeline()
                continue

            audio_b64: str = data.get("audio_b64", "")
            if not audio_b64:
                continue

            # Per-turn language (client sends current selector value with each audio message)
            raw_lang = data.get("lang", supported_lang)[:2].lower()
            turn_lang = raw_lang if raw_lang in _VOICE_MAP else supported_lang

            # Cancel any in-progress pipeline before starting new one
            await _cancel_pipeline()

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

            _pipeline_task = asyncio.create_task(_run_pipeline(audio_bytes, turn_lang))

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
