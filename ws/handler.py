import re
import uuid
import asyncio
import logging
from datetime import datetime, timezone, timedelta
from google import genai
from google.genai import types as genai_types
from fastapi import WebSocket, WebSocketDisconnect
from orchestrator.adk_runner import get_runner, session_service
from services.conversation_logger import conversation_logger

import config

logger = logging.getLogger(__name__)

WELCOME_MESSAGES = {
    "es": (
        "¡Hola! Soy el asistente virtual de Stayforlong.\n"
        "Puedo ayudarte con:\n"
        "• Consultas sobre tu reserva\n"
        "• Incidencias o problemas durante la estancia\n"
        "• Información sobre nuestras propiedades\n\n"
        "¿En qué puedo ayudarte hoy?"
    ),
    "en": (
        "Hello! I'm the Stayforlong virtual assistant.\n"
        "I can help you with:\n"
        "• Booking queries\n"
        "• Incidents or problems during your stay\n"
        "• Property information\n\n"
        "How can I help you today?"
    ),
    "pt": (
        "Olá! Sou o assistente virtual do Stayforlong.\n"
        "Posso ajudá-lo com:\n"
        "• Consultas sobre a sua reserva\n"
        "• Incidentes ou problemas durante a estadia\n"
        "• Informações sobre as nossas propriedades\n\n"
        "Como posso ajudá-lo hoje?"
    ),
    "fr": (
        "Bonjour ! Je suis l'assistant virtuel de Stayforlong.\n"
        "Je peux vous aider avec :\n"
        "• Questions sur votre réservation\n"
        "• Incidents ou problèmes pendant votre séjour\n"
        "• Informations sur nos propriétés\n\n"
        "Comment puis-je vous aider aujourd'hui ?"
    ),
    "de": (
        "Hallo! Ich bin der virtuelle Assistent von Stayforlong.\n"
        "Ich kann Ihnen helfen bei:\n"
        "• Fragen zu Ihrer Buchung\n"
        "• Vorfällen oder Problemen während Ihres Aufenthalts\n"
        "• Informationen zu unseren Unterkünften\n\n"
        "Wie kann ich Ihnen heute helfen?"
    ),
    "it": (
        "Ciao! Sono l'assistente virtuale di Stayforlong.\n"
        "Posso aiutarti con:\n"
        "• Domande sulla tua prenotazione\n"
        "• Incidenti o problemi durante il soggiorno\n"
        "• Informazioni sulle nostre strutture\n\n"
        "Come posso aiutarti oggi?"
    ),
    "ca": (
        "Hola! Soc l'assistent virtual de Stayforlong.\n"
        "Puc ajudar-te amb:\n"
        "• Consultes sobre la teva reserva\n"
        "• Incidències o problemes durant l'estada\n"
        "• Informació sobre les nostres propietats\n\n"
        "En què et puc ajudar avui?"
    ),
}

CONTINUATION_FALLBACK = {
    "es": "¡Bienvenido de nuevo! Veo que ya hemos hablado antes. ¿En qué puedo ayudarte hoy?",
    "en": "Welcome back! I can see we've spoken before. How can I help you today?",
    "pt": "Bem-vindo de volta! Vejo que já falámos antes. Como posso ajudá-lo hoje?",
    "fr": "Bon retour ! Je vois que nous avons déjà parlé. Comment puis-je vous aider aujourd'hui ?",
    "de": "Willkommen zurück! Ich sehe, dass wir schon gesprochen haben. Wie kann ich Ihnen heute helfen?",
    "it": "Bentornato! Vedo che abbiamo già parlato. Come posso aiutarti oggi?",
    "ca": "Benvingut de nou! Veig que ja hem parlat abans. En què et puc ajudar avui?",
}

_genai_client: genai.Client | None = None


def _get_genai_client() -> genai.Client:
    global _genai_client
    if _genai_client is None:
        _genai_client = genai.Client()
    return _genai_client


async def _build_continuation_greeting(history: list[dict], lang: str, lang_name: str) -> str:
    """Use the LLM to generate a context-aware greeting based on prior conversation."""
    # Use last 6 messages max to keep the prompt small
    recent = history[-6:]
    summary_lines = []
    for m in recent:
        role_label = "User" if m["role"] == "user" else "Assistant"
        summary_lines.append(f"{role_label}: {m['content'][:200]}")
    conversation_excerpt = "\n".join(summary_lines)

    prompt = (
        f"You are the Stayforlong virtual assistant. A returning user has reconnected. "
        f"Below is the end of their previous conversation.\n\n"
        f"{conversation_excerpt}\n\n"
        f"Write a SHORT (1-2 sentences) welcome-back greeting in {lang_name} that "
        f"naturally acknowledges the context of the previous conversation and invites "
        f"the user to continue or ask anything new. "
        f"Do NOT list your capabilities. Be warm and conversational."
    )

    try:
        client = _get_genai_client()
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=config.GEMINI_MODEL,
            contents=prompt,
        )
        text = response.text.strip() if response.text else ""
        if text:
            return text
    except Exception as exc:
        logger.warning("Continuation greeting generation failed: %s", exc)

    return CONTINUATION_FALLBACK.get(lang, CONTINUATION_FALLBACK["en"])


LANG_NAMES = {
    "es": "Spanish", "en": "English", "pt": "Portuguese",
    "fr": "French", "de": "German", "it": "Italian", "ca": "Catalan",
}

_GREETING_RE = re.compile(
    r"^\s*("
    r"hola|hello|hi|hey|hei|salut|bonjour|ciao|ola|ol[aá]|buenos\s+d[íi]as|"
    r"buenas\s+tardes|buenas\s+noches|buen\s+d[íi]a|good\s+morning|"
    r"good\s+afternoon|good\s+evening|good\s+day|guten\s+tag|guten\s+morgen|"
    r"guten\s+abend|buongiorno|buonasera|bon\s+matin|bonsoir|"
    r"ola|ol[aá]|bom\s+dia|boa\s+tarde|boa\s+noite|"
    r"howdy|sup|what'?s\s+up|greetings"
    r")[!.,\s]*$",
    re.IGNORECASE | re.UNICODE,
)


def _is_greeting(text: str) -> bool:
    """Return True if the message is nothing more than a simple greeting."""
    words = text.split()
    if len(words) > 6:
        return False
    return bool(_GREETING_RE.match(text.strip()))


async def _get_history(user_id: str) -> list[dict]:
    """
    Recover messages from the most recent Vertex AI session of this user.
    Returns an empty list on any error or when outside HISTORY_RECOVERY_HOURS.
    """
    try:
        response = await session_service.list_sessions(
            app_name="stayforlong",
            user_id=user_id,
        )
        sessions = getattr(response, "sessions", []) or []
        if not sessions:
            return []

        # Most recent session by last_update_time
        sessions_sorted = sorted(
            sessions,
            key=lambda s: getattr(s, "last_update_time", 0),
            reverse=True,
        )
        recent = sessions_sorted[0]

        # Enforce recovery window
        last_update = getattr(recent, "last_update_time", 0)
        if last_update:
            cutoff = (
                datetime.now(timezone.utc).timestamp()
                - config.HISTORY_RECOVERY_HOURS * 3600
            )
            if last_update < cutoff:
                return []

        # Fetch full session to read events
        full_session = await session_service.get_session(
            app_name="stayforlong",
            user_id=user_id,
            session_id=recent.id,
        )
        if not full_session:
            return []

        messages: list[dict] = []
        for event in getattr(full_session, "events", []):
            content = getattr(event, "content", None)
            if not content:
                continue
            role = getattr(content, "role", None)
            parts = getattr(content, "parts", []) or []
            text = "".join(
                t for p in parts
                if (t := getattr(p, "text", None)) is not None
            )
            if not text:
                continue
            author = getattr(event, "author", None)
            messages.append({
                "role":    "user" if role == "user" else "assistant",
                "content": text,
                "agent":   author if role != "user" else None,
            })
        return messages
    except Exception as exc:
        logger.warning("History recovery failed for user %s: %s", user_id, exc)
        return []


async def websocket_endpoint(websocket: WebSocket, lang: str = "en", user_id: str = ""):
    await websocket.accept()

    # Send an immediate ping so Railway's reverse-proxy doesn't close the
    # connection while we make the initial Vertex AI calls (history + session).
    await websocket.send_json({"type": "ping"})

    # Keepalive task: ping every 20 s to prevent Railway's idle timeout from
    # dropping long-lived connections with no user traffic.
    async def _keepalive():
        while True:
            await asyncio.sleep(20)
            try:
                await websocket.send_json({"type": "ping"})
            except Exception:
                break

    _ping_task = asyncio.create_task(_keepalive())

    supported_lang = lang if lang in WELCOME_MESSAGES else "en"
    lang_name = LANG_NAMES.get(supported_lang, "English")

    # Assign an anonymous ID when the client doesn't provide one
    if not user_id:
        user_id = f"anon_{uuid.uuid4().hex[:12]}"

    # ── History recovery from VertexAiSessionService ──────────────────────────
    history_messages = await _get_history(user_id)

    # ── Lazy state — session & conv created on first substantive message ──────
    session_id: str | None = None
    conv_id:    str | None = None

    async def _ensure_session():
        """Create ADK session + open Cloud Logging conv the first time needed."""
        nonlocal session_id, conv_id
        if session_id is not None:
            return
        adk_session = await session_service.create_session(
            app_name="stayforlong",
            user_id=user_id,
            state={"lang": supported_lang, "lang_name": lang_name},
        )
        session_id = adk_session.id
        conv_id = str(uuid.uuid4())
        await conversation_logger.log_conversation_start(
            conv_id, user_id, session_id, supported_lang
        )
        # Notify client of the real session_id now that it exists
        await websocket.send_json({
            "type":       "session_init",
            "session_id": session_id,
            "user_id":    user_id,
            "lang":       supported_lang,
            "history":    [],
        })

    # ── Send session_init (with history) before welcome — session_id TBD ─────
    await websocket.send_json({
        "type":       "session_init",
        "session_id": None,
        "user_id":    user_id,
        "lang":       supported_lang,
        "history":    history_messages,
    })

    # ── Welcome message ───────────────────────────────────────────────────────
    if history_messages:
        welcome_text = await _build_continuation_greeting(history_messages, supported_lang, lang_name)
    else:
        welcome_text = WELCOME_MESSAGES[supported_lang]
    await websocket.send_json({
        "type":    "message",
        "content": welcome_text,
        "agent":   "Stayforlong Assistant",
    })
    # Welcome is logged after the session is materialised (see message loop)

    # ── Main message loop ─────────────────────────────────────────────────────
    try:
        while True:
            data = await websocket.receive_json()
            user_message = data.get("message", "").strip()
            if not user_message:
                continue

            # ── Greeting gate: don't create a session for trivial openers ─────
            if session_id is None and _is_greeting(user_message):
                await websocket.send_json({
                    "type":    "message",
                    "content": welcome_text,
                    "agent":   "Stayforlong Assistant",
                })
                continue

            # ── Materialise session + conv on first substantive message ───────
            first_message = session_id is None
            await _ensure_session()

            if first_message:
                await conversation_logger.log_message(
                    conv_id, "assistant", welcome_text, "Stayforlong Assistant"
                )

            await conversation_logger.log_message(conv_id, "user", user_message)
            await websocket.send_json({"type": "typing", "agent": "Stayforlong"})

            try:
                content = genai_types.Content(
                    role="user",
                    parts=[genai_types.Part(text=user_message)],
                )

                reply_text = ""
                reply_agent = "Stayforlong"

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
                                    reply_text += part.text
                        if event.author:
                            reply_agent = event.author

            except Exception as e:
                logger.error("Error in ADK run_async: %s", e, exc_info=True)
                await websocket.send_json({
                    "type":    "error",
                    "content": "An internal error occurred. Please try again.",
                })
                continue

            if reply_text:
                await websocket.send_json({
                    "type":    "message",
                    "content": reply_text,
                    "agent":   reply_agent,
                })
                await conversation_logger.log_message(
                    conv_id, "assistant", reply_text, reply_agent
                )

    except WebSocketDisconnect:
        logger.info("Session %s (user %s) disconnected.", session_id, user_id)
        # Session kept in Vertex AI for history recovery on next connect
        if conv_id:
            await conversation_logger.log_conversation_end(conv_id)
    except Exception as e:
        logger.error("WebSocket error: %s", e, exc_info=True)
        if conv_id:
            await conversation_logger.log_conversation_end(conv_id)
    finally:
        _ping_task.cancel()
