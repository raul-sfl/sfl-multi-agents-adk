import logging
from fastapi import WebSocket, WebSocketDisconnect
from orchestrator.types import Session
from orchestrator.engine import run_turn
from agents.triage import triage_agent

logger = logging.getLogger(__name__)

# In-memory session store (sufficient for PoC)
sessions: dict[str, Session] = {}

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

LANG_NAMES = {
    "es": "Spanish", "en": "English", "pt": "Portuguese",
    "fr": "French", "de": "German", "it": "Italian", "ca": "Catalan",
}


async def websocket_endpoint(websocket: WebSocket, lang: str = "en"):
    await websocket.accept()

    # Normalise to a supported language, fallback to English
    supported_lang = lang if lang in WELCOME_MESSAGES else "en"
    lang_name = LANG_NAMES.get(supported_lang, "English")

    session = Session(
        active_agent=triage_agent,
        context={"lang": supported_lang, "lang_name": lang_name},
    )
    sessions[session.session_id] = session

    await websocket.send_json({
        "type": "session_init",
        "session_id": session.session_id,
        "lang": supported_lang,
    })
    await websocket.send_json({
        "type": "message",
        "content": WELCOME_MESSAGES[supported_lang],
        "agent": "Stayforlong Assistant",
    })

    try:
        while True:
            data = await websocket.receive_json()
            user_message = data.get("message", "").strip()
            if not user_message:
                continue

            # Send typing indicator
            await websocket.send_json({
                "type": "typing",
                "agent": session.active_agent.name,
            })

            try:
                reply = await run_turn(session, user_message)
            except Exception as e:
                logger.error(f"Error in run_turn: {e}", exc_info=True)
                await websocket.send_json({
                    "type": "error",
                    "content": "Ha ocurrido un error interno. Por favor, inténtalo de nuevo.",
                })
                continue

            await websocket.send_json({
                "type": "message",
                "content": reply,
                "agent": session.active_agent.name,
            })

    except WebSocketDisconnect:
        logger.info(f"Session {session.session_id} disconnected.")
        sessions.pop(session.session_id, None)
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        sessions.pop(session.session_id, None)
