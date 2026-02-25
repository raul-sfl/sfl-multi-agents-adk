import logging
from google.genai import types as genai_types
from fastapi import WebSocket, WebSocketDisconnect
from orchestrator.adk_runner import runner, session_service

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

LANG_NAMES = {
    "es": "Spanish", "en": "English", "pt": "Portuguese",
    "fr": "French", "de": "German", "it": "Italian", "ca": "Catalan",
}


async def websocket_endpoint(websocket: WebSocket, lang: str = "en"):
    await websocket.accept()

    supported_lang = lang if lang in WELCOME_MESSAGES else "en"
    lang_name = LANG_NAMES.get(supported_lang, "English")

    # Create ADK session with language context in state
    adk_session = await session_service.create_session(
        app_name="stayforlong",
        user_id=f"ws-{id(websocket)}",
        state={"lang": supported_lang, "lang_name": lang_name},
    )
    session_id = adk_session.id

    await websocket.send_json({
        "type": "session_init",
        "session_id": session_id,
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

            await websocket.send_json({
                "type": "typing",
                "agent": "Stayforlong",
            })

            try:
                content = genai_types.Content(
                    role="user",
                    parts=[genai_types.Part(text=user_message)],
                )

                reply_text = ""
                reply_agent = "Stayforlong"

                async for event in runner.run_async(
                    user_id=f"ws-{id(websocket)}",
                    session_id=session_id,
                    new_message=content,
                ):
                    # Update typing indicator with active agent on intermediate events
                    if event.author and not event.is_final_response():
                        await websocket.send_json({
                            "type": "typing",
                            "agent": event.author,
                        })

                    # Capture final response
                    if event.is_final_response():
                        if event.content and event.content.parts:
                            for part in event.content.parts:
                                if hasattr(part, "text") and part.text:
                                    reply_text += part.text
                        if event.author:
                            reply_agent = event.author

            except Exception as e:
                logger.error(f"Error in ADK run_async: {e}", exc_info=True)
                await websocket.send_json({
                    "type": "error",
                    "content": "Ha ocurrido un error interno. Por favor, inténtalo de nuevo.",
                })
                continue

            if reply_text:
                await websocket.send_json({
                    "type": "message",
                    "content": reply_text,
                    "agent": reply_agent,
                })

    except WebSocketDisconnect:
        logger.info(f"Session {session_id} disconnected.")
        await session_service.delete_session(
            app_name="stayforlong",
            user_id=f"ws-{id(websocket)}",
            session_id=session_id,
        )
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
