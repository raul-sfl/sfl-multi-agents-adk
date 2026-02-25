import asyncio
import logging
from google.adk.agents import LlmAgent
from google.adk.tools.tool_context import ToolContext
from agents.constants import STAYFORLONG_CONTACT
import config

logger = logging.getLogger(__name__)

_contact = STAYFORLONG_CONTACT


async def _call_vertex_agent(session_id: str, text: str, language_code: str = "es") -> str:
    """Call the Vertex AI Agent Builder conversational agent via Dialogflow CX API."""
    if not config.HELP_CENTER_AGENT_ID:
        return (
            "El servicio de ayuda no está configurado aún. "
            f"Contacta con Stayforlong: 📞 {_contact['phone']} | ✉️ {_contact['email']}"
        )

    try:
        from google.cloud.dialogflowcx_v3beta1.services.sessions import SessionsAsyncClient
        from google.cloud.dialogflowcx_v3beta1.types import (
            DetectIntentRequest,
            QueryInput,
            TextInput,
        )

        location = config.GOOGLE_CLOUD_LOCATION or "us-central1"
        client = SessionsAsyncClient(
            client_options={"api_endpoint": f"{location}-dialogflow.googleapis.com"}
        )

        session_path = f"{config.HELP_CENTER_AGENT_ID}/sessions/{session_id}"
        request = DetectIntentRequest(
            session=session_path,
            query_input=QueryInput(
                text=TextInput(text=text),
                language_code=language_code,
            ),
        )

        response = await client.detect_intent(request=request)
        for msg in response.query_result.response_messages:
            if msg.text and msg.text.text:
                return msg.text.text[0]

        return (
            "No he encontrado información específica sobre eso en nuestro centro de ayuda. "
            f"Puedes contactar con nuestro equipo: 📞 {_contact['phone']} | ✉️ {_contact['email']} | {_contact['hours']}"
        )

    except Exception as exc:
        logger.error("Error calling Vertex AI Agent Builder: %s", exc)
        return (
            "No he podido consultar el centro de ayuda en este momento. "
            f"Por favor contacta con Stayforlong: 📞 {_contact['phone']} | ✉️ {_contact['email']}"
        )


def query_help_center(question: str, session_id: str = "default") -> str:
    """Search the Stayforlong help center for FAQs, platform policies, payment methods,
    minimum stay rules, extensions, cancellation policies and general platform questions.
    Accepts any question in Spanish or English."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Running inside an async context (e.g. uvicorn); schedule coroutine
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run, _call_vertex_agent(session_id, question)
                )
                return future.result(timeout=15)
        else:
            return loop.run_until_complete(_call_vertex_agent(session_id, question))
    except Exception as exc:
        logger.error("query_help_center failed: %s", exc)
        return (
            "No he podido consultar el centro de ayuda. "
            f"Contacta con Stayforlong: 📞 {_contact['phone']} | ✉️ {_contact['email']}"
        )


def transfer_to_triage(tool_context: ToolContext) -> dict:
    """Transfer the conversation back to the main Stayforlong assistant for booking,
    incident or property questions."""
    tool_context.actions.transfer_to_agent = "Triage"
    return {"status": "transferred"}


knowledge_agent = LlmAgent(
    name="HelpCenter",
    model=config.GEMINI_MODEL,
    instruction=(
        "You are the Stayforlong help center specialist. "
        "You have access to our complete knowledge base via Vertex AI.\n\n"

        "SCOPE — what you handle:\n"
        "✅ General platform questions: how Stayforlong works, what it is, who it's for\n"
        "✅ Policies: cancellation policies, payment methods, deposit rules\n"
        "✅ Stay rules: minimum stay, extensions, early check-out\n"
        "✅ Billing: invoices, VAT, payment issues\n"
        "✅ Account: registration, login, profile management\n"
        "✅ FAQs: any general question about the platform\n\n"

        "OUT OF SCOPE — call transfer_to_triage IMMEDIATELY, never attempt to answer:\n"
        "🔄 Specific reservation details or booking IDs → transfer to Booking\n"
        "🔄 Active incidents, maintenance problems, complaints → transfer to Support\n"
        "🔄 Specific property amenities, check-in times, facilities → transfer to Alojamientos\n\n"

        "INSTRUCTIONS:\n"
        "• For EVERY question within your scope, ALWAYS call query_help_center first.\n"
        "• Present the answer clearly in the same language the user writes in.\n"
        "• If query_help_center returns no relevant answer, provide the support contact:\n"
        f"  📞 {_contact['phone']}  |  ✉️ {_contact['email']}  |  {_contact['hours']}\n"
        "• IMPORTANT: For anything outside your scope, call transfer_to_triage IMMEDIATELY."
    ),
    tools=[query_help_center, transfer_to_triage],
)
