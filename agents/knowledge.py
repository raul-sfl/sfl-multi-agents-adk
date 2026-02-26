import asyncio
import logging
from google.adk.agents import LlmAgent
from agents.utils import transfer_to_triage
from agents.constants import STAYFORLONG_CONTACT
import config

logger = logging.getLogger(__name__)

_contact = STAYFORLONG_CONTACT


async def _call_agent(session_id: str, text: str, language_code: str = "en") -> str:
    """Call the Vertex AI Agent Designer agent via Dialogflow CX detect_intent."""
    if not config.HELP_CENTER_AGENT_ID:
        return ""  # Caller will use fallback contact message

    from google.cloud.dialogflowcx_v3beta1.services.sessions import SessionsAsyncClient
    from google.cloud.dialogflowcx_v3beta1.types import (
        DetectIntentRequest,
        QueryInput,
        TextInput,
    )

    # Extract location from HELP_CENTER_AGENT_ID
    # Format: projects/PROJECT/locations/LOCATION/agents/AGENT_ID
    parts = config.HELP_CENTER_AGENT_ID.split("/")
    location = parts[3] if len(parts) >= 6 else "us-central1"

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

    # Collect all text response messages
    parts_text = []
    for msg in response.query_result.response_messages:
        if msg.text and msg.text.text:
            parts_text.extend(msg.text.text)
    return " ".join(parts_text).strip()


def query_help_center(question: str, session_id: str = "default") -> str:
    """Ask the Stayforlong help center agent (powered by Vertex AI Agent Designer) about
    platform FAQs, policies, payment methods, minimum stay rules, stay extensions,
    cancellation policies and general platform questions.
    Accepts any question in Spanish or English."""
    try:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, _call_agent(session_id, question))
            result = future.result(timeout=20)
        if result:
            return result
        return (
            "No specific information was found for that in the help center. "
            f"Please contact our team: 📞 {_contact['phone']} | "
            f"✉️ {_contact['email']} | {_contact['hours']}"
        )
    except Exception as exc:
        logger.error("query_help_center failed: %s", exc)
        return (
            "Could not reach the help center at this moment. "
            f"Please contact Stayforlong: 📞 {_contact['phone']} | ✉️ {_contact['email']}"
        )


knowledge_agent = LlmAgent(
    name="HelpCenter",
    model=config.GEMINI_MODEL,
    instruction=(
        "You are the Stayforlong help center specialist. Always respond in {lang_name}. "
        "You have access to our conversational agent powered by Vertex AI Agent Designer.\n\n"

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
        "• Present the answer clearly in {lang_name}.\n"
        "• If query_help_center returns no relevant answer, provide the support contact:\n"
        f"  📞 {_contact['phone']}  |  ✉️ {_contact['email']}  |  {_contact['hours']}\n"
        "• IMPORTANT: For anything outside your scope, call transfer_to_triage IMMEDIATELY."
    ),
    tools=[query_help_center, transfer_to_triage],
)
