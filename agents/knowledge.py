import asyncio
import logging
from google.adk.agents import LlmAgent
from google.adk.tools.tool_context import ToolContext
from agents.constants import STAYFORLONG_CONTACT
import config

logger = logging.getLogger(__name__)

_contact = STAYFORLONG_CONTACT


async def _search_vertex(query: str) -> str:
    """Query Vertex AI Search and return the LLM-generated summary."""
    if not config.HELP_CENTER_SEARCH_ENGINE_ID:
        return ""  # Caller will use fallback contact message

    from google.cloud import discoveryengine_v1 as discoveryengine

    client = discoveryengine.SearchServiceAsyncClient()
    serving_config = (
        f"projects/{config.GOOGLE_CLOUD_PROJECT}"
        f"/locations/global/collections/default_collection"
        f"/engines/{config.HELP_CENTER_SEARCH_ENGINE_ID}"
        f"/servingConfigs/default_config"
    )
    request = discoveryengine.SearchRequest(
        serving_config=serving_config,
        query=query,
        page_size=5,
        content_search_spec=discoveryengine.SearchRequest.ContentSearchSpec(
            summary_spec=discoveryengine.SearchRequest.ContentSearchSpec.SummarySpec(
                summary_result_count=3,
                language_code="es",
            ),
            snippet_spec=discoveryengine.SearchRequest.ContentSearchSpec.SnippetSpec(
                return_snippet=True,
            ),
        ),
    )
    response = await client.search(request)

    # Prefer AI-generated summary from the search results
    if response.summary and response.summary.summary_text:
        return response.summary.summary_text

    # Fallback: collect top snippets from individual results
    snippets = []
    for result in response.results:
        doc = result.document
        if doc.derived_struct_data:
            for s in doc.derived_struct_data.get("snippets", []):
                text = s.get("snippet", "").strip()
                if text:
                    snippets.append(text)
        if len(snippets) >= 3:
            break

    if snippets:
        return "\n\n".join(snippets)

    return ""


def query_help_center(question: str) -> str:
    """Search the Stayforlong help center for FAQs, platform policies, payment methods,
    minimum stay rules, stay extensions, cancellation policies and general platform questions.
    Accepts any question in Spanish or English."""
    try:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, _search_vertex(question))
            result = future.result(timeout=15)
        if result:
            return result
        return (
            "No he encontrado información específica sobre eso en el help center. "
            f"Contacta con nuestro equipo: 📞 {_contact['phone']} | "
            f"✉️ {_contact['email']} | {_contact['hours']}"
        )
    except Exception as exc:
        logger.error("query_help_center failed: %s", exc)
        return (
            "No he podido consultar el centro de ayuda en este momento. "
            f"Por favor contacta con Stayforlong: 📞 {_contact['phone']} | ✉️ {_contact['email']}"
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
        "You have access to our complete knowledge base via Vertex AI Search.\n\n"

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
