import logging
from agents.utils import transfer_to_triage
from agents.constants import STAYFORLONG_CONTACT
import config
from agents.plugin import AgentPlugin

logger = logging.getLogger(__name__)

_contact = STAYFORLONG_CONTACT


def _search_vertex_ai(query: str) -> str:
    """Query the Vertex AI Search data store and return a synthesised answer."""
    from google.cloud import discoveryengine_v1 as discoveryengine

    project = config.GOOGLE_CLOUD_PROJECT
    datastore_id = config.VERTEX_AI_SEARCH_ENGINE_ID

    client = discoveryengine.SearchServiceClient()

    base = f"projects/{project}/locations/global/collections/default_collection"
    candidate_configs = [
        f"{base}/engines/{datastore_id}/servingConfigs/default_search",
        f"{base}/engines/{datastore_id}/servingConfigs/default_config",
        f"{base}/dataStores/{datastore_id}/servingConfigs/default_search",
        f"{base}/dataStores/{datastore_id}/servingConfigs/default_config",
    ]

    def _build_request(serving_config: str) -> discoveryengine.SearchRequest:
        return discoveryengine.SearchRequest(
            serving_config=serving_config,
            query=query,
            page_size=5,
            content_search_spec=discoveryengine.SearchRequest.ContentSearchSpec(
                summary_spec=discoveryengine.SearchRequest.ContentSearchSpec.SummarySpec(
                    summary_result_count=5,
                    include_citations=False,
                    language_code="es",
                ),
                snippet_spec=discoveryengine.SearchRequest.ContentSearchSpec.SnippetSpec(
                    return_snippet=True,
                ),
            ),
            query_expansion_spec=discoveryengine.SearchRequest.QueryExpansionSpec(
                condition=discoveryengine.SearchRequest.QueryExpansionSpec.Condition.AUTO,
            ),
        )

    response = None
    last_exc = None
    for cfg in candidate_configs:
        try:
            logger.info("Trying serving_config: %s", cfg)
            response = client.search(_build_request(cfg))
            logger.info("Success with serving_config: %s", cfg)
            break
        except Exception as exc:
            logger.warning("Failed serving_config %s: %s", cfg, exc)
            last_exc = exc

    if response is None:
        raise last_exc

    # Prefer the auto-generated summary when available
    summary_text = ""
    if response.summary and response.summary.summary_text:
        summary_text = response.summary.summary_text.strip()

    if summary_text:
        return summary_text

    # Fall back to concatenating the top snippets
    snippets = []
    for result in response.results:
        doc_data = result.document.derived_struct_data
        for snippet_item in doc_data.get("snippets", []):
            snippet = snippet_item.get("snippet", "").strip()
            if snippet:
                snippets.append(snippet)
    if snippets:
        return "\n\n".join(snippets[:3])

    return ""


def query_help_center(question: str) -> str:
    """Search the Stayforlong help center (powered by Vertex AI Search) for answers about
    platform FAQs, policies, payment methods, minimum stay rules, stay extensions,
    cancellation policies and general platform questions.
    Accepts any question in Spanish or English."""
    try:
        result = _search_vertex_ai(question)
        if result:
            return result
        return (
            "No specific information was found for that in the help center. "
            f"Please contact our team: 📞 {_contact['phone']} | "
            f"✉️ {_contact['email']} | {_contact['hours']}"
        )
    except Exception as exc:
        logger.error("query_help_center failed: %s", exc, exc_info=True)
        return (
            "Could not reach the help center at this moment. "
            f"Please contact Stayforlong: 📞 {_contact['phone']} | ✉️ {_contact['email']}"
        )


PLUGIN = AgentPlugin(
    name="HelpCenter",
    routing_hint=(
        "General platform FAQs, policies, payment methods, minimum stay, "
        "extensions, account questions, and any topic not covered by other specialists"
    ),
    instruction=(
        "You are the Stayforlong help center specialist. Always respond in {lang_name}. "
        "You have been transferred from the main assistant — the user's question is already in the conversation. "
        "NEVER greet the user or say 'Hola' / 'Hello' / 'How can I help' — go straight to answering.\n\n"

        "SCOPE — what you handle:\n"
        "✅ General platform questions: how Stayforlong works, what it is, who it's for\n"
        "✅ Policies: cancellation policies, payment methods, deposit rules\n"
        "✅ Stay rules: minimum stay, extensions, early check-out\n"
        "✅ Billing: invoices, VAT, payment issues\n"
        "✅ Account: registration, login, profile management\n"
        "✅ FAQs: any general question about the platform\n"
        "✅ Any question not handled by other specialists — you are the final fallback\n\n"

        "TRANSFER to another specialist ONLY for these specific cases:\n"
        "🔄 User provides a booking ID (SFL-XXXX-NNN) or email and asks for their specific "
        "reservation details → call transfer_to_triage\n"
        "🔄 Active incidents, maintenance problems, complaints during stay → call transfer_to_triage\n"
        "🔄 Specific property amenities, check-in times, facilities → call transfer_to_triage\n"
        "• If user asks about their reservation WITHOUT providing a booking ID or email, "
        "ask them to provide it: 'Para consultar tu reserva específica, necesito tu ID de reserva "
        "(formato SFL-XXXX-NNN) o tu email.'\n\n"

        "INSTRUCTIONS:\n"
        "• For questions within your scope, call query_help_center first.\n"
        "• Present the answer clearly in {lang_name}.\n"
        "• For anything truly unknown or unanswerable, NEVER call transfer_to_triage — "
        "instead provide the support contact directly:\n"
        f"  📞 {_contact['phone']}  |  ✉️ {_contact['email']}  |  {_contact['hours']}\n"
        "• You are the last resort: always resolve or provide contact info, never leave the guest without an answer."
    ),
    model=config.GEMINI_MODEL,
    is_fallback=True,  # HelpCenter es el fallback de último recurso
    get_tools=lambda: [query_help_center, transfer_to_triage],
)
