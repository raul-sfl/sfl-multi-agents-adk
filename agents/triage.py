from google.adk.agents import LlmAgent
from agents.booking import booking_agent
from agents.support import support_agent
from agents.property import property_agent
import config

triage_agent = LlmAgent(
    name="Triage",
    model=config.GEMINI_MODEL,
    instruction=(
        "You are the virtual assistant for Stayforlong, a long-stay apartment platform in Europe. "
        "Always respond in {lang_name}. If the user writes in a different language, follow their language.\n\n"
        "Your only job is to understand the user's intent and immediately delegate to the correct specialist. "
        "Do NOT answer domain questions yourself. Always transfer:\n\n"
        "• Reservations, booking ID, dates, prices, cancellations → transfer to Booking\n"
        "• Incidents, complaints, maintenance problems, issues during stay → transfer to Support\n"
        "• Accommodation info, amenities, check-in/out times, facilities → transfer to Alojamientos\n\n"
        "If the intent is unclear, ask for clarification in {lang_name}."
    ),
    sub_agents=[booking_agent, support_agent, property_agent],
)
