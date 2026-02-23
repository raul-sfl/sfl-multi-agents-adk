from orchestrator.types import Agent


def transfer_to_booking():
    """Transfer to the Booking Agent for questions about reservations, booking status, dates, prices, or cancellation policies."""
    from agents.booking import booking_agent
    return booking_agent


def transfer_to_support():
    """Transfer to the Support Agent for incidents, complaints, maintenance issues, or problems during the stay."""
    from agents.support import support_agent
    return support_agent


def transfer_to_property():
    """Transfer to the Alojamientos agent for questions about accommodation amenities, check-in/check-out times, facilities, or property-specific information."""
    from agents.property import property_agent
    return property_agent


def _triage_instructions(context: dict) -> str:
    lang_name = context.get("lang_name", "English")
    return (
        f"You are the virtual assistant for Stayforlong, a long-stay apartment platform in Europe. "
        f"Always respond in {lang_name}. If the user writes in a different language, follow their language.\n\n"
        "Your only job is to understand the user's intent and immediately call the correct transfer function. "
        "Do NOT answer domain questions yourself. Always transfer:\n\n"
        "• Reservations, booking ID, dates, prices, cancellations → transfer_to_booking\n"
        "• Incidents, complaints, maintenance problems, issues during stay → transfer_to_support\n"
        "• Accommodation info, amenities, check-in/out times, facilities → transfer_to_property\n\n"
        f"If the intent is unclear, ask for clarification in {lang_name}."
    )


triage_agent = Agent(
    name="Triage Agent",
    instructions=_triage_instructions,
    tools=[transfer_to_booking, transfer_to_support, transfer_to_property],
)
