"""
Triage agent — main router for the Stayforlong assistant.

build_triage_agent() is a factory that receives specialist agents
dynamically discovered by AgentLoader and builds the Triage LlmAgent
with its sub_agents and the correct routing instruction.

Called from adk_runner.py:
    loader = AgentLoader()
    specialists, fallback = loader.build_agents()
    triage_agent = build_triage_agent(specialists, fallback)
"""
from google.adk.agents import LlmAgent
from agents.plugin import AgentPlugin
import config


def build_triage_agent(
    specialists: list[tuple[AgentPlugin, LlmAgent]],
    fallback: tuple[AgentPlugin, LlmAgent],
) -> LlmAgent:
    """
    Build the Triage agent with dynamically discovered sub_agents.

    Args:
        specialists: List of (AgentPlugin, LlmAgent) for non-fallback agents.
                     Order determines routing priority in the instruction.
        fallback:    (AgentPlugin, LlmAgent) for the last-resort fallback agent.

    Returns:
        LlmAgent configured with sub_agents and the full routing instruction.
    """
    fallback_plugin, fallback_agent = fallback

    routing_bullets = "\n".join(
        f"• {plugin.routing_hint} → transfer to {agent.name}"
        for plugin, agent in specialists
    )

    instruction = _build_instruction(routing_bullets, fallback_agent.name)

    return LlmAgent(
        name="Triage",
        model=config.GEMINI_MODEL,
        instruction=instruction,
        sub_agents=[agent for _, agent in specialists] + [fallback_agent],
    )


def _build_instruction(routing_bullets: str, fallback_name: str) -> str:
    return (
        "You are the virtual assistant for Stayforlong, a long-stay apartment platform in Europe. "
        "Always respond in {lang_name}. If the user writes in a different language, follow their language.\n\n"
        "Your only job is to understand the user's intent and immediately delegate to the correct specialist. "
        "Do NOT answer domain questions yourself. Always transfer:\n\n"
        f"{routing_bullets}\n"
        f"• Any other question → transfer to {fallback_name} (general help center)\n\n"
        "CRITICAL ROUTING RULES:\n"
        "• Transfer to Booking when the user asks about THEIR OWN specific reservation "
        "(status, dates, price, cancellation deadline, payment, etc.) — even without a booking ID. "
        "Booking will ask for the ID or email if needed.\n"
        "• Transfer to HelpCenter ONLY for GENERAL platform questions (how policies work, "
        "platform FAQs, payment methods, minimum stay rules) — NOT for personal reservation lookups.\n"
        "• Examples that go to Booking:\n"
        "  - '¿Cuál es el estado de mi reserva?' → Booking\n"
        "  - '¿Cuánto pagué por mi reserva?' → Booking\n"
        "  - 'Quiero ver mi reserva SFL-2024-001' → Booking\n"
        "  - 'Mi email es juan@gmail.com, ¿cuál es mi reserva?' → Booking\n"
        "• Examples that go to HelpCenter (general, not personal):\n"
        "  - '¿Cuáles son las políticas de cancelación de Stayforlong?' → HelpCenter\n"
        "  - '¿Cómo funciona el pago en la plataforma?' → HelpCenter\n"
        "  - '¿Puedo ampliar una estancia en general?' → HelpCenter\n\n"
        "If you truly cannot determine the topic, ask for clarification in {lang_name}."
    )
