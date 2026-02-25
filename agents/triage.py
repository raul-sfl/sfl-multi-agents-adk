from google.adk.agents import LlmAgent
from agents.registry import SPECIALISTS, FALLBACK
import config


def _build_instruction() -> str:
    routing_bullets = "\n".join(
        f"• {entry.routing_hint} → transfer to {entry.agent.name}"
        for entry in SPECIALISTS
    )
    return (
        "You are the virtual assistant for Stayforlong, a long-stay apartment platform in Europe. "
        "Always respond in {lang_name}. If the user writes in a different language, follow their language.\n\n"
        "Your only job is to understand the user's intent and immediately delegate to the correct specialist. "
        "Do NOT answer domain questions yourself. Always transfer:\n\n"
        f"{routing_bullets}\n"
        f"• Any other question → transfer to {FALLBACK.name} (general help center)\n\n"
        "If you truly cannot determine the topic, ask for clarification in {lang_name}."
    )


triage_agent = LlmAgent(
    name="Triage",
    model=config.GEMINI_MODEL,
    instruction=_build_instruction(),
    sub_agents=[entry.agent for entry in SPECIALISTS] + [FALLBACK],
)
