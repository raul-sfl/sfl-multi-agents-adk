"""
Agent Registry — single source of truth for specialist routing.

To add a new specialist agent:
  1. Create backend/agents/your_agent.py with your_agent = LlmAgent(...)
  2. Import it below and add AgentEntry(your_agent, "What topics it covers")
  3. Done — Triage auto-discovers it, no changes to triage.py needed.
"""
from dataclasses import dataclass
from google.adk.agents import LlmAgent

# ── Specialist agents (order matters: more specific first) ────────────────
from agents.booking import booking_agent
from agents.support import support_agent
from agents.property import property_agent

# ── Fallback agent (HelpCenter — always the last resort) ─────────────────
from agents.knowledge import knowledge_agent


@dataclass
class AgentEntry:
    agent: LlmAgent
    routing_hint: str  # One-line description used in Triage's routing instruction


# ── REGISTRY ─────────────────────────────────────────────────────────────
# Add new specialists HERE. Triage auto-generates its routing from this list.
SPECIALISTS: list[AgentEntry] = [
    AgentEntry(
        booking_agent,
        "Reservations, booking ID, dates, prices, cancellations, refunds",
    ),
    AgentEntry(
        support_agent,
        "Incidents, complaints, maintenance problems, issues during stay",
    ),
    AgentEntry(
        property_agent,
        "Accommodation info, amenities, check-in/out times, facilities",
    ),
    # ← Add new specialists here ↑
]

FALLBACK: LlmAgent = knowledge_agent  # Called when no specialist matches
