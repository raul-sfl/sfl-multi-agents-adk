"""
AgentPlugin — standard contract for auto-discovered specialist agents.

Each module in agents/specialists/ must export a PLUGIN variable of this type.
AgentLoader scans that directory and dynamically builds LlmAgent objects
from these metadata descriptors.

Keeping this definition separate from agent_loader.py avoids circular imports:
    specialists/*.py  →  agents/plugin.py
    agent_loader.py   →  agents/plugin.py  +  agents/specialists.*
"""
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class AgentPlugin:
    name: str
    """Agent name. Must match LlmAgent.name and be unique within the app."""

    routing_hint: str
    """One-line description of when the Triage should transfer to this agent.
    Inserted verbatim as a bullet point in the Triage agent's instruction."""

    instruction: str
    """Full system prompt for the agent. May contain {lang_name} —
    ADK interpolates it automatically from the session state."""

    model: str
    """Gemini model name, e.g. config.GEMINI_MODEL."""

    is_fallback: bool = False
    """True only for the last-resort fallback agent (HelpCenter/Knowledge).
    Exactly 1 plugin in specialists/ must have is_fallback=True."""

    get_tools: Callable[[], list] = field(default_factory=lambda: (lambda: []))
    """Zero-argument callable that returns the list of tool functions.
    Called in build_agents() to attach local tools to the LlmAgent."""
