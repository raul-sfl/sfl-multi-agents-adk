"""
Standard ADK entrypoint — exports root_agent for adk web and Vertex AI.

This file follows the ADK convention: any ADK ecosystem tool
(adk web, adk deploy, provision.py) looks for a module with a root_agent variable.

Local usage (development):
    adk web          # → opens UI at http://localhost:8000 with the agent tree

Usage in provision (deploy to Vertex AI Agent Engine):
    from agent import root_agent
    AdkApp(agent=root_agent) → agent_engines.create(...)

The FastAPI backend imports root_agent from here via adk_runner.py.
"""
import config  # Must be first — sets env vars before ADK initializes
from orchestrator.agent_loader import AgentLoader
from agents.triage import build_triage_agent

_loader = AgentLoader()
_specialists, _fallback = _loader.build_agents_merged()

# root_agent: full tree Triage → [Booking, Support, Property, HelpCenter] + any GCS agents
# ADK uses this variable to discover the multi-agent system.
root_agent = build_triage_agent(_specialists, _fallback)
