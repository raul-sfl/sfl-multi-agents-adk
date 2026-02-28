"""
Entrypoint estándar ADK — exporta root_agent para adk web y Vertex AI.

Este archivo sigue la convención ADK: cualquier herramienta del ecosistema ADK
(adk web, adk deploy, provision.py) busca un módulo con una variable root_agent.

Uso local (desarrollo):
    cd backend
    adk web          # → abre UI en http://localhost:8000 con el árbol de agentes

Uso en provision (deploy a Vertex AI Agent Engine):
    from agent import root_agent
    AdkApp(agent=root_agent) → agent_engines.create(...)

El FastAPI backend importa root_agent desde aquí via adk_runner.py.
"""
import config  # Must be first — sets env vars before ADK initializes
from orchestrator.agent_loader import AgentLoader
from agents.triage import build_triage_agent

_loader = AgentLoader()
_specialists, _fallback = _loader.build_agents()

# root_agent: árbol completo Triage → [Booking, Support, Alojamientos, HelpCenter]
# ADK usa esta variable para descubrir el sistema multi-agente.
root_agent = build_triage_agent(_specialists, _fallback)
