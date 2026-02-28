"""
AgentPlugin — contrato estándar para agentes especialistas auto-descubiertos.

Cada módulo en backend/agents/specialists/ debe exportar una variable PLUGIN
de este tipo. AgentLoader escanea ese directorio y construye LlmAgent objects
dinámicamente a partir de estos metadatos.

Separar esta definición de agent_loader.py evita importaciones circulares:
    specialists/*.py  →  agents/plugin.py
    agent_loader.py   →  agents/plugin.py  +  agents/specialists.*
"""
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class AgentPlugin:
    name: str
    """Nombre del agente. Debe coincidir con LlmAgent.name y es único por app."""

    routing_hint: str
    """Una línea que describe cuándo el Triage debe transferir a este agente.
    Se inserta tal cual en la instrucción del Triage como bullet point."""

    instruction: str
    """System prompt completo del agente. Puede contener {lang_name} —
    ADK lo interpola automáticamente desde el estado de sesión."""

    model: str
    """Nombre del modelo Gemini, p.ej. config.GEMINI_MODEL."""

    is_fallback: bool = False
    """True únicamente para el agente de último recurso (HelpCenter/Knowledge).
    Exactamente 1 plugin en specialists/ debe tener is_fallback=True."""

    get_tools: Callable[[], list] = field(default_factory=lambda: (lambda: []))
    """Callable sin argumentos que devuelve la lista de funciones-herramienta.
    Se llama en build_agents() para asociar las tools locales al LlmAgent."""
