"""
tool_registry — mapa de herramientas disponibles para los agentes gestionados via admin.

Permite seleccionar herramientas por nombre cuando se crea/edita un agente en el admin.
transfer_to_triage siempre se incluye automáticamente en todos los agentes.
"""
from typing import Callable
import logging

logger = logging.getLogger(__name__)


def _build_registry() -> dict[str, Callable]:
    """Build the tool registry lazily (avoids circular imports at module level)."""
    from agents.utils import transfer_to_triage
    from agents.specialists.booking import (
        lookup_reservation,
        get_reservations_by_email,
        check_cancellation_policy,
    )
    from agents.specialists.property import (
        lookup_property,
        get_property_amenities,
        get_checkin_info,
    )
    from agents.specialists.support import (
        lookup_incident,
        create_incident,
        escalate_to_human,
    )
    from agents.specialists.knowledge import query_help_center

    return {
        "transfer_to_triage": transfer_to_triage,
        "lookup_reservation": lookup_reservation,
        "get_reservations_by_email": get_reservations_by_email,
        "check_cancellation_policy": check_cancellation_policy,
        "lookup_property": lookup_property,
        "get_property_amenities": get_property_amenities,
        "get_checkin_info": get_checkin_info,
        "lookup_incident": lookup_incident,
        "create_incident": create_incident,
        "escalate_to_human": escalate_to_human,
        "query_help_center": query_help_center,
    }


def get_tools_for(tool_names: list[str]) -> list[Callable]:
    """
    Resolve tool names → callables.
    transfer_to_triage is always appended if not already in the list.
    Unknown names are skipped with a warning.
    """
    registry = _build_registry()
    names = list(tool_names)
    if "transfer_to_triage" not in names:
        names.append("transfer_to_triage")
    tools = []
    for name in names:
        fn = registry.get(name)
        if fn:
            tools.append(fn)
        else:
            logger.warning("Unknown tool '%s' in registry — skipped.", name)
    return tools


def list_available_tools() -> list[dict]:
    """Return metadata for all registered tools (used by GET /admin/api/agents/tools)."""
    registry = _build_registry()
    return [
        {
            "name": name,
            "description": (fn.__doc__ or "").strip().split("\n")[0],
            "always_included": name == "transfer_to_triage",
        }
        for name, fn in registry.items()
    ]
