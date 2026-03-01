"""
Admin API router — /admin/api

Exposes:
  GET    /admin/api/stats                         → aggregate counts
  GET    /admin/api/conversations                 → list (filters: status, lang, limit)
  GET    /admin/api/conversations/{id}            → conversation metadata
  GET    /admin/api/conversations/{id}/messages   → message thread
  GET    /admin/api/users/{user_id}/conversations → all convs for a user

  GET    /admin/api/agents                        → list all agents (Python + GCS merged)
  POST   /admin/api/agents                        → create new GCS-managed agent
  PATCH  /admin/api/agents/{name}                 → update agent (persisted to GCS/local)
  DELETE /admin/api/agents/{name}                 → delete GCS-managed agent
  GET    /admin/api/agents/tools                  → list available tools for selection

Auth: set ADMIN_API_KEY in .env.
  Authorization: Bearer YOUR_KEY  OR  ?key=YOUR_KEY
If ADMIN_API_KEY is empty the API is open (dev only).
Dashboard UI lives in sfl-multi-agents-admin.
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, Header, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

router = APIRouter(prefix="/admin", tags=["admin"])
logger = logging.getLogger(__name__)


# ── Auth dependency ───────────────────────────────────────────────────────────

async def require_admin(
    authorization: Optional[str] = Header(None),
    key: Optional[str] = Query(None),
) -> None:
    import config
    if not config.ADMIN_API_KEY:
        return  # open in dev mode
    provided: Optional[str] = None
    if authorization and authorization.startswith("Bearer "):
        provided = authorization[7:].strip()
    if not provided and key:
        provided = key
    if provided != config.ADMIN_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key. Pass ?key=YOUR_KEY or Authorization: Bearer header.",
        )


# ── API routes — conversations & stats ───────────────────────────────────────

@router.get("/api/stats")
async def api_stats(_auth=Depends(require_admin)):
    from services.conversation_logger import conversation_logger
    return await conversation_logger.get_stats()


@router.get("/api/conversations")
async def api_list_conversations(
    status: Optional[str] = None,
    lang: Optional[str] = None,
    limit: int = Query(50, ge=1, le=200),
    _auth=Depends(require_admin),
):
    from services.conversation_logger import conversation_logger
    result = await conversation_logger.list_conversations(limit=limit)
    convs = result.get("items", []) if isinstance(result, dict) else result
    if status:
        convs = [c for c in convs if c.get("status") == status]
    if lang:
        convs = [c for c in convs if c.get("language") == lang]
    return convs


@router.get("/api/conversations/{conv_id}")
async def api_get_conversation(conv_id: str, _auth=Depends(require_admin)):
    from services.conversation_logger import conversation_logger
    conv = await conversation_logger.get_conversation(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return conv


@router.get("/api/conversations/{conv_id}/messages")
async def api_get_messages(
    conv_id: str,
    limit: int = Query(100, ge=1, le=500),
    _auth=Depends(require_admin),
):
    from services.conversation_logger import conversation_logger
    return await conversation_logger.get_conversation_messages(conv_id, limit=limit)


@router.get("/api/users/{user_id}/conversations")
async def api_user_conversations(
    user_id: str,
    status: Optional[str] = None,
    lang: Optional[str] = None,
    _auth=Depends(require_admin),
):
    from services.conversation_logger import conversation_logger
    result = await conversation_logger.list_conversations(user_id=user_id)
    convs = result.get("items", []) if isinstance(result, dict) else result
    if status:
        convs = [c for c in convs if c.get("status") == status]
    if lang:
        convs = [c for c in convs if c.get("language") == lang]
    return convs


# ── Agent management ──────────────────────────────────────────────────────────

def _agent_tools_list(plugin) -> list[str]:
    """Get tool names from a Python plugin's get_tools() result."""
    try:
        return [t.__name__ if callable(t) else str(t) for t in (plugin.get_tools() or [])]
    except Exception:
        return []


# ── GET /api/agents/tools  (must be registered BEFORE /api/agents/{name}) ────

@router.get("/api/agents/tools")
async def api_list_tools(_auth=Depends(require_admin)):
    """Return all registered tools available for agent configuration."""
    from agents.tool_registry import list_available_tools
    return list_available_tools()


# ── GET /api/agents ───────────────────────────────────────────────────────────

@router.get("/api/agents")
async def api_list_agents(_auth=Depends(require_admin)):
    """Return all agents — Python source (with any GCS overrides) plus GCS-only agents."""
    from orchestrator.agent_loader import AgentLoader
    from services.agent_gcs_store import load_all

    loader = AgentLoader()
    plugins = loader.get_plugins()
    gcs_configs = load_all()
    python_names = {p.name for p in plugins}

    result = []

    # Python source agents (override from GCS if present)
    for p in plugins:
        override = gcs_configs.get(p.name, {})
        result.append({
            "name": p.name,
            "model": override.get("model", p.model),
            "instruction": override.get("instruction", p.instruction),
            "routing_hint": override.get("routing_hint", getattr(p, "routing_hint", "")),
            "is_fallback": override.get("is_fallback", p.is_fallback),
            "tools": override.get("tools", _agent_tools_list(p)),
            "source": "gcs" if p.name in gcs_configs else "python",
            "has_python_source": True,
        })

    # GCS-only agents (not present in Python source files)
    for name, cfg in gcs_configs.items():
        if name not in python_names:
            result.append({**cfg, "source": "gcs", "has_python_source": False})

    return result


# ── POST /api/agents — create GCS-managed agent ───────────────────────────────

class CreateAgent(BaseModel):
    name: str
    routing_hint: str
    instruction: str
    model: str = "gemini-2.5-flash"
    is_fallback: bool = False
    tools: list[str] = ["transfer_to_triage"]


@router.post("/api/agents", status_code=201)
async def api_create_agent(body: CreateAgent, _auth=Depends(require_admin)):
    """Create a new GCS-managed agent. Fails if the name already exists."""
    from orchestrator.agent_loader import AgentLoader
    from services.agent_gcs_store import load_all, save_agent

    # Check for name conflicts across Python source and GCS
    loader = AgentLoader()
    python_names = {p.name for p in loader.get_plugins()}
    gcs_names = set(load_all().keys())
    if body.name in python_names | gcs_names:
        raise HTTPException(status_code=409, detail=f"Agent '{body.name}' already exists.")

    agent_dict = {**body.model_dump(), "source": "gcs"}
    try:
        save_agent(agent_dict)
    except OSError as exc:
        logger.error("Failed to save new agent '%s': %s", body.name, exc)
        raise HTTPException(status_code=500, detail=str(exc))

    _try_rebuild_runner(body.name, "created")
    return agent_dict


# ── PATCH /api/agents/{name} — update instruction/fields ─────────────────────

class PatchAgent(BaseModel):
    instruction: Optional[str] = None
    routing_hint: Optional[str] = None
    model: Optional[str] = None
    tools: Optional[list[str]] = None


@router.patch("/api/agents/{agent_name}")
async def api_patch_agent(
    agent_name: str,
    body: PatchAgent,
    _auth=Depends(require_admin),
):
    """
    Update an agent's fields and persist to GCS/local store.
    Works for both Python source agents (override) and GCS-managed agents.
    """
    from orchestrator.agent_loader import AgentLoader
    from services.agent_gcs_store import load_all, save_agent

    loader = AgentLoader()
    plugins = loader.get_plugins()
    plugin = next((p for p in plugins if p.name == agent_name), None)
    gcs_configs = load_all()
    current_gcs = gcs_configs.get(agent_name, {})

    if plugin is None and agent_name not in gcs_configs:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found.")

    # Build the merged record: start from current GCS (or derive from Python plugin)
    updated = {
        "name": agent_name,
        "routing_hint": current_gcs.get(
            "routing_hint", getattr(plugin, "routing_hint", "") if plugin else ""
        ),
        "instruction": current_gcs.get(
            "instruction", plugin.instruction if plugin else ""
        ),
        "model": current_gcs.get(
            "model", plugin.model if plugin else "gemini-2.5-flash"
        ),
        "is_fallback": current_gcs.get(
            "is_fallback", plugin.is_fallback if plugin else False
        ),
        "tools": current_gcs.get(
            "tools", _agent_tools_list(plugin) if plugin else ["transfer_to_triage"]
        ),
        "source": "gcs",
    }

    if body.instruction is not None:
        updated["instruction"] = body.instruction
    if body.routing_hint is not None:
        updated["routing_hint"] = body.routing_hint
    if body.model is not None:
        updated["model"] = body.model
    if body.tools is not None:
        updated["tools"] = body.tools

    try:
        save_agent(updated)
    except OSError as exc:
        logger.error("Failed to save agent '%s': %s", agent_name, exc)
        raise HTTPException(status_code=500, detail=str(exc))

    _try_rebuild_runner(agent_name, "updated")
    return updated


# ── DELETE /api/agents/{name} — remove GCS-managed agent ─────────────────────

@router.delete("/api/agents/{agent_name}")
async def api_delete_agent(agent_name: str, _auth=Depends(require_admin)):
    """
    Delete a GCS-managed agent. Python source agents cannot be deleted from the admin
    (remove the .py file from agents/specialists/ instead).
    """
    from orchestrator.agent_loader import AgentLoader
    from services.agent_gcs_store import delete_agent

    loader = AgentLoader()
    if any(p.name == agent_name for p in loader.get_plugins()):
        raise HTTPException(
            status_code=400,
            detail=(
                f"'{agent_name}' is defined in Python source code and cannot be deleted "
                f"from the admin. Remove agents/specialists/{agent_name.lower()}.py and redeploy."
            ),
        )

    deleted = delete_agent(agent_name)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found.")

    _try_rebuild_runner(agent_name, "deleted")
    return {"deleted": agent_name}


# ── DELETE /api/agents/{name}/override — reset Python source agent to defaults ─

@router.delete("/api/agents/{agent_name}/override")
async def api_reset_agent(agent_name: str, _auth=Depends(require_admin)):
    """
    Remove the GCS override for a Python source agent, reverting it to its
    Python source defaults. Has no effect if no override exists.
    """
    from orchestrator.agent_loader import AgentLoader
    from services.agent_gcs_store import delete_agent

    loader = AgentLoader()
    if not any(p.name == agent_name for p in loader.get_plugins()):
        raise HTTPException(
            status_code=400,
            detail=f"'{agent_name}' is not a Python source agent. Use DELETE to remove GCS agents.",
        )

    deleted = delete_agent(agent_name)
    _try_rebuild_runner(agent_name, "reset to Python defaults")
    return {"reset": agent_name, "had_override": deleted}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _try_rebuild_runner(agent_name: str, action: str) -> None:
    """Rebuild the ADK runner after an agent change. Logs but never raises."""
    try:
        from orchestrator.adk_runner import rebuild_runner
        rebuild_runner()
        logger.info("Runner rebuilt after agent '%s' was %s.", agent_name, action)
    except Exception as exc:
        logger.warning(
            "Agent '%s' %s in store, but runner rebuild failed: %s",
            agent_name, action, exc,
        )
