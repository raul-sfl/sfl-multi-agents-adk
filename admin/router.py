"""
Admin API router — /admin/api

Exposes:
  GET /admin/api/stats                        → aggregate counts
  GET /admin/api/conversations                → list (filters: status, lang, limit)
  GET /admin/api/conversations/{id}           → conversation metadata
  GET /admin/api/conversations/{id}/messages  → message thread
  GET /admin/api/users/{user_id}/conversations → all convs for a user
  GET /admin/api/agents                       → list all specialist agents
  PATCH /admin/api/agents/{name}              → update agent instruction

Auth: set ADMIN_API_KEY in .env.
  Authorization: Bearer YOUR_KEY  OR  ?key=YOUR_KEY
If ADMIN_API_KEY is empty the API is open (dev only).
Dashboard UI lives in sfl-multi-agents-admin.
"""

from fastapi import APIRouter, Depends, HTTPException, Header, Query
from pydantic import BaseModel
from typing import Optional

router = APIRouter(prefix="/admin", tags=["admin"])


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


# ── API routes ────────────────────────────────────────────────────────────────

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

class PatchAgent(BaseModel):
    instruction: Optional[str] = None


@router.get("/api/agents")
async def api_list_agents(_auth=Depends(require_admin)):
    """Return metadata for all registered specialist agents."""
    from orchestrator.agent_loader import AgentLoader
    loader = AgentLoader()
    plugins = loader.get_plugins()
    return [
        {
            "name": p.name,
            "model": p.model,
            "instruction": p.instruction,
            "routing_hint": getattr(p, "routing_hint", ""),
            "is_fallback": p.is_fallback,
            "tools": [t.__name__ if callable(t) else str(t) for t in (p.get_tools() or [])],
        }
        for p in plugins
    ]


@router.patch("/api/agents/{agent_name}")
async def api_patch_agent(
    agent_name: str,
    body: PatchAgent,
    _auth=Depends(require_admin),
):
    """
    Update an agent's instruction at runtime.

    NOTE: This updates the in-process plugin object so changes are reflected in
    new ADK runner instances. To persist across restarts, update the source file
    in agents/specialists/ and redeploy, or use Vertex AI Agent Engine versioning.
    """
    from orchestrator.agent_loader import AgentLoader
    loader = AgentLoader()
    plugins = loader.get_plugins()
    plugin = next((p for p in plugins if p.name == agent_name), None)
    if not plugin:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found.")
    if body.instruction is not None:
        plugin.instruction = body.instruction
    return {"name": plugin.name, "instruction": plugin.instruction}
