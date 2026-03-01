import logging as _logging
import threading
import config  # Must be imported first to set env vars before ADK initializes
from google.adk.runners import Runner
from agent import root_agent  # Standard ADK entrypoint — Triage + all sub_agents

_logger = _logging.getLogger(__name__)
_lock = threading.Lock()

# ── Session service ───────────────────────────────────────────────────────────
# Use VertexAiSessionService when TRIAGE_ENGINE_ID is configured (GCP / production).
# Falls back to InMemorySessionService for local development without GCP credentials.
# TRIAGE_ENGINE_ID also accepts AGENT_ENGINE_ID as a backward-compat alias.

if config.TRIAGE_ENGINE_ID and config.GOOGLE_CLOUD_PROJECT:
    from google.adk.sessions import VertexAiSessionService

    session_service = VertexAiSessionService(
        project=config.GOOGLE_CLOUD_PROJECT,
        location=config.AGENT_ENGINE_LOCATION,
        agent_engine_id=config.TRIAGE_ENGINE_ID,
    )
    _logger.info(
        "Using VertexAiSessionService (project=%s, engine=%s)",
        config.GOOGLE_CLOUD_PROJECT,
        config.TRIAGE_ENGINE_ID,
    )
else:
    from google.adk.sessions import InMemorySessionService

    session_service = InMemorySessionService()
    _logger.warning(
        "TRIAGE_ENGINE_ID not set — using InMemorySessionService (sessions lost on restart)."
    )

# ── Runner ────────────────────────────────────────────────────────────────────
# _runner is a module-level variable that can be replaced by rebuild_runner().
# Use get_runner() for late binding — never import `runner` directly.

_runner: Runner = Runner(
    agent=root_agent,
    app_name="stayforlong",
    session_service=session_service,
)


def get_runner() -> Runner:
    """Late-binding accessor. Always returns the current runner instance."""
    return _runner


def rebuild_runner() -> None:
    """
    Rebuild the runner with the current merged agent tree (Python source + GCS store).

    Called automatically after any admin create/update/delete operation.
    New WebSocket sessions created after this call will use the updated agents.
    Existing in-flight sessions are not affected.
    """
    global _runner
    from orchestrator.agent_loader import AgentLoader
    from agents.triage import build_triage_agent

    loader = AgentLoader()
    specialists, fallback = loader.build_agents_merged()
    new_root = build_triage_agent(specialists, fallback)

    new_runner = Runner(
        agent=new_root,
        app_name="stayforlong",
        session_service=session_service,
    )
    with _lock:
        _runner = new_runner

    _logger.info("Runner rebuilt with updated agent tree.")
