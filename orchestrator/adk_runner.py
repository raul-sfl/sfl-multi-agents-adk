import config  # Must be imported first to set env vars before ADK initializes
from google.adk.runners import Runner
from agents.triage import triage_agent

# ── Session service ───────────────────────────────────────────────────────────
# Use VertexAiSessionService when AGENT_ENGINE_ID is configured (GCP / production).
# Falls back to InMemorySessionService for local development without GCP credentials.

if config.AGENT_ENGINE_ID and config.GOOGLE_CLOUD_PROJECT:
    import logging as _logging
    from google.adk.sessions import VertexAiSessionService

    session_service = VertexAiSessionService(
        project=config.GOOGLE_CLOUD_PROJECT,
        location=config.GOOGLE_CLOUD_LOCATION,
        agent_engine_id=config.AGENT_ENGINE_ID,
    )
    _logging.getLogger(__name__).info(
        "Using VertexAiSessionService (project=%s, engine=%s)",
        config.GOOGLE_CLOUD_PROJECT,
        config.AGENT_ENGINE_ID,
    )
else:
    import logging as _logging
    from google.adk.sessions import InMemorySessionService

    session_service = InMemorySessionService()
    _logging.getLogger(__name__).warning(
        "AGENT_ENGINE_ID not set — using InMemorySessionService (sessions lost on restart)."
    )

# ── Runner ────────────────────────────────────────────────────────────────────
# app_name is scoped per Agent Engine resource via agent_engine_id in the
# session service constructor, so any string works here.
runner = Runner(
    agent=triage_agent,
    app_name="stayforlong",
    session_service=session_service,
)
