import config  # Must be imported first to set env vars before ADK initializes
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from agents.triage import triage_agent

session_service = InMemorySessionService()

runner = Runner(
    agent=triage_agent,
    app_name="stayforlong",
    session_service=session_service,
)
