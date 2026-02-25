import os
from dotenv import load_dotenv

load_dotenv()

# AI Studio API key — mapped to GOOGLE_API_KEY which ADK reads automatically
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY and not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

# Vertex AI (optional) — set GOOGLE_GENAI_USE_VERTEXAI=true to use Vertex instead of AI Studio
# Also requires: GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION
# Local auth: gcloud auth application-default login
# Railway: set GOOGLE_APPLICATION_CREDENTIALS to service account JSON path
_use_vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "false").lower() == "true"
if _use_vertex:
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "1"
    os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "us-central1")

# Exposed for health endpoint and diagnostics
USE_VERTEX_AI = _use_vertex
GOOGLE_CLOUD_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
GOOGLE_CLOUD_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

# Vertex AI Agent Designer — HelpCenter conversational agent (Dialogflow CX Playbooks)
# Format: projects/PROJECT/locations/LOCATION/agents/AGENT_ID
# Get AGENT_ID from the URL in Agent Designer:
#   https://console.cloud.google.com/gen-app-builder/locations/LOCATION/agents/AGENT_ID
HELP_CENTER_AGENT_ID = os.environ.get("HELP_CENTER_AGENT_ID", "")
