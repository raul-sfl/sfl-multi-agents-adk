import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

# Vertex AI — set GOOGLE_GENAI_USE_VERTEXAI=true to use Vertex instead of AI Studio
# Also requires: GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION
# Local auth: gcloud auth application-default login
# Railway: set GOOGLE_APPLICATION_CREDENTIALS to service account JSON path
_use_vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "false").lower() == "true"
if _use_vertex:
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "1"
    os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "us-central1")
    # When using Vertex AI, do NOT set GOOGLE_API_KEY — having both project and
    # api_key causes BaseApiClient.__init__ to fail before _http_options is set,
    # producing AttributeError in aclose() cleanup tasks.
    os.environ.pop("GOOGLE_API_KEY", None)
else:
    # AI Studio mode: map GEMINI_API_KEY → GOOGLE_API_KEY which ADK reads
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
    if GEMINI_API_KEY and not os.environ.get("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

# Exposed for health endpoint and diagnostics
USE_VERTEX_AI = _use_vertex
GOOGLE_CLOUD_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
GOOGLE_CLOUD_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

# Location for the Agent Engine (session storage). Defaults to GOOGLE_CLOUD_LOCATION.
# Set AGENT_ENGINE_LOCATION to keep sessions in a different region than model calls
# (e.g. europe-west1 for data residency while model runs in us-central1).
AGENT_ENGINE_LOCATION = os.environ.get("AGENT_ENGINE_LOCATION", GOOGLE_CLOUD_LOCATION)

# ── Vertex AI Search (Discovery Engine) ─────────────────────────────────────
# Search engine ID for the Stayforlong Help Center Vertex AI Search index.
# Format: just the engine ID (e.g. "stayforlong-help-center_1772048036019")
# Get it from: https://console.cloud.google.com/gen-app-builder/engines
VERTEX_AI_SEARCH_ENGINE_ID = os.environ.get("VERTEX_AI_SEARCH_ENGINE_ID", "")

# ── Vertex AI Agent Engine (VertexAiSessionService) ──────────────────────────
# Numeric resource ID of the reasoning engine, e.g. "1234567890123456"
# Get it from: gcloud ai reasoning-engines list --location=LOCATION
# If empty, falls back to InMemorySessionService (local dev without GCP)
AGENT_ENGINE_ID = os.environ.get("AGENT_ENGINE_ID", "")

# ── Conversation logging (Cloud Logging) ─────────────────────────────────────
# Set CLOUD_LOGGING_ENABLED=false to disable (logs go nowhere).
# Uses GOOGLE_CLOUD_PROJECT for the log destination.
# Local dev: gcloud auth application-default login
CLOUD_LOGGING_ENABLED = os.environ.get("CLOUD_LOGGING_ENABLED", "true").lower() == "true"

# Log name written to Cloud Logging (under the GCP project)
CLOUD_LOGGING_LOG_NAME = os.environ.get("CLOUD_LOGGING_LOG_NAME", "stayforlong-conversations")

# How many hours back to look for a previous session to offer history recovery
HISTORY_RECOVERY_HOURS = int(os.environ.get("HISTORY_RECOVERY_HOURS", "48"))

# ── Admin dashboard ───────────────────────────────────────────────────────────
# Protect the /admin dashboard with a secret key.
# If empty, the dashboard is open (dev mode only — set a key in production!).
ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY", "")
