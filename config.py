import os
import json
import tempfile

# In Vertex AI Agent Engine runtime, _vertex_env.py is bundled via extra_packages
# and pre-sets os.environ with project/location/model values. Safe no-op locally.
try:
    import _vertex_env  # noqa: F401
except ImportError:
    pass

from dotenv import load_dotenv

load_dotenv()

# ── Railway / Cloud Run: GCP service account from env var ─────────────────────
# Set GOOGLE_APPLICATION_CREDENTIALS_JSON to the full contents of your
# service account JSON key (copy-paste the entire JSON as a single env var).
# This writes it to a temp file and sets GOOGLE_APPLICATION_CREDENTIALS so
# Google client libraries pick it up automatically.
_sa_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON", "").strip()
if _sa_json and not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
    try:
        _sa_data = json.loads(_sa_json)
        _sa_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, prefix="gcp_sa_"
        )
        json.dump(_sa_data, _sa_file)
        _sa_file.close()
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = _sa_file.name
    except Exception as _e:
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "Failed to write GOOGLE_APPLICATION_CREDENTIALS_JSON to temp file: %s", _e
        )

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

# Vertex AI — set GOOGLE_GENAI_USE_VERTEXAI=true to use Vertex instead of AI Studio
# Also requires: GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION
# Local auth: gcloud auth application-default login
# Railway: set GOOGLE_APPLICATION_CREDENTIALS_JSON to the service account JSON content
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
# Numeric resource ID of the Triage reasoning engine (session storage).
# TRIAGE_ENGINE_ID is the canonical name; AGENT_ENGINE_ID is supported as alias.
# Get it from: gcloud ai reasoning-engines list --location=LOCATION
# If empty, falls back to InMemorySessionService (local dev without GCP)
TRIAGE_ENGINE_ID = (
    os.environ.get("TRIAGE_ENGINE_ID")
    or os.environ.get("AGENT_ENGINE_ID", "")
)
AGENT_ENGINE_ID = TRIAGE_ENGINE_ID  # backward-compat alias

# ── Vertex AI Agent Provisioning ─────────────────────────────────────────────
# GCS bucket for staging agent artifacts when deploying to Vertex AI.
# Format: gs://your-bucket-name  (only required to run provision.py)
VERTEX_STAGING_BUCKET = os.environ.get("VERTEX_STAGING_BUCKET", "")

# Python packages bundled when deploying agent plugins to Vertex AI Reasoning Engine.
VERTEX_AGENT_REQUIREMENTS = [
    "google-adk",
    "google-cloud-discoveryengine",
    "google-cloud-logging>=3.0.0",
    "python-dotenv>=1.0.0",
    "pydantic>=2.0.0",
]

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

# ── CORS ──────────────────────────────────────────────────────────────────────
# Comma-separated list of allowed origins.
# ADMIN_ORIGIN: URL where sfl-multi-agents-admin is hosted.
# FRONTEND_ORIGIN: URL where sfl-multi-agents-chat widget/demo is hosted.
# Defaults to ["*"] when neither is set (local dev).
_admin_origin = os.environ.get("ADMIN_ORIGIN", "")
_frontend_origin = os.environ.get("FRONTEND_ORIGIN", "")
_explicit_origins = [o.strip() for o in [_admin_origin, _frontend_origin] if o.strip()]
CORS_ORIGINS: list[str] = _explicit_origins if _explicit_origins else ["*"]
