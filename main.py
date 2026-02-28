import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from ws.handler import websocket_endpoint
from admin.router import router as admin_router
import config  # Must be imported first to set env vars before ADK initializes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
# Silence verbose INFO from Google AI/ADK internals — keep WARNING+ only
logging.getLogger("google_genai").setLevel(logging.WARNING)
logging.getLogger("google_adk").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Stayforlong Chat PoC",
    description="Multi-agent AI chat for Stayforlong customer support",
    version="0.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(admin_router)

# Thread pool for running blocking provision tasks without blocking the event loop
_provision_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="provision")


@app.on_event("startup")
async def startup_provision():
    """
    Provisions agents on Vertex AI in the background on server startup.

    Fast-path: if all agents are already registered the function completes
    in ~2s without any deployment. Does not block request handling.
    Silently skipped if GOOGLE_CLOUD_PROJECT or VERTEX_STAGING_BUCKET
    are not configured (local dev without GCP).
    """
    if not config.GOOGLE_CLOUD_PROJECT or not config.VERTEX_STAGING_BUCKET:
        return

    loop = asyncio.get_event_loop()

    def _provision():
        from orchestrator.provision import run_provision
        run_provision(force=False)

    loop.run_in_executor(_provision_executor, _provision)


@app.websocket("/ws")
async def ws_route(websocket: WebSocket):
    lang    = websocket.query_params.get("lang", "en")[:2].lower()
    user_id = websocket.query_params.get("user_id", "").strip()
    await websocket_endpoint(websocket, lang, user_id)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": config.GEMINI_MODEL,
        "backend": "vertex_ai" if config.USE_VERTEX_AI else "ai_studio",
        "project": config.GOOGLE_CLOUD_PROJECT if config.USE_VERTEX_AI else None,
        "api_key_configured": bool(getattr(config, "GEMINI_API_KEY", "")),
        "cloud_logging_enabled": config.CLOUD_LOGGING_ENABLED,
        "triage_engine_id": config.TRIAGE_ENGINE_ID or None,
    }


@app.get("/")
async def root():
    return {"status": "ok", "service": "sfl-multi-agents-adk"}
