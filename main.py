import logging
from pathlib import Path
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from ws.handler import websocket_endpoint
from admin.router import router as admin_router
import config  # Must be imported first to set env vars before ADK initializes

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

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
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(admin_router)


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
        "api_key_configured": bool(config.GEMINI_API_KEY),
        "cloud_logging_enabled": config.CLOUD_LOGGING_ENABLED,
        "agent_engine_id": config.AGENT_ENGINE_ID or None,
    }


@app.get("/")
async def serve_demo():
    return FileResponse(str(FRONTEND_DIR / "demo.html"))

app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
