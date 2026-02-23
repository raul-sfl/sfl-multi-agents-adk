import logging
from pathlib import Path
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from ws.handler import websocket_endpoint
import config  # Ensures genai.configure() is called on startup

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Stayforlong Chat PoC",
    description="Multi-agent AI chat for Stayforlong customer support",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws")
async def ws_route(websocket: WebSocket):
    lang = websocket.query_params.get("lang", "en")[:2].lower()
    await websocket_endpoint(websocket, lang)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": config.GEMINI_MODEL,
        "api_key_configured": bool(config.GEMINI_API_KEY),
    }


@app.get("/")
async def serve_demo():
    return FileResponse(str(FRONTEND_DIR / "demo.html"))

app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
