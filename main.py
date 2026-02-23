import logging
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from ws.handler import websocket_endpoint
import config  # Ensures genai.configure() is called on startup

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
async def root():
    return {
        "service": "Stayforlong Chat PoC",
        "websocket_endpoint": "ws://localhost:8000/ws",
        "health": "/health",
    }
