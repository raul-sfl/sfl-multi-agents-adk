# sfl-multi-agents-adk

FastAPI WebSocket server powering the Stayforlong multi-agent AI pipeline on Vertex AI (Google ADK).

## Architecture

```
Client (WebSocket) → Triage Agent
                        ├── Booking Agent   (reservations, cancellations)
                        ├── Support Agent   (incidents, escalation)
                        ├── Property Agent  (amenities, check-in/out)
                        └── HelpCenter Agent (Vertex AI Search)
```

Sessions are stored in **Vertex AI Agent Engine** (VertexAiSessionService). Conversations are logged to **Cloud Logging**.

## API surface

| Endpoint | Description |
|---|---|
| `WS /ws?lang=es&user_id=XXX` | Main chat WebSocket |
| `GET /health` | Health + config check |
| `GET /admin/api/stats` | Conversation aggregate stats |
| `GET /admin/api/conversations` | List conversations (filters: status, lang, limit) |
| `GET /admin/api/conversations/{id}` | Conversation metadata |
| `GET /admin/api/conversations/{id}/messages` | Message thread |
| `GET /admin/api/users/{user_id}/conversations` | All convs for a user |

Auth for `/admin/api/*`: `Authorization: Bearer YOUR_KEY` or `?key=YOUR_KEY`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env with your values
```

## Run

```bash
uvicorn main:app --port 8000 --reload
```

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `GEMINI_MODEL` | No | Model name (default: `gemini-2.5-flash`) |
| `GOOGLE_GENAI_USE_VERTEXAI` | Yes (prod) | Set `true` to use Vertex AI instead of AI Studio |
| `GOOGLE_CLOUD_PROJECT` | Yes (prod) | GCP project ID |
| `GOOGLE_CLOUD_LOCATION` | No | Region (default: `us-central1`) |
| `TRIAGE_ENGINE_ID` | No | Vertex AI Reasoning Engine ID for session storage |
| `VERTEX_STAGING_BUCKET` | No | GCS bucket for agent provisioning |
| `VERTEX_AI_SEARCH_ENGINE_ID` | No | Vertex AI Search engine ID for HelpCenter |
| `CLOUD_LOGGING_ENABLED` | No | `true`/`false` (default: `true`) |
| `ADMIN_API_KEY` | No | Secret key for `/admin/api/*` (open if empty) |
| `ADMIN_ORIGIN` | No | CORS origin for sfl-multi-agents-admin |
| `FRONTEND_ORIGIN` | No | CORS origin for sfl-multi-agents-chat |
| `GEMINI_API_KEY` | No | Only for AI Studio mode (no Vertex) |

## Provision agents on Vertex AI

```bash
# Deploy agent tree for the first time
python -m orchestrator.provision

# Force redeploy (updates instructions/tools)
python -m orchestrator.provision --force

# List current state
python -m orchestrator.provision --list
```

## Related repos

- **sfl-multi-agents-chat** — GTM-injectable chat widget
- **sfl-multi-agents-admin** — Conversations & agent management dashboard
