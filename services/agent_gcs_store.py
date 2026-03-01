"""
agent_gcs_store — CRUD para configuración de agentes en GCS.

GCS path: agent_configs/agents.json dentro de VERTEX_STAGING_BUCKET.
Fallback local: agents/agent_configs.json (dev sin GCP o sin bucket configurado).

Schema:
{
  "Booking": {
    "name": "Booking",
    "routing_hint": "...",
    "instruction": "...",
    "model": "gemini-2.5-flash",
    "is_fallback": false,
    "tools": ["lookup_reservation", "transfer_to_triage"],
    "source": "gcs",
    "updated_at": "2026-03-01T12:00:00Z"
  }
}
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_GCS_OBJECT = "agent_configs/agents.json"
_LOCAL_FALLBACK = Path(__file__).parent.parent / "agents" / "agent_configs.json"


def _bucket_name() -> Optional[str]:
    """Strip gs:// prefix from VERTEX_STAGING_BUCKET. Returns None if not configured."""
    import config
    bucket = getattr(config, "VERTEX_STAGING_BUCKET", "") or ""
    if bucket.startswith("gs://"):
        return bucket[5:].rstrip("/")
    return bucket.rstrip("/") or None


def _use_gcs() -> bool:
    import config
    return bool(getattr(config, "GOOGLE_CLOUD_PROJECT", None) and _bucket_name())


# ── GCS operations ─────────────────────────────────────────────────────────────

def _gcs_load() -> dict:
    import config
    from google.cloud import storage
    client = storage.Client(project=config.GOOGLE_CLOUD_PROJECT)
    bucket = client.bucket(_bucket_name())
    blob = bucket.blob(_GCS_OBJECT)
    if not blob.exists():
        return {}
    try:
        return json.loads(blob.download_as_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to load GCS agent configs: %s", exc)
        return {}


def _gcs_save(data: dict) -> None:
    import config
    from google.cloud import storage
    client = storage.Client(project=config.GOOGLE_CLOUD_PROJECT)
    bucket = client.bucket(_bucket_name())
    blob = bucket.blob(_GCS_OBJECT)
    blob.upload_from_string(
        json.dumps(data, ensure_ascii=False, indent=2),
        content_type="application/json",
    )


# ── Local fallback operations ───────────────────────────────────────────────────

def _local_load() -> dict:
    if not _LOCAL_FALLBACK.exists():
        return {}
    try:
        return json.loads(_LOCAL_FALLBACK.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to load local agent configs: %s", exc)
        return {}


def _local_save(data: dict) -> None:
    tmp = _LOCAL_FALLBACK.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.rename(_LOCAL_FALLBACK)


# ── Public API ──────────────────────────────────────────────────────────────────

def load_all() -> dict[str, dict]:
    """Return all stored agent configs {name: config_dict}. Never raises."""
    try:
        return _gcs_load() if _use_gcs() else _local_load()
    except Exception as exc:
        logger.error("agent_gcs_store.load_all failed: %s", exc)
        return {}


def save_agent(agent_dict: dict) -> None:
    """Upsert one agent config. agent_dict must have 'name' key."""
    name = agent_dict["name"]
    agent_dict = {**agent_dict, "updated_at": datetime.now(timezone.utc).isoformat()}
    data = load_all()
    data[name] = agent_dict
    if _use_gcs():
        _gcs_save(data)
    else:
        _local_save(data)
    logger.info("Saved agent config '%s'", name)


def delete_agent(name: str) -> bool:
    """Delete agent config by name. Returns True if deleted, False if not found."""
    data = load_all()
    if name not in data:
        return False
    del data[name]
    if _use_gcs():
        _gcs_save(data)
    else:
        _local_save(data)
    logger.info("Deleted agent config '%s'", name)
    return True
