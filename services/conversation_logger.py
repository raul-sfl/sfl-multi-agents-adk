"""
Firestore-based conversation logger.

Persists every chat session and message so conversations can be audited,
reviewed in the admin dashboard, and recovered by returning users.

Gracefully no-ops when Firestore is unavailable (FIRESTORE_ENABLED=false
or missing GCP credentials) — the rest of the app works unaffected.

Firestore schema
----------------
users/{user_id}
    created_at          Timestamp
    last_seen_at        Timestamp
    preferred_language  str
    conversation_count  int
    last_conversation_id str | None

conversations/{conversation_id}
    user_id             str
    session_id          str  (ADK session UUID)
    started_at          Timestamp
    last_activity_at    Timestamp
    language            str
    agents_used         list[str]
    message_count       int
    status              "active" | "closed"

conversations/{conversation_id}/messages/{auto_id}
    role        "user" | "assistant"
    content     str
    agent       str | None   (assistant only)
    timestamp   Timestamp
"""

import uuid
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

import config

logger = logging.getLogger(__name__)

# ── Firestore singleton ───────────────────────────────────────────────────────

_db_initialized = False
_db = None


def _get_db():
    """Return the Firestore AsyncClient, or None if unavailable."""
    global _db_initialized, _db
    if _db_initialized:
        return _db
    _db_initialized = True
    if not config.FIRESTORE_ENABLED:
        logger.info("Conversation logging disabled (FIRESTORE_ENABLED=false).")
        return None
    try:
        from google.cloud import firestore  # noqa: PLC0415
        kwargs = {}
        if config.GOOGLE_CLOUD_PROJECT:
            kwargs["project"] = config.GOOGLE_CLOUD_PROJECT
        _db = firestore.AsyncClient(**kwargs)
        logger.info("Firestore client ready (project=%s).", config.GOOGLE_CLOUD_PROJECT or "default")
    except Exception as exc:
        logger.warning("Firestore unavailable — conversation logging disabled: %s", exc)
        _db = None
    return _db


def _col(name: str) -> str:
    """Apply the optional collection prefix."""
    return f"{config.FIRESTORE_COLLECTION_PREFIX}{name}"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ts_to_iso(value) -> Optional[str]:
    """Convert a Firestore Timestamp / datetime to ISO-8601 string."""
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _serialize_conv(doc_id: str, data: dict) -> dict:
    return {
        "id": doc_id,
        "user_id": data.get("user_id"),
        "session_id": data.get("session_id"),
        "language": data.get("language"),
        "status": data.get("status"),
        "message_count": data.get("message_count", 0),
        "agents_used": data.get("agents_used", []),
        "started_at": _ts_to_iso(data.get("started_at")),
        "last_activity_at": _ts_to_iso(data.get("last_activity_at")),
    }


def _serialize_msg(doc_id: str, data: dict) -> dict:
    return {
        "id": doc_id,
        "role": data.get("role"),
        "content": data.get("content"),
        "agent": data.get("agent"),
        "timestamp": _ts_to_iso(data.get("timestamp")),
    }


# ── ConversationLogger ────────────────────────────────────────────────────────

class ConversationLogger:
    """All public methods are fire-and-forget safe: they never raise."""

    # ── User ─────────────────────────────────────────────────────────────────

    async def ensure_user(self, user_id: str, language: str) -> None:
        """Create user document on first visit; update last_seen on return."""
        db = _get_db()
        if not db:
            return
        try:
            from google.cloud import firestore  # noqa: PLC0415
            now = datetime.now(timezone.utc)
            ref = db.collection(_col("users")).document(user_id)
            snap = await ref.get()
            if not snap.exists:
                await ref.set({
                    "created_at": now,
                    "last_seen_at": now,
                    "preferred_language": language,
                    "conversation_count": 0,
                    "last_conversation_id": None,
                })
            else:
                await ref.update({"last_seen_at": now})
        except Exception as exc:
            logger.error("ensure_user(%s): %s", user_id, exc)

    # ── Conversation lifecycle ────────────────────────────────────────────────

    async def create_conversation(
        self, user_id: str, session_id: str, language: str
    ) -> str:
        """Open a new conversation. Returns its Firestore document ID."""
        conv_id = str(uuid.uuid4())
        db = _get_db()
        if not db:
            return conv_id
        try:
            from google.cloud import firestore  # noqa: PLC0415
            now = datetime.now(timezone.utc)
            await db.collection(_col("conversations")).document(conv_id).set({
                "user_id": user_id,
                "session_id": session_id,
                "started_at": now,
                "last_activity_at": now,
                "language": language,
                "agents_used": [],
                "message_count": 0,
                "status": "active",
            })
            await db.collection(_col("users")).document(user_id).update({
                "last_conversation_id": conv_id,
                "conversation_count": firestore.Increment(1),
            })
        except Exception as exc:
            logger.error("create_conversation(%s): %s", user_id, exc)
        return conv_id

    async def log_message(
        self,
        conv_id: str,
        role: str,
        content: str,
        agent: Optional[str] = None,
    ) -> None:
        """Append a message to an open conversation."""
        if not conv_id:
            return
        db = _get_db()
        if not db:
            return
        try:
            from google.cloud import firestore  # noqa: PLC0415
            now = datetime.now(timezone.utc)
            conv_ref = db.collection(_col("conversations")).document(conv_id)
            await conv_ref.collection("messages").add({
                "role": role,
                "content": content,
                "agent": agent,
                "timestamp": now,
            })
            update: dict = {
                "last_activity_at": now,
                "message_count": firestore.Increment(1),
            }
            if agent and role == "assistant":
                update["agents_used"] = firestore.ArrayUnion([agent])
            await conv_ref.update(update)
        except Exception as exc:
            logger.error("log_message(%s): %s", conv_id, exc)

    async def end_conversation(self, conv_id: str) -> None:
        """Mark a conversation as closed (called on WebSocket disconnect)."""
        if not conv_id:
            return
        db = _get_db()
        if not db:
            return
        try:
            await db.collection(_col("conversations")).document(conv_id).update({
                "status": "closed",
                "last_activity_at": datetime.now(timezone.utc),
            })
        except Exception as exc:
            logger.error("end_conversation(%s): %s", conv_id, exc)

    # ── History recovery ──────────────────────────────────────────────────────

    async def get_recent_conversation(self, user_id: str) -> Optional[dict]:
        """
        Return the most recent conversation for this user if it's within the
        HISTORY_RECOVERY_HOURS window, otherwise None.
        """
        db = _get_db()
        if not db:
            return None
        try:
            user_snap = await db.collection(_col("users")).document(user_id).get()
            if not user_snap.exists:
                return None
            last_conv_id = user_snap.to_dict().get("last_conversation_id")
            if not last_conv_id:
                return None

            conv_snap = await db.collection(_col("conversations")).document(last_conv_id).get()
            if not conv_snap.exists:
                return None

            conv_data = conv_snap.to_dict()
            last_activity = conv_data.get("last_activity_at")
            if last_activity:
                cutoff = datetime.now(timezone.utc) - timedelta(hours=config.HISTORY_RECOVERY_HOURS)
                # last_activity may be a DatetimeWithNanoseconds (always UTC-aware)
                ts = last_activity
                if hasattr(ts, "tzinfo") and ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts < cutoff:
                    return None

            return _serialize_conv(conv_snap.id, conv_data)
        except Exception as exc:
            logger.error("get_recent_conversation(%s): %s", user_id, exc)
            return None

    async def get_conversation_messages(
        self, conv_id: str, limit: int = 50
    ) -> list[dict]:
        """Return messages for a conversation ordered by timestamp."""
        db = _get_db()
        if not db:
            return []
        try:
            snaps = await (
                db.collection(_col("conversations"))
                .document(conv_id)
                .collection("messages")
                .order_by("timestamp")
                .limit(limit)
                .get()
            )
            return [_serialize_msg(s.id, s.to_dict()) for s in snaps]
        except Exception as exc:
            logger.error("get_conversation_messages(%s): %s", conv_id, exc)
            return []

    # ── Admin queries ─────────────────────────────────────────────────────────

    async def list_conversations(
        self,
        user_id: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict]:
        """
        List recent conversations, optionally filtered by user_id.
        Results are sorted newest-first.
        """
        db = _get_db()
        if not db:
            return []
        try:
            from google.cloud import firestore  # noqa: PLC0415
            col = db.collection(_col("conversations"))
            if user_id:
                # Avoid composite-index requirement: filter in Firestore by equality,
                # sort in Python.
                snaps = await col.where("user_id", "==", user_id).limit(limit * 4).get()
                snaps = sorted(
                    snaps,
                    key=lambda s: s.to_dict().get("last_activity_at")
                    or datetime.min.replace(tzinfo=timezone.utc),
                    reverse=True,
                )[:limit]
            else:
                snaps = await (
                    col.order_by(
                        "last_activity_at",
                        direction=firestore.Query.DESCENDING,
                    )
                    .limit(limit)
                    .get()
                )
            return [_serialize_conv(s.id, s.to_dict()) for s in snaps]
        except Exception as exc:
            logger.error("list_conversations: %s", exc)
            return []

    async def get_conversation(self, conv_id: str) -> Optional[dict]:
        """Return metadata for a single conversation."""
        db = _get_db()
        if not db:
            return None
        try:
            snap = await db.collection(_col("conversations")).document(conv_id).get()
            if not snap.exists:
                return None
            return _serialize_conv(snap.id, snap.to_dict())
        except Exception as exc:
            logger.error("get_conversation(%s): %s", conv_id, exc)
            return None

    async def list_user_conversations(
        self, user_id: str, limit: int = 20
    ) -> list[dict]:
        return await self.list_conversations(user_id=user_id, limit=limit)

    async def get_stats(self) -> dict:
        """Return aggregate counts for the dashboard stats bar."""
        db = _get_db()
        if not db:
            return {}
        try:
            today = datetime.now(timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            col = db.collection(_col("conversations"))
            users_col = db.collection(_col("users"))

            all_convs, today_convs, active_convs, all_users = await _gather(
                col.get(),
                col.where("started_at", ">=", today).get(),
                col.where("status", "==", "active").get(),
                users_col.get(),
            )
            return {
                "total_conversations": len(all_convs),
                "conversations_today": len(today_convs),
                "active_conversations": len(active_convs),
                "total_users": len(all_users),
            }
        except Exception as exc:
            logger.error("get_stats: %s", exc)
            return {}


async def _gather(*coros):
    """Run coroutines concurrently and return results in order."""
    import asyncio
    return await asyncio.gather(*coros)


# Singleton instance used across the application
conversation_logger = ConversationLogger()
