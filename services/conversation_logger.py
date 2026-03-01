"""
Cloud Logging-based conversation logger.

Replaces the previous Firestore implementation. All conversation events are
written as structured JSON log entries to a named log in Cloud Logging.

Log name: CLOUD_LOGGING_LOG_NAME (default "stayforlong-conversations")
GCP project: GOOGLE_CLOUD_PROJECT

Event types written
-------------------
conversation_start  — one per WebSocket session (open)
message             — one per user/assistant turn
conversation_end    — one per WebSocket disconnect
tag_update          — one per quality-tag change from the admin dashboard

Querying
--------
Cloud Logging filter syntax is used for all admin queries. Entries have both
jsonPayload (structured data) and labels (indexed, cheap to filter on).

Graceful degradation
--------------------
All methods silently no-op when Cloud Logging is unavailable or
CLOUD_LOGGING_ENABLED=false. The rest of the app continues unaffected.
"""

import asyncio
import uuid
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

import config

logger = logging.getLogger(__name__)

# ── Cloud Logging singleton ───────────────────────────────────────────────────

_cl_initialized = False
_cl_client = None   # google.cloud.logging.Client
_cl_logger = None   # google.cloud.logging.Logger (named log)


def _get_cl():
    """Return (client, logger) tuple, or (None, None) if unavailable."""
    global _cl_initialized, _cl_client, _cl_logger
    if _cl_initialized:
        return _cl_client, _cl_logger
    _cl_initialized = True
    if not config.CLOUD_LOGGING_ENABLED:
        logger.info("Conversation logging disabled (CLOUD_LOGGING_ENABLED=false).")
        return None, None
    if not config.GOOGLE_CLOUD_PROJECT:
        logger.warning("GOOGLE_CLOUD_PROJECT not set — conversation logging disabled.")
        return None, None
    try:
        from google.cloud import logging as cloud_logging  # noqa: PLC0415
        _cl_client = cloud_logging.Client(project=config.GOOGLE_CLOUD_PROJECT)
        _cl_logger = _cl_client.logger(config.CLOUD_LOGGING_LOG_NAME)
        logger.info(
            "Cloud Logging ready (project=%s, log=%s).",
            config.GOOGLE_CLOUD_PROJECT,
            config.CLOUD_LOGGING_LOG_NAME,
        )
    except Exception as exc:
        logger.warning("Cloud Logging unavailable — conversation logging disabled: %s", exc)
        _cl_client, _cl_logger = None, None
    return _cl_client, _cl_logger


def _log_name() -> str:
    return (
        f"projects/{config.GOOGLE_CLOUD_PROJECT}"
        f"/logs/{config.CLOUD_LOGGING_LOG_NAME}"
    )


# ── Serializers ───────────────────────────────────────────────────────────────

def _ts_to_iso(value) -> Optional[str]:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _iso_to_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def _entry_to_conv(entry) -> dict:
    payload = entry.payload if isinstance(entry.payload, dict) else {}
    ts = entry.timestamp
    return {
        "id":               payload.get("conversation_id", ""),
        "user_id":          payload.get("user_id", ""),
        "session_id":       payload.get("session_id", ""),
        "language":         payload.get("language", ""),
        "status":           payload.get("status", "active"),
        "message_count":    payload.get("message_count", 0),
        "agents_used":      payload.get("agents_used", []),
        "tags":             payload.get("tags", []),
        "started_at":       _ts_to_iso(ts),
        "last_activity_at": payload.get("last_activity_at") or _ts_to_iso(ts),
    }


def _entry_to_msg(entry) -> dict:
    payload = entry.payload if isinstance(entry.payload, dict) else {}
    return {
        "id":        entry.insert_id or "",
        "role":      payload.get("role", ""),
        "content":   payload.get("content", ""),
        "agent":     payload.get("agent"),
        "timestamp": _ts_to_iso(entry.timestamp),
    }


# ── ConversationLogger ────────────────────────────────────────────────────────

class ConversationLogger:
    """All public methods are fire-and-forget safe: they never raise."""

    # ── Write events ──────────────────────────────────────────────────────────

    async def log_conversation_start(
        self, conv_id: str, user_id: str, session_id: str, language: str
    ) -> None:
        _client, cl = _get_cl()
        if not cl:
            return
        try:
            now = datetime.now(timezone.utc).isoformat()
            cl.log_struct(
                {
                    "event_type":       "conversation_start",
                    "conversation_id":  conv_id,
                    "user_id":          user_id,
                    "session_id":       session_id,
                    "language":         language,
                    "status":           "active",
                    "tags":             [],
                    "agents_used":      [],
                    "message_count":    0,
                    "last_activity_at": now,
                },
                labels={
                    "conversation_id": conv_id,
                    "user_id":         user_id,
                    "event_type":      "conversation_start",
                    "language":        language,
                },
                severity="INFO",
            )
        except Exception as exc:
            logger.error("log_conversation_start(%s): %s", conv_id, exc)

    async def log_message(
        self,
        conv_id: str,
        role: str,
        content: str,
        agent: Optional[str] = None,
    ) -> None:
        if not conv_id:
            return
        _client, cl = _get_cl()
        if not cl:
            return
        try:
            cl.log_struct(
                {
                    "event_type":      "message",
                    "conversation_id": conv_id,
                    "role":            role,
                    "content":         content,
                    "agent":           agent,
                    "timestamp":       datetime.now(timezone.utc).isoformat(),
                },
                labels={
                    "conversation_id": conv_id,
                    "event_type":      "message",
                    "role":            role,
                },
                severity="INFO",
            )
        except Exception as exc:
            logger.error("log_message(%s): %s", conv_id, exc)

    async def log_conversation_end(self, conv_id: str) -> None:
        if not conv_id:
            return
        _client, cl = _get_cl()
        if not cl:
            return
        try:
            cl.log_struct(
                {
                    "event_type":      "conversation_end",
                    "conversation_id": conv_id,
                    "status":          "closed",
                },
                labels={
                    "conversation_id": conv_id,
                    "event_type":      "conversation_end",
                },
                severity="INFO",
            )
        except Exception as exc:
            logger.error("log_conversation_end(%s): %s", conv_id, exc)

    async def set_conversation_tags(self, conv_id: str, tags: list[str]) -> None:
        if not conv_id:
            return
        _client, cl = _get_cl()
        if not cl:
            return
        try:
            cl.log_struct(
                {
                    "event_type":      "tag_update",
                    "conversation_id": conv_id,
                    "tags":            tags,
                },
                labels={
                    "conversation_id": conv_id,
                    "event_type":      "tag_update",
                },
                severity="INFO",
            )
        except Exception as exc:
            logger.error("set_conversation_tags(%s): %s", conv_id, exc)

    # ── Admin read queries ────────────────────────────────────────────────────

    async def list_conversations(
        self,
        user_id: Optional[str] = None,
        limit: int = 50,
        cursor: Optional[str] = None,       # page_token from Cloud Logging
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        agent: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> dict:
        """
        List conversations ordered newest-first.
        Returns {"items": [...], "next_cursor": page_token | None}.
        """
        cl_client, _cl = _get_cl()
        if not cl_client:
            return {"items": [], "next_cursor": None}
        try:
            from google.cloud import logging as cloud_logging  # noqa: PLC0415

            filters = [
                f'logName="{_log_name()}"',
                'jsonPayload.event_type="conversation_start"',
            ]
            if user_id:
                filters.append(f'labels.user_id="{user_id}"')
            if date_from:
                filters.append(f'timestamp >= "{date_from}"')
            if date_to:
                filters.append(f'timestamp <= "{date_to}"')

            filter_str = " AND ".join(filters)
            fetch_limit = limit * 4 if (agent or tag) else limit + 1

            page_iter = cl_client.list_entries(
                filter_=filter_str,
                order_by=cloud_logging.DESCENDING,
                page_size=fetch_limit,
            )
            entries = list(page_iter)

            # Enrich with latest tags, message counts and agents
            conv_ids = [
                e.payload.get("conversation_id")
                for e in entries
                if isinstance(e.payload, dict)
            ]
            tag_map, msg_stats = await asyncio.gather(
                self._latest_tags_for(conv_ids),
                self._message_stats_for(conv_ids),
            )

            items = []
            for entry in entries:
                c = _entry_to_conv(entry)
                if tag_map.get(c["id"]):
                    c["tags"] = tag_map[c["id"]]
                stats = msg_stats.get(c["id"])
                if stats:
                    c["message_count"] = stats["count"]
                    c["agents_used"]   = stats["agents"]
                    if stats["last_at"]:
                        c["last_activity_at"] = stats["last_at"]
                if agent and agent not in (c.get("agents_used") or []):
                    continue
                if tag and tag not in (c.get("tags") or []):
                    continue
                items.append(c)
                if len(items) >= limit:
                    break

            next_cursor = (
                getattr(page_iter, "next_page_token", None)
                or (entries[-1].insert_id if len(entries) > limit else None)
            )
            return {"items": items[:limit], "next_cursor": next_cursor}
        except Exception as exc:
            logger.error("list_conversations: %s", exc)
            return {"items": [], "next_cursor": None}

    async def get_conversation(self, conv_id: str) -> Optional[dict]:
        cl_client, _cl = _get_cl()
        if not cl_client:
            return None
        try:
            from google.cloud import logging as cloud_logging  # noqa: PLC0415

            entries = list(cl_client.list_entries(
                filter_=(
                    f'logName="{_log_name()}"'
                    f' AND jsonPayload.event_type="conversation_start"'
                    f' AND labels.conversation_id="{conv_id}"'
                ),
                order_by=cloud_logging.DESCENDING,
                page_size=1,
            ))
            if not entries:
                return None
            c = _entry_to_conv(entries[0])
            # Enrich with tags, message stats and closed status
            latest, msg_stats, closed = await asyncio.gather(
                self._latest_tags_for([conv_id]),
                self._message_stats_for([conv_id]),
                asyncio.to_thread(lambda: list(cl_client.list_entries(
                    filter_=(
                        f'logName="{_log_name()}"'
                        f' AND jsonPayload.event_type="conversation_end"'
                        f' AND labels.conversation_id="{conv_id}"'
                    ),
                    page_size=1,
                ))),
            )
            if latest.get(conv_id):
                c["tags"] = latest[conv_id]
            stats = msg_stats.get(conv_id)
            if stats:
                c["message_count"] = stats["count"]
                c["agents_used"]   = stats["agents"]
                if stats["last_at"]:
                    c["last_activity_at"] = stats["last_at"]
            if closed:
                c["status"] = "closed"
            return c
        except Exception as exc:
            logger.error("get_conversation(%s): %s", conv_id, exc)
            return None

    async def get_conversation_messages(
        self, conv_id: str, limit: int = 100
    ) -> list[dict]:
        cl_client, _cl = _get_cl()
        if not cl_client:
            return []
        try:
            from google.cloud import logging as cloud_logging  # noqa: PLC0415

            entries = list(cl_client.list_entries(
                filter_=(
                    f'logName="{_log_name()}"'
                    f' AND jsonPayload.event_type="message"'
                    f' AND labels.conversation_id="{conv_id}"'
                ),
                order_by=cloud_logging.ASCENDING,
                page_size=limit,
            ))
            return [_entry_to_msg(e) for e in entries]
        except Exception as exc:
            logger.error("get_conversation_messages(%s): %s", conv_id, exc)
            return []

    async def get_stats(self) -> dict:
        cl_client, _cl = _get_cl()
        if not cl_client:
            return {}
        try:
            from google.cloud import logging as cloud_logging  # noqa: PLC0415
            import asyncio

            now = datetime.now(timezone.utc)
            today_iso = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
            # "Active now" = started within last 2 h and not yet closed
            active_cutoff_iso = (now - timedelta(hours=2)).isoformat()
            base = f'logName="{_log_name()}"'

            def _count(extra: str) -> int:
                f = f'{base} AND {extra}'
                return sum(1 for _ in cl_client.list_entries(filter_=f, page_size=500))

            def _unique_users() -> int:
                return len({
                    e.labels.get("user_id", "")
                    for e in cl_client.list_entries(
                        filter_=f'{base} AND jsonPayload.event_type="conversation_start"',
                        page_size=500,
                    )
                    if e.labels.get("user_id")
                })

            def _active_conversations() -> int:
                # Conversations started in the last 2 h with no conversation_end
                recent_ids = {
                    e.labels.get("conversation_id")
                    for e in cl_client.list_entries(
                        filter_=(
                            f'{base} AND jsonPayload.event_type="conversation_start"'
                            f' AND timestamp >= "{active_cutoff_iso}"'
                        ),
                        page_size=200,
                    )
                    if e.labels.get("conversation_id")
                }
                if not recent_ids:
                    return 0
                closed_ids = {
                    e.labels.get("conversation_id")
                    for e in cl_client.list_entries(
                        filter_=(
                            f'{base} AND jsonPayload.event_type="conversation_end"'
                            f' AND timestamp >= "{active_cutoff_iso}"'
                        ),
                        page_size=200,
                    )
                    if e.labels.get("conversation_id")
                }
                return len(recent_ids - closed_ids)

            total, today_count, unique_users, active_count = await asyncio.gather(
                asyncio.to_thread(_count, 'jsonPayload.event_type="conversation_start"'),
                asyncio.to_thread(_count, f'jsonPayload.event_type="conversation_start" AND timestamp >= "{today_iso}"'),
                asyncio.to_thread(_unique_users),
                asyncio.to_thread(_active_conversations),
            )

            return {
                "total_conversations":  total,
                "conversations_today":  today_count,
                "active_conversations": active_count,
                "total_users":          unique_users,
            }
        except Exception as exc:
            logger.error("get_stats: %s", exc)
            return {}

    async def export_conversations(
        self,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        status: Optional[str] = None,
        language: Optional[str] = None,
        agent: Optional[str] = None,
        tag: Optional[str] = None,
        limit: int = 1000,
    ) -> list[dict]:
        result = await self.list_conversations(
            limit=limit,
            date_from=date_from,
            date_to=date_to,
            agent=agent,
            tag=tag,
        )
        items = result["items"]
        if status:
            items = [c for c in items if c.get("status") == status]
        if language:
            items = [c for c in items if c.get("language") == language]
        return items

    # ── Internal helpers ──────────────────────────────────────────────────────

    async def _message_stats_for(self, conv_ids: list[str]) -> dict[str, dict]:
        """Return {conv_id: {count, agents, last_at}} from message events."""
        if not conv_ids:
            return {}
        cl_client, _cl = _get_cl()
        if not cl_client:
            return {}
        try:
            from google.cloud import logging as cloud_logging  # noqa: PLC0415

            id_clause = " OR ".join(
                f'labels.conversation_id="{cid}"' for cid in conv_ids
            )
            entries = list(cl_client.list_entries(
                filter_=(
                    f'logName="{_log_name()}"'
                    f' AND jsonPayload.event_type="message"'
                    f' AND ({id_clause})'
                ),
                order_by=cloud_logging.ASCENDING,
                page_size=min(len(conv_ids) * 100, 1000),
            ))
            result: dict[str, dict] = {}
            for e in entries:
                payload = e.payload if isinstance(e.payload, dict) else {}
                cid = payload.get("conversation_id", "")
                if not cid:
                    continue
                if cid not in result:
                    result[cid] = {"count": 0, "agents": [], "last_at": None}
                result[cid]["count"] += 1
                agent = payload.get("agent")
                if agent and agent not in result[cid]["agents"]:
                    result[cid]["agents"].append(agent)
                result[cid]["last_at"] = _ts_to_iso(e.timestamp)
            return result
        except Exception as exc:
            logger.error("_message_stats_for: %s", exc)
            return {}

    async def _latest_tags_for(self, conv_ids: list[str]) -> dict[str, list[str]]:
        """Return {conv_id: [tags]} using the latest tag_update entry per conv."""
        if not conv_ids:
            return {}
        cl_client, _cl = _get_cl()
        if not cl_client:
            return {}
        try:
            from google.cloud import logging as cloud_logging  # noqa: PLC0415

            id_clause = " OR ".join(
                f'labels.conversation_id="{cid}"' for cid in conv_ids
            )
            entries = list(cl_client.list_entries(
                filter_=(
                    f'logName="{_log_name()}"'
                    f' AND jsonPayload.event_type="tag_update"'
                    f' AND ({id_clause})'
                ),
                order_by=cloud_logging.DESCENDING,
                page_size=len(conv_ids) * 5,
            ))
            result: dict[str, list[str]] = {}
            for e in entries:
                payload = e.payload if isinstance(e.payload, dict) else {}
                cid = payload.get("conversation_id", "")
                if cid and cid not in result:
                    result[cid] = payload.get("tags", [])
            return result
        except Exception as exc:
            logger.error("_latest_tags_for: %s", exc)
            return {}


# Singleton instance used across the application
conversation_logger = ConversationLogger()
