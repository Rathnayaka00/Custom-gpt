from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pymongo.errors import PyMongoError

from app.core.config import settings
from app.db.mongo import get_mongo_db

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


async def ensure_session_indexes() -> None:
    """
    Creates a TTL index so sessions auto-expire.
    Safe to call repeatedly.
    """
    db = get_mongo_db()
    coll = db[settings.MONGODB_SESSIONS_COLLECTION]
    try:
        # TTL: expires_at must be a BSON datetime (timezone-aware is OK; Mongo stores UTC)
        await coll.create_index("session_id", unique=True)
        await coll.create_index("expires_at", expireAfterSeconds=0)
    except Exception as e:
        # Don't crash the app on startup for index issues; log and keep running.
        logger.warning("Failed to ensure Mongo session indexes: %s", e)


async def create_session(document_ids: List[str]) -> str:
    session_id = str(uuid4())
    now = _utcnow()
    expires_at = now + timedelta(days=int(settings.MONGODB_SESSION_TTL_DAYS))

    doc = {
        "session_id": session_id,
        "document_ids": list(dict.fromkeys([d for d in (document_ids or []) if d])),
        "messages": [],
        "created_at": now,
        "updated_at": now,
        "expires_at": expires_at,
    }

    db = get_mongo_db()
    coll = db[settings.MONGODB_SESSIONS_COLLECTION]
    try:
        await coll.insert_one(doc)
    except PyMongoError as e:
        logger.exception("Failed to create session")
        raise RuntimeError(f"MongoDB error creating session: {e}") from e

    return session_id


async def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    if not session_id:
        return None
    db = get_mongo_db()
    coll = db[settings.MONGODB_SESSIONS_COLLECTION]
    try:
        return await coll.find_one({"session_id": session_id}, {"_id": 0})
    except PyMongoError as e:
        logger.exception("Failed to fetch session")
        raise RuntimeError(f"MongoDB error fetching session: {e}") from e


async def append_messages(session_id: str, messages: List[Dict[str, Any]]) -> None:
    if not session_id or not messages:
        return
    now = _utcnow()
    enriched = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or "").strip()
        content = str(m.get("content") or "").strip()
        if not role or not content:
            continue
        enriched.append({"role": role, "content": content, "created_at": now})
    if not enriched:
        return

    db = get_mongo_db()
    coll = db[settings.MONGODB_SESSIONS_COLLECTION]
    try:
        await coll.update_one(
            {"session_id": session_id},
            {
                "$push": {"messages": {"$each": enriched}},
                "$set": {"updated_at": now},
            },
        )
    except PyMongoError as e:
        logger.exception("Failed to append messages")
        raise RuntimeError(f"MongoDB error appending messages: {e}") from e


async def append_generated_prompt(
    session_id: str,
    *,
    intent: str,
    prompt: str,
    document_ids: Optional[List[str]] = None,
    attachment_details: Optional[str] = None,
    used_context: bool = False,
) -> None:
    if not session_id:
        return
    now = _utcnow()
    item: Dict[str, Any] = {
        "intent": (intent or "").strip(),
        "prompt": (prompt or "").strip(),
        "document_ids": list(dict.fromkeys([d for d in (document_ids or []) if d])),
        "used_context": bool(used_context),
        "created_at": now,
    }
    if attachment_details and attachment_details.strip():
        item["attachment_details"] = attachment_details.strip()

    db = get_mongo_db()
    coll = db[settings.MONGODB_SESSIONS_COLLECTION]
    try:
        await coll.update_one(
            {"session_id": session_id},
            {
                "$push": {"generated_prompts": item},
                "$set": {"updated_at": now},
            },
        )
    except PyMongoError as e:
        logger.exception("Failed to append generated prompt")
        raise RuntimeError(f"MongoDB error appending generated prompt: {e}") from e


async def append_execution(
    session_id: str,
    *,
    query: str,
    prompt_used: str,
    answer: str,
    document_ids: Optional[List[str]] = None,
    used_context: bool = False,
) -> None:
    if not session_id:
        return
    now = _utcnow()
    item: Dict[str, Any] = {
        "query": (query or "").strip(),
        "prompt_used": (prompt_used or "").strip(),
        "answer": (answer or "").strip(),
        "document_ids": list(dict.fromkeys([d for d in (document_ids or []) if d])),
        "used_context": bool(used_context),
        "created_at": now,
    }
    db = get_mongo_db()
    coll = db[settings.MONGODB_SESSIONS_COLLECTION]
    try:
        await coll.update_one(
            {"session_id": session_id},
            {"$push": {"executions": item}, "$set": {"updated_at": now}},
        )
    except PyMongoError as e:
        logger.exception("Failed to append execution")
        raise RuntimeError(f"MongoDB error appending execution: {e}") from e


