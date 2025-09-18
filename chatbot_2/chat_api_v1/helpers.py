"""Helpers for Chat API service."""
from __future__ import annotations

import time, uuid, requests
from sqlalchemy.orm import Session
from orm import Conversation


def ensure_conversation(db: Session, thread_id: str, message_id: str, user_input: str):
    conv = Conversation(thread_id=thread_id, message_id=message_id, user_input=user_input)
    db.merge(conv)
    db.commit()


def call_km(km_api: str, thread_id: str, message_id: str, user_input: str, timeout: int = 60) -> dict:
    payload = {"thread_id": thread_id, "message_id": message_id, "user_input": user_input}
    r = requests.post(f"{km_api}/process", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()
