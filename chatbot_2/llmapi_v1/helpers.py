"""Helper utilities for LLM API service.

Centralizes repeated chat / completion invocations and safety parsing so
`main.py` remains concise. No external side effects beyond calling OpenAI
and returning structured dicts.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import json, re
from openai import OpenAI


class LLMClient:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def chat(self, system: str | None, user: str, temperature: float = 0.2) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        return resp.choices[0].message.content

    def embed(self, text: str, model: str = "text-embedding-ada-002") -> List[float]:
        emb = self.client.embeddings.create(model=model, input=text)
        return emb.data[0].embedding


def parse_json_object(raw: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None
