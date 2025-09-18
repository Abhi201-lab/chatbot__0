"""Helper utilities for ingestion service.

Provides small focused functions for:
  * Splitting documents
  * Generating (optional normalized) embeddings
  * Bulk persisting VectorChunk rows
"""
from __future__ import annotations

from typing import List, Sequence
import math
from sqlalchemy.orm import Session
from orm import VectorChunk


def maybe_normalize(embeddings: Sequence[Sequence[float]], enable: bool) -> List[List[float]]:
    if not enable:
        return [list(e) for e in embeddings]
    normed: List[List[float]] = []
    for emb in embeddings:
        s = sum(v * v for v in emb) or 1.0
        l2 = math.sqrt(s)
        normed.append([v / l2 for v in emb])
    return normed


def persist_chunks(db: Session, texts, metas, embeddings) -> int:
    rows = []
    for idx, (txt, meta, emb) in enumerate(zip(texts, metas, embeddings)):
        rows.append(VectorChunk(
            thread_id=meta.get("thread_id"),
            source=meta.get("source") or meta.get("original_source"),
            chunk_index=meta.get("chunk_index", idx),
            content=txt,
            embedding=emb,
        ))
    db.bulk_save_objects(rows)
    db.commit()
    return len(rows)
