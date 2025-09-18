"""Knowledge Manager API

Core responsibilities:
    * Pre-safety inspection of user query
    * Intent classification & out-of-scope gating
    * Query rephrasing (semantic search optimization)
    * Hybrid retrieval (pgvector preferred, FAISS fallback)
    * Answer synthesis via LLM
    * Post-safety inspection & grounding heuristic enforcement
    * Optional structured tracing for observability

Environment flags control behavior (see README_RAG_SIMPLE.md once generated).
"""

import sys
from fastapi import FastAPI, Request
import time, requests, re

from environment import load_env, env
from logger import get_logger
from model import KMRequest
from vectorstore import PGVectorRetriever
import uuid as _uuid

# new import for prompt templates
from prompts.prompt_loader import format_prompt, PromptNotFoundError
from pipeline import (
    Config, TraceState, safety_pre, classify_intent, rephrase_query,
    retrieve_and_build_prompt, synthesize_answer, safety_post,
    enforce_grounding, build_response, embed
)


# ensure logs go to stdout so uvicorn displays them
load_env()
log = get_logger("km_api")

LLM_API = env("LLM_API_URL")
INCLUDE_DEBUG_ANSWER = env("INCLUDE_DEBUG_ANSWER", default="0", required=False) in ("1", "true", "TRUE", "yes")
ENFORCE_GROUNDED = env("ENFORCE_GROUNDED", default="1", required=False) in ("1", "true", "TRUE", "yes")
GROUNDING_MIN_OVERLAP = float(env("GROUNDING_MIN_OVERLAP", default="0.07", required=False))
# Minimum similarity (0-1) required before we consider the retrieval "confident".
# Lowering this allows more low-confidence contexts through instead of returning
# an immediate unknown. User requested a quick relaxation; default moved 0.40 -> 0.30.
RETRIEVAL_MIN_SCORE = float(env("RETRIEVAL_MIN_SCORE", default="0.30", required=False))
DOMAIN_KEYWORDS = {
    "pay","payment","online","bill","rebate","receipt","ecs","deposit","tariff","meter",
    "load","reconnect","ac","air","conditioner","wiring","security","advance"
}
VECTOR_K = int(env("VECTOR_K", default="4", required=False))
DATABASE_URL = env("DATABASE_URL")  # now required
TRACE_DEFAULT = env("TRACE_DEFAULT", default="0", required=False) in ("1","true","TRUE","yes")
NORMALIZE_EMBEDDINGS = env("NORMALIZE_EMBEDDINGS", default="1", required=False) in ("1","true","TRUE","yes")

pg_retriever = None


def get_pg_retriever():
    global pg_retriever
    if pg_retriever is None:
        try:
            pg_retriever = PGVectorRetriever(DATABASE_URL)
        except Exception:
            log.exception("Failed to init PGVector retriever")
            pg_retriever = None
    return pg_retriever


app = FastAPI(title="Knowledge Manager API", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/vector_stats")
def vector_stats():
    """Basic stats for pgvector table (row count)."""
    retr = get_pg_retriever()
    if retr is None:
        return {"status": "error", "reason": "retriever not available"}
    # lightweight count
    try:
        from sqlalchemy import create_engine, text
        eng = retr.engine
        with eng.connect() as conn:
            count = conn.execute(text("SELECT count(*) FROM vector_chunks")).scalar() or 0
        return {"status": "ok", "row_count": int(count)}
    except Exception:
        log.exception("Failed to fetch vector stats")
        return {"status": "error", "reason": "stats_failed"}


@app.middleware("http")
async def log_requests(request: Request, call_next):
    log.info(f"Incoming request: {request.method} {request.url}")
    try:
        resp = await call_next(request)
        log.info(f"Request completed: {request.method} {request.url} status={resp.status_code}")
        return resp
    except Exception:
        log.exception("Unhandled exception in request pipeline")
        return {"bot_output": "Internal server error", "citations": []}


def _call_llm_synthesize(query: str, context: str, timeout_sec: int = 15) -> str:
    """Wrapper for synthesize LLM endpoint (robust to failures)."""
    try:
        resp = requests.post(
            f"{LLM_API}/synthesize",
            json={"query": query, "context": context},
            timeout=timeout_sec,
        )
        resp.raise_for_status()
        return resp.json().get("answer", "")
    except Exception:
        log.exception("LLM synthesize call failed")
        return ""


def _maybe_trace(trace_events, trace_id: str, start_time: float, event: str, **data):
    """Append a structured trace event and mirror to logs."""
    if trace_events is not None:
        trace_events.append({
            "t": round(time.time() - start_time, 4),
            "event": event,
            **data
        })
    log.info("TRACE[%s] %s %s", trace_id, event, data if data else "")


def _embed_query_via_api(text: str, timings: dict, trace_events, trace_id, t0):
    """Call central embedding endpoint and return vector list[float]."""
    emb = []
    try:
        t_emb = time.time()
        r = requests.post(f"{LLM_API}/embed", json={"text": text}, timeout=15)
        r.raise_for_status()
        emb = r.json().get("embedding", [])
        if NORMALIZE_EMBEDDINGS and emb:
            import math
            s = sum(e*e for e in emb) or 1.0
            l2 = math.sqrt(s)
            emb = [e / l2 for e in emb]
        timings['embedding'] = time.time() - t_emb
        _maybe_trace(trace_events, trace_id, t0, "embedded_query", dim=len(emb))
    except Exception:
        log.exception("Embedding API call failed")
    return emb


def _retrieve_pgvector(query_embedding, k: int, timings: dict, trace_events, trace_id, t0):
    contexts, citations, retrieval_meta = [], [], []
    retr = get_pg_retriever()
    if retr and query_embedding:
        try:
            t_r = time.time(); rows = retr.similarity_search(query_embedding, k=k); timings['retrieve'] = time.time() - t_r
            for content, score, source, vid in rows:
                contexts.append(content)
                citations.append({"source": source, "id": vid, "score": score})
                retrieval_meta.append({"score": score, "source": source, "id": vid})
            _maybe_trace(trace_events, trace_id, t0, "retrieval_results", backend='pgvector', hits=len(rows), k=k)
        except Exception:
            log.exception("PGVector retrieval failed")
    return contexts, citations, retrieval_meta


# Simple domain typo corrections to improve recall (before embedding)
COMMON_TYPO_MAP = {
    "reciept": "receipt",
    "recipt": "receipt",
    "paymnt": "payment",
    "recieved": "received",
    "receieved": "received",
    "adress": "address",
}

# Acronym / shorthand expansions (additive). We may run a second retrieval pass
# with these expansions appended if the first pass yields no contexts.
ACRONYM_EXPANSIONS = {
    "ecs": ["electronic clearing service"],
    # Add domain-specific acronyms here.
}

def _expand_query_variants(original: str) -> list[str]:
    """Return a ranked list of variant queries for multi-pass retrieval.

    Strategy:
      1. Start with typo-corrected base.
      2. If contains known acronym, append expansions as separate variants and also
         a fusion variant original + expansion phrase (improves recall when docs
         contain expanded form only).
    """
    base = _apply_typo_corrections(original)
    variants = [base]
    lower = base.lower()
    for acro, expansions in ACRONYM_EXPANSIONS.items():
        if acro in lower.split():  # crude token check
            for exp in expansions:
                if exp not in lower:
                    variants.append(exp)
                    variants.append(f"{base} {exp}")
    # Deduplicate preserving order
    seen = set(); out = []
    for v in variants:
        if v not in seen:
            seen.add(v); out.append(v)
    return out

def _apply_typo_corrections(text: str) -> str:
    lower = text.lower()
    for wrong, right in COMMON_TYPO_MAP.items():
        if wrong in lower:
            lower = lower.replace(wrong, right)
    return lower if text.islower() else lower  # keep simple (could preserve case later)


@app.get("/debug_retrieve")
def debug_retrieve(q: str, k: int = 4):
    """Diagnostic endpoint: shows raw retrieval rows (content preview + score)."""
    t0 = time.time()
    q_fixed = _apply_typo_corrections(q)
    emb = _embed_query_via_api(q_fixed, {}, None, "debug", t0)
    if not emb:
        return {"query": q, "fixed": q_fixed, "error": "embed_failed"}
    contexts, citations, meta = _retrieve_pgvector(emb, k, {}, None, "debug", t0)
    previews = [c[:160].replace('\n',' ') for c in contexts]
    return {"query": q, "fixed": q_fixed, "hits": len(contexts), "previews": previews, "citations": citations}


@app.post("/process")
def process(req: KMRequest):
    """Full RAG pipeline (modularized).

    Optional tracing: pass {"trace": true} to receive internal events.
    """
    cfg = Config(
        llm_api=LLM_API,
        vector_k=VECTOR_K,
        retrieval_min_score=RETRIEVAL_MIN_SCORE,
        grounding_min_overlap=GROUNDING_MIN_OVERLAP,
        enforce_grounded=ENFORCE_GROUNDED,
        include_debug_answer=INCLUDE_DEBUG_ANSWER,
        normalize_embeddings=NORMALIZE_EMBEDDINGS,
        domain_keywords=DOMAIN_KEYWORDS,
        trace_default=TRACE_DEFAULT,
    )
    timings: dict = {}
    trace = TraceState(enabled=bool(getattr(req, 'trace', False) or cfg.trace_default))
    trace.add("request_received", query_len=len(req.user_input))

    # 1. Safety pre
    safety_result = safety_pre(req.user_input, cfg, trace, timings)
    if safety_result.get("block", (not safety_result.get("safe", True))):
        return {"bot_output": "The query was blocked by safety checks.", "citations": [], "moderation": safety_result, **({"trace": trace.events, "trace_id": trace.trace_id} if trace.enabled else {})}

    # 2. Intent
    intent = classify_intent(req.user_input, cfg, trace, timings)
    if intent.out_of_scope:
        resp = {
            "bot_output": "I don't have information about that. Please ask about billing, payment, ECS, security deposit, tariff, meter, reconnection, load, AC installation, wiring or related support topics.",
            "citations": [],
            "intent": intent.__dict__,
            "timings": timings,
        }
        if trace.enabled: resp.update({"trace": trace.events, "trace_id": trace.trace_id})
        return resp

    # 3. Rephrase
    rephrased = rephrase_query(req.user_input, cfg, trace, timings)

    # 4. Retrieve & build prompt
    retr = get_pg_retriever()
    if not retr:
        return {"bot_output": "Retriever unavailable", "citations": []}
    retrieval = retrieve_and_build_prompt(rephrased, cfg, trace, timings, retr)
    if retrieval is None:
        return {"bot_output": "Sorry, I couldn’t find relevant information in the documents.", "citations": []}
    if retrieval.contexts == [] and retrieval.prompt == "":  # low confidence gating path
        return {"bot_output": "I don't know.", "citations": retrieval.citations}

    # 5. Synthesize
    answer = synthesize_answer(retrieval.prompt, cfg, trace, timings)
    if answer is None:
        return {"bot_output": "Synthesis failed. Try again later.", "citations": []}
    original_answer = answer

    # 6. Post safety
    answer, post = safety_post(answer, cfg, trace, timings)

    # 7. Grounding
    answer = enforce_grounding(answer, original_answer, retrieval.contexts, cfg, trace)

    timings['total'] = sum(v for v in timings.values())  # approximate
    resp = build_response(answer, retrieval.citations, intent, retrieval.scores, timings, post, cfg, original_answer)
    if trace.enabled:
        resp['trace'] = trace.events
        resp['trace_id'] = trace.trace_id
        trace.add("response_ready", answer_chars=len(answer))
    return resp


@app.post("/rag_simple")
def rag_simple(req: KMRequest):
    """Ultra-minimal RAG: embed -> retrieve -> simple prompt -> answer.
    Does NOT run safety, intent, or grounding enforcement. For educational use.
    """
    t0 = time.time()
    trace_enabled = bool(getattr(req, 'trace', False))
    trace_id = str(_uuid.uuid4())
    query = req.user_input
    k = VECTOR_K
    timings = {}
    trace_events = [] if trace_enabled else None
    def trace(event, **data):
        if trace_events is not None:
            trace_events.append({"t": round(time.time()-t0,4), "event": event, **data})
        log.info("TRACE[%s] %s %s", trace_id, event, data if data else "")
    # 1. Embed
    vector_emb = []
    try:
        emb_t0 = time.time(); r = requests.post(f"{LLM_API}/embed", json={"text": query}, timeout=15); r.raise_for_status(); vector_emb = r.json().get('embedding', []); timings['embedding'] = time.time() - emb_t0
        trace("embedded_query", dim=len(vector_emb))
    except Exception:
        log.exception("Embedding failed in rag_simple")
        return {"bot_output": "Embedding failed", "citations": []}

    # 2. Retrieve (pgvector preferred)
    contexts = []
    citations = []
    used_backend = 'pgvector'
    retr = get_pg_retriever()
    if retr:
        r_t0 = time.time(); rows = retr.similarity_search(vector_emb, k=k); timings['retrieve'] = time.time() - r_t0
        trace("retrieval_results", backend="pgvector", hits=len(rows), k=k)
        for content, score, source, vid in rows:
            contexts.append(content)
            citations.append({"source": source, "id": vid, "score": score})
    else:
        return {"bot_output": "Retriever unavailable", "citations": []}

    if not contexts:
        return {"bot_output": "I don't know.", "citations": [], "backend": used_backend}

    # 3. Simple prompt
    context_block = "\n---\n".join(contexts)
    trace("assembled_context", total_chars=len(context_block))
    prompt = (
        "You are a concise assistant. Use ONLY the context. If answer absent, say I don't know.\n"\
        f"Context:\n---\n{context_block}\n---\nQuestion: {query}\nAnswer:" )

    # 4. LLM generate
    try:
        g_t0 = time.time(); r = requests.post(f"{LLM_API}/chat", json={"prompt": prompt}, timeout=40); timings['llm_generate'] = time.time() - g_t0
        r.raise_for_status(); answer = r.json().get('answer','').strip()
        trace("llm_answer", answer_chars=len(answer))
    except Exception:
        log.exception("LLM call failed in rag_simple")
        return {"bot_output": "Generation failed", "citations": citations, "backend": used_backend}

    timings['total'] = time.time() - t0
    resp = {"bot_output": answer, "citations": citations, "timings": timings, "backend": used_backend}
    if trace_events is not None:
        resp['trace_id'] = trace_id
        resp['trace'] = trace_events
    return resp


# (health endpoint already defined above)
