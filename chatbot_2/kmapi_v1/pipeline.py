
from __future__ import annotations

from dataclasses import dataclass, field
import time, requests, re, uuid as _uuid, math, json as _json
from typing import Any, Dict, List, Optional, Tuple

from prompts.prompt_loader import format_prompt, PromptNotFoundError


@dataclass
class Config:
    llm_api: str
    vector_k: int
    retrieval_min_score: float
    grounding_min_overlap: float
    enforce_grounded: bool
    include_debug_answer: bool
    normalize_embeddings: bool
    domain_keywords: set[str]
    trace_default: bool = False


@dataclass
class TraceState:
    enabled: bool
    trace_id: str = field(default_factory=lambda: str(_uuid.uuid4()))
    start: float = field(default_factory=time.time)
    events: List[Dict[str, Any]] = field(default_factory=list)

    def add(self, event: str, **data):
        if not self.enabled:
            return
        self.events.append({
            "t": round(time.time() - self.start, 4),
            "event": event,
            **data,
        })


@dataclass
class IntentResult:
    intent: str = "unknown"
    out_of_scope: bool = True
    reason: str = "classifier_failed"


@dataclass
class RetrievalResult:
    contexts: List[str]
    citations: List[Dict[str, Any]]
    scores: List[float]
    prompt: str


def embed(text: str, cfg: Config, trace: TraceState, timings: Dict[str, float], label: str = "embed") -> List[float]:
    emb: List[float] = []
    t0 = time.time()
    try:
        r = requests.post(f"{cfg.llm_api}/embed", json={"text": text}, timeout=15)
        r.raise_for_status()
        emb = r.json().get("embedding", [])
        if cfg.normalize_embeddings and emb:
            s = sum(e * e for e in emb) or 1.0
            l2 = math.sqrt(s)
            emb = [e / l2 for e in emb]
    except Exception:
        # caller decides on failure handling
        pass
    timings[label] = time.time() - t0
    trace.add("embedded_query", dim=len(emb), label=label)
    return emb


def safety_pre(user_input: str, cfg: Config, trace: TraceState, timings: Dict[str, float]) -> Dict[str, Any]:
    t0 = time.time()
    try:
        resp = requests.post(f"{cfg.llm_api}/inspect", json={"text": user_input}, timeout=8)
        data = resp.json()
        timings['safety_inspect'] = time.time() - t0
        trace.add("safety_pre", **{k: data.get(k) for k in ("policy", "block", "categories")})
        return data
    except Exception:
        timings['safety_inspect'] = time.time() - t0
        return {"block": True, "safe": False, "error": "inspect_failed"}


def classify_intent(user_input: str, cfg: Config, trace: TraceState, timings: Dict[str, float]) -> IntentResult:
    res = IntentResult()
    t0 = time.time()
    try:
        try:
            tmpl = format_prompt("intent_classify_v1", question=user_input)
            sys_part = tmpl.get('system') or ''
            user_part = tmpl.get('user') or ''
            prompt = f"{sys_part}\n\n{user_part}" if sys_part else user_part
        except PromptNotFoundError:
            prompt = (
                "You output ONLY minified JSON:{\"intent\":string,\"out_of_scope\":boolean,\"reason\":string}. "
                "If query not about billing, online/advance payment, ECS, security deposit, tariff/billing, name/address change, "
                "reconnection, load enhancement, AC installation, meter, wiring say out_of_scope true.\nQuery: " + user_input
            )
        r = requests.post(f"{cfg.llm_api}/chat", json={"prompt": prompt}, timeout=12)
        r.raise_for_status()
        raw = r.json().get('answer', '')
        parsed = None
        try:
            parsed = _json.loads(raw)
        except Exception:
            import re as _re
            m = _re.search(r"\{.*\}", raw, _re.DOTALL)
            if m:
                try:
                    parsed = _json.loads(m.group(0))
                except Exception:
                    pass
        if isinstance(parsed, dict):
            res.intent = parsed.get('intent', res.intent)
            res.out_of_scope = bool(parsed.get('out_of_scope', res.out_of_scope))
            res.reason = parsed.get('reason', res.reason)
    except Exception:
        pass
    timings['intent_classify'] = time.time() - t0
    trace.add("intent_result", intent=res.intent, out_of_scope=res.out_of_scope, reason=res.reason)
    return res


def rephrase_query(user_input: str, cfg: Config, trace: TraceState, timings: Dict[str, float]) -> str:
    t0 = time.time()
    rephrased = user_input
    try:
        tmpl = format_prompt("rephrase_v1", question=user_input)
        synth_query = tmpl['user']
        synth_context = tmpl['system']
    except (PromptNotFoundError, KeyError):
        synth_query = 'Return STRICT JSON only: {"intent": string, "rephrased": string suitable for vector search}.'
        synth_context = f"User query: {user_input}"
    try:
        r = requests.post(f"{cfg.llm_api}/synthesize", json={"query": synth_query, "context": synth_context}, timeout=15)
        r.raise_for_status()
        raw = r.json().get("answer", "")
        if raw:
            try:
                obj = _json.loads(raw)
                rephrased = obj.get("rephrased") or obj.get("intent") or user_input
            except Exception:
                if len(raw) < 160:
                    rephrased = raw
    except Exception:
        pass
    timings['rephrase'] = time.time() - t0
    trace.add("rephrase_result", rephrased=rephrased[:160])
    return rephrased


COMMON_TYPO_MAP = {
    "reciept": "receipt",
    "recipt": "receipt",
    "paymnt": "payment",
    "recieved": "received",
    "receieved": "received",
    "adress": "address",
}

ACRONYM_EXPANSIONS = {"ecs": ["electronic clearing service"]}


def _apply_typos(q: str) -> str:
    lower = q.lower()
    for wrong, right in COMMON_TYPO_MAP.items():
        if wrong in lower:
            lower = lower.replace(wrong, right)
    return lower


def _expand_variants(q: str) -> List[str]:
    base = _apply_typos(q)
    variants = [base]
    lower = base.lower()
    for acro, exps in ACRONYM_EXPANSIONS.items():
        if acro in lower.split():
            for exp in exps:
                if exp not in lower:
                    variants.append(exp)
                    variants.append(f"{base} {exp}")
    seen = set(); ordered = []
    for v in variants:
        if v not in seen:
            seen.add(v); ordered.append(v)
    return ordered


def retrieve_and_build_prompt(rephrased: str, cfg: Config, trace: TraceState, timings: Dict[str, float], retriever) -> RetrievalResult | None:
    t0 = time.time()
    variants = _expand_variants(rephrased)
    all_rows: List[Tuple[str, Dict[str, Any]]] = []
    for variant in variants:
        emb = embed(variant, cfg, trace, timings, label=f"embed_variant")
        if not emb:
            continue
        try:
            rt = time.time(); rows = retriever.similarity_search(emb, k=cfg.vector_k); timings['retrieve'] = time.time() - rt
        except Exception:
            rows = []
        if rows:
            trace.add("retrieval_results", backend="pgvector", hits=len(rows), k=cfg.vector_k)
            for content, score, source, vid in rows:
                all_rows.append((content, {"source": source, "id": vid, "score": score}))
        if len(all_rows) >= cfg.vector_k:
            break
    # Merge unique citations
    contexts, citations = [], []
    seen = set()
    for content, cit in all_rows:
        key = (cit['source'], cit['id'])
        if key in seen:
            continue
        seen.add(key)
        contexts.append(content)
        citations.append(cit)
        if len(contexts) >= cfg.vector_k:
            break
    timings['retrieval_total'] = time.time() - t0
    if not contexts:
        trace.add("retrieval_empty", variants=variants)
        return None
    ctx_join = "\n---\n".join(contexts)
    try:
        answer_tmpl = format_prompt("rag_answer_v1", context=ctx_join, user_query=rephrased)
        prompt = f"{answer_tmpl['system']}\n\n{answer_tmpl['user']}"
    except Exception:
        prompt = (
            "Answer ONLY from the context. If not present, say you don't know.\n"
            f"Context:\n---\n{ctx_join}\n---\n\nQ: {rephrased}\nA:"
        )
    scores = [c.get('score') for c in citations if isinstance(c.get('score'), (int, float))]
    # Confidence heuristic (mirrors original logic)
    try:
        best = max(scores) if scores else None
        combined_len = sum(len(c) for c in contexts)
        context_lower = " ".join(contexts).lower()
        has_domain = any(k in context_lower for k in cfg.domain_keywords)
        if best is not None and best < cfg.retrieval_min_score and combined_len < 120 and not has_domain:
            return RetrievalResult(contexts=[], citations=citations, scores=scores, prompt="")
        trace.add("similarity_stats", best=best, context_chars=combined_len)
    except Exception:
        pass
    return RetrievalResult(contexts=contexts, citations=citations, scores=scores, prompt=prompt)


def synthesize_answer(prompt: str, cfg: Config, trace: TraceState, timings: Dict[str, float]) -> str | None:
    t0 = time.time()
    try:
        r = requests.post(f"{cfg.llm_api}/chat", json={"prompt": prompt}, timeout=40)
        r.raise_for_status()
        answer = r.json().get("answer", "").strip()
        timings['llm_generate'] = time.time() - t0
        trace.add("llm_answer", chars=len(answer))
        return answer
    except Exception:
        timings['llm_generate'] = time.time() - t0
        return None


def safety_post(answer: str, cfg: Config, trace: TraceState, timings: Dict[str, float]) -> tuple[str, Optional[Dict[str, Any]]]:
    t0 = time.time(); post = None
    try:
        post = requests.post(f"{cfg.llm_api}/inspect", json={"text": answer}, timeout=8).json()
        timings['post_inspect'] = time.time() - t0
        block_gen = post.get("block", (not post.get("safe", True)))
        trace.add("safety_post", block=block_gen, policy=post.get('policy'))
        if block_gen:
            return "The generated content was moderated and withheld.", post
    except Exception:
        timings['post_inspect'] = time.time() - t0
    return answer, post


def enforce_grounding(answer: str, original_answer: str, contexts: List[str], cfg: Config, trace: TraceState) -> str:
    if not cfg.enforce_grounded or answer.startswith("The generated content was moderated"):
        return answer
    try:
        joined_context = "\n".join(contexts).lower()
        tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", original_answer.lower())
        domain_in_answer = any(k in original_answer.lower() for k in cfg.domain_keywords)
        if tokens:
            overlapping = [tok for tok in tokens if tok in joined_context]
            overlap_ratio = (len(overlapping) / max(len(tokens), 1)) if tokens else 0.0
            short_answer = len(original_answer.strip()) < 40
            if not domain_in_answer and overlap_ratio < cfg.grounding_min_overlap and not short_answer:
                answer = "I don't know."
            trace.add("grounding", ratio=round(overlap_ratio, 3), domain_in_answer=domain_in_answer)
    except Exception:
        pass
    return answer


def build_response(answer: str, citations: List[Dict[str, Any]], intent: IntentResult, scores: List[float], timings: Dict[str, float], post: Optional[Dict[str, Any]] | None, cfg: Config, original_answer: str) -> Dict[str, Any]:
    resp = {
        "bot_output": answer,
        "citations": citations,
        "timings": timings,
        "intent": intent.__dict__,
        "retrieval_stats": {"scores_present": any(isinstance(s, (int, float)) for s in scores), "backend": "pgvector"},
    }
    if post is not None:
        resp['moderation_post'] = post
    if cfg.include_debug_answer and answer != original_answer:
        resp['debug_raw_answer'] = original_answer
    return resp
