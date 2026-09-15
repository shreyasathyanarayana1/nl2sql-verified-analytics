"""Thin OpenAI client (chat + embeddings) using only the standard library.

`enabled()` is False without OPENAI_API_KEY; every caller has a deterministic
fallback, so the platform runs fully offline for development and demos.
"""
import hashlib
import json
import math
import re
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from app.config import OPENAI_API_KEY, OPENAI_MODEL

EMBED_DIM = 512
LOCAL_EMBEDDER = "local-hashed-bow-v1"
OPENAI_EMBEDDER = "text-embedding-3-small"



# Set when OpenAI is configured but unusable (no credits, bad key, network); the app then
# runs fully offline instead of failing request by request.
_disabled_reason: Optional[str] = None


def enabled() -> bool:
    return bool(OPENAI_API_KEY) and _disabled_reason is None


def status() -> Dict[str, Any]:
    if not OPENAI_API_KEY:
        return {"mode": "offline", "reason": "No OPENAI_API_KEY configured"}
    if _disabled_reason:
        return {"mode": "offline", "reason": _disabled_reason}
    return {"mode": "openai", "reason": None}


def _describe(error: Exception) -> str:
    if isinstance(error, urllib.error.HTTPError):
        try:
            detail = json.loads(error.read().decode()).get("error", {})
            code = detail.get("code") or detail.get("type") or ""
            message = detail.get("message", "")
        except Exception:
            code, message = "", ""
        if code in ("credit_balance_exhausted", "insufficient_quota"):
            return "OpenAI account has no credits remaining — add billing credits to enable GPT-4o"
        if error.code == 401:
            return "OpenAI rejected the API key (invalid or revoked)"
        return f"OpenAI error {error.code}{': ' + message[:120] if message else ''}"
    return f"OpenAI unreachable ({type(error).__name__})"


def _disable_if_fatal(error: Exception) -> None:
    """Quota and auth failures won't fix themselves mid-session, so stop calling the API."""
    global _disabled_reason
    if isinstance(error, urllib.error.HTTPError) and error.code in (401, 403, 429):
        _disabled_reason = _describe(error)


def check_available() -> Dict[str, Any]:
    """Startup probe: one tiny embedding call decides whether the OpenAI path is usable."""
    global _disabled_reason
    if not OPENAI_API_KEY:
        return status()
    try:
        _post("embeddings", {"model": OPENAI_EMBEDDER, "input": ["ping"]}, timeout=15)
        _disabled_reason = None
    except Exception as e:  # noqa: BLE001 - any failure means offline mode
        _disabled_reason = _describe(e)
    return status()


def embedder_name() -> str:
    return OPENAI_EMBEDDER if enabled() else LOCAL_EMBEDDER


def _post(path: str, payload: Dict[str, Any], timeout: int = 30) -> Dict[str, Any]:
    req = urllib.request.Request(
        f"https://api.openai.com/v1/{path}",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def chat_json(system: str, user: str) -> Dict[str, Any]:
    try:
        data = _post("chat/completions", {
            "model": OPENAI_MODEL,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        })
    except Exception as e:
        _disable_if_fatal(e)
        raise
    return json.loads(data["choices"][0]["message"]["content"])


def _tokens(text: str) -> List[str]:
    out = []
    for tok in re.findall(r"[a-z0-9_]+", text.lower()):
        parts = [tok] + (tok.split("_") if "_" in tok else [])
        for p in parts:
            out.append(p)
            if len(p) > 4 and p.endswith("s"):
                out.append(p[:-1])
    return out


def _local_embed(text: str) -> List[float]:
    vec = [0.0] * EMBED_DIM
    for tok in _tokens(text):
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        vec[h % EMBED_DIM] += 1.0 if (h >> 12) % 2 else -1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def embed(texts: List[str]) -> List[List[float]]:
    if enabled():
        try:
            data = _post("embeddings", {"model": OPENAI_EMBEDDER, "input": texts})
            return [item["embedding"] for item in data["data"]]
        except Exception as e:
            _disable_if_fatal(e)
            raise
    return [_local_embed(t) for t in texts]


def cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)
