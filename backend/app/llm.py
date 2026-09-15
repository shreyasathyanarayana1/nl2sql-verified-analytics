"""Backward-compatible local math helpers.

The production pipeline in this repository does not use a hosted LLM.
"""
from typing import Any, Dict, List
import math

def enabled() -> bool:
    return False

def status() -> Dict[str, Any]:
    return {"mode": "local", "reason": "Hosted LLM disabled; local ML models are used."}

def embedder_name() -> str:
    return "local-tfidf-v1"

def cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    return dot/(na*nb) if na and nb else 0.0
