"""Minimal HS256 JWT auth with two demo personas (Administrator, Insights User)."""
import base64
import hashlib
import hmac
import json
import time
from typing import Any, Dict

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import JWT_SECRET, JWT_TTL_SECONDS

# Demo accounts. Replace with a real user table / OAuth provider before deployment.
USERS = {
    "admin": {"password": "admin123", "role": "admin", "display_name": "Administrator"},
    "analyst": {"password": "analyst123", "role": "insights_user", "display_name": "Insights User"},
}

bearer = HTTPBearer(auto_error=False)


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _unb64(data: str) -> bytes:
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))


def issue_token(username: str) -> str:
    user = USERS[username]
    header = _b64(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    payload = _b64(json.dumps({"sub": username, "role": user["role"], "exp": int(time.time()) + JWT_TTL_SECONDS}).encode())
    sig = _b64(hmac.new(JWT_SECRET.encode(), f"{header}.{payload}".encode(), hashlib.sha256).digest())
    return f"{header}.{payload}.{sig}"


def decode_token(token: str) -> Dict[str, Any]:
    try:
        header, payload, sig = token.split(".")
        expected = _b64(hmac.new(JWT_SECRET.encode(), f"{header}.{payload}".encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(sig, expected):
            raise ValueError("bad signature")
        claims = json.loads(_unb64(payload))
        if claims["exp"] < time.time() or claims["sub"] not in USERS:
            raise ValueError("expired")
        return claims
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")


def authenticate(username: str, password: str) -> Dict[str, Any]:
    user = USERS.get(username)
    if not user or not hmac.compare_digest(user["password"], password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password")
    return {"username": username, "role": user["role"], "display_name": user["display_name"]}


def current_user(creds: HTTPAuthorizationCredentials = Depends(bearer)) -> Dict[str, Any]:
    if creds is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    claims = decode_token(creds.credentials)
    user = USERS[claims["sub"]]
    return {"username": claims["sub"], "role": user["role"], "display_name": user["display_name"]}


def require_admin(user: Dict[str, Any] = Depends(current_user)) -> Dict[str, Any]:
    if user["role"] != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Administrator role required")
    return user
