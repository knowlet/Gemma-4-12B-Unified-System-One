"""Local and Modal use the same application and validation boundary."""

from __future__ import annotations

import secrets
import threading

from .contracts import DecisionRequest
from .errors import (
    BackendResponseError,
    BackendTimeoutError,
    BackendTransportError,
    RequestValidationError,
)


def create_app(backend, *, api_key=None):
    from fastapi import Depends, FastAPI, Header, HTTPException

    app = FastAPI(title="Gemma Unified System One", version="0.1.0")
    lock = threading.Lock()

    def authorize(authorization: str | None = Header(default=None)):
        if api_key and not secrets.compare_digest(authorization or "", f"Bearer {api_key}"):
            raise HTTPException(status_code=401, detail="Invalid bearer token")

    @app.get("/healthz")
    def health():
        return {"status": "ok", "model": backend.name}

    @app.post("/decide", dependencies=[Depends(authorize)])
    @app.post("/v1/systemone", dependencies=[Depends(authorize)])
    def decide(request: DecisionRequest):
        from .backends import normalize_response

        try:
            with lock:
                return normalize_response(request, backend.predict(request))
        except RequestValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except BackendTimeoutError as exc:
            raise HTTPException(status_code=504, detail=str(exc)) from exc
        except (BackendResponseError, BackendTransportError) as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    return app
