import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.routes import api_keys, auth, events, github, projects

logger = logging.getLogger("jarvis")

app = FastAPI(title="J.A.R.V.I.S. Backend", version="0.2.0")

# CORS for the future React frontend only (credentials require an explicit origin).
# Telemetry SDKs run server-side and send a Bearer key, so they need no CORS.
try:
    _frontend_origin = get_settings().frontend_origin
except RuntimeError as exc:  # misconfigured env: keep /health and /docs alive
    logger.error("%s", exc)
    _frontend_origin = None

if _frontend_origin:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[_frontend_origin],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization"],
    )

app.include_router(events.router)
app.include_router(auth.router)
app.include_router(api_keys.router)
app.include_router(github.router)
app.include_router(projects.router)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error")
    return JSONResponse(status_code=500, content={"status": "error", "message": "Internal server error"})
