import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.routes import events

logger = logging.getLogger("jarvis")

app = FastAPI(title="J.A.R.V.I.S. Backend", version="0.1.0")
app.include_router(events.router)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error")
    return JSONResponse(status_code=500, content={"status": "error", "message": "Internal server error"})

