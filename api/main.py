"""FastAPI app: CORS, chat routes, static UI, health, lifespan hooks."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.chat import orchestrator, router as chat_router
from config import PROJECT_ROOT


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("api.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("SAATHI API starting up")
    try:
        yield
    finally:
        logger.info("SAATHI API shutting down — releasing orchestrator")
        try:
            await orchestrator.close()
        except Exception as e:
            logger.warning("Orchestrator close failed: %s", e)


app = FastAPI(
    title="SAATHI API",
    description="Peer support chatbot for Indian college students",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Dev only — restrict in production via env
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)

# Serve the UI bundle if `ui/` exists. The mount is conditional so the API
# still boots cleanly in headless / API-only deployments.
ui_path = PROJECT_ROOT / "ui"
if ui_path.exists() and ui_path.is_dir():
    app.mount("/ui", StaticFiles(directory=str(ui_path), html=True), name="ui")
    logger.info("UI mounted at /ui from %s", ui_path)
else:
    logger.info("No ui/ directory found — skipping static mount")


@app.get("/health")
async def health() -> dict:
    """Liveness probe. Used by load balancers and the smoke test."""
    return {"status": "ok", "service": "saathi"}
