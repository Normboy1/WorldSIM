"""SIMLAB FastAPI application entry point.

Run with:
    uvicorn simlab.api.fastapi_server.main:app --host 0.0.0.0 --port 8000
    # or
    simlab-api
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from simlab.api.fastapi_server.cors import cors_allow_credentials, cors_origins
from simlab.api.fastapi_server.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("simlab").info("SIMLAB API v0.1.0 started.")
    yield


app = FastAPI(
    title="SIMLAB API",
    description=(
        "AI-powered centralized scientific simulation platform. "
        "Supports math, physics, chemistry, and materials science simulations."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

configured_cors_origins = cors_origins()

# Local browser/notebook access by default. Override SIMLAB_CORS_ORIGINS
# with a comma-separated allowlist for deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=configured_cors_origins,
    allow_credentials=cors_allow_credentials(configured_cors_origins),
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="")


def start():
    """Entry point for simlab-api CLI script."""
    uvicorn.run(
        "simlab.api.fastapi_server.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    start()
