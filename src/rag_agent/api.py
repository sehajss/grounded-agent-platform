from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from sqlalchemy import text

from rag_agent.auth_routes import router as auth_router
from rag_agent.database import close_database, engine
from rag_agent.embeddings import Embedder
from rag_agent.ingestion import IngestionService
from rag_agent.query_service import QueryService
from rag_agent.rag_routes import router as rag_router
from rag_agent.workspace_routes import router as workspace_router


HTTP_REQUESTS = Counter(
    "grounded_agent_http_requests_total",
    "HTTP requests served.",
    ("method", "route", "status"),
)
HTTP_LATENCY = Histogram(
    "grounded_agent_http_request_duration_seconds",
    "HTTP request duration in seconds.",
    ("method", "route"),
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        embedder = await run_in_threadpool(Embedder)
        app.state.embedder = embedder
        app.state.ingestion_service = IngestionService(embedder)
        app.state.query_service = QueryService(embedder)
        yield
    finally:
        await close_database()


app = FastAPI(
    title="Grounded Agent Platform",
    version="0.1.0",
    description="Workspace-scoped retrieval and policy-controlled developer-agent APIs.",
    lifespan=lifespan,
)

app.include_router(auth_router)
app.include_router(workspace_router)
app.include_router(rag_router)


@app.middleware("http")
async def observe_requests(request: Request, call_next):
    request_id = uuid.uuid4().hex
    request.state.request_id = request_id
    started = time.perf_counter()
    status = 500

    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        route = request.scope.get("route")
        route_path = getattr(route, "path", "unmatched")
        HTTP_REQUESTS.labels(request.method, route_path, str(status)).inc()
        HTTP_LATENCY.labels(request.method, route_path).observe(
            time.perf_counter() - started
        )


@app.get("/health/live", tags=["health"])
async def live() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/ready", tags=["health"])
async def ready() -> dict[str, str]:
    try:
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
            result = await connection.execute(
                text(
                    "SELECT EXISTS "
                    "(SELECT 1 FROM pg_extension WHERE extname = 'vector')"
                )
            )
            vector_enabled = result.scalar_one()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Database is unavailable.",
        ) from exc

    if not vector_enabled:
        raise HTTPException(
            status_code=503,
            detail="Database is reachable, but pgvector is not enabled yet.",
        )

    return {"status": "ready", "database": "ok", "pgvector": "enabled"}


@app.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )