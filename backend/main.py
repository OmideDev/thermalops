from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.assistant import router as assistant_router
from app.api.v1.buildings import router as buildings_router
from app.api.v1.equipment import router as equipment_router
from app.api.v1.health import router as health_router
from app.core.config import settings
from app.services.assistant import try_create_assistant_from_settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Optional experimental assistant init.

    Failures leave ``app.state.assistant = None`` so Equipment/Building APIs
    still start. Disabled by default (assistant_enabled=false).
    """
    app.state.assistant = try_create_assistant_from_settings(settings)
    if app.state.assistant is None:
        logger.info("Assistant not loaded (disabled or unavailable)")
    try:
        yield
    finally:
        service = getattr(app.state, "assistant", None)
        if service is not None:
            service.close()
        app.state.assistant = None


app = FastAPI(
    title=settings.app_name,
    description="HVAC Operations and Monitoring Platform",
    version=settings.app_version,
    lifespan=lifespan,
)

# Allows the Vite frontend (and same-origin proxies) to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(buildings_router)
app.include_router(equipment_router)
app.include_router(assistant_router)
