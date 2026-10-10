from fastapi import FastAPI

from app.api.v1.health import router as health_router
from app.core.config import settings

app = FastAPI(
    title=settings.app_name,
    description="HVAC Operations and Monitoring Platform",
    version=settings.app_version,)

app.include_router(health_router)
