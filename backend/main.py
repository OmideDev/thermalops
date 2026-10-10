from fastapi import FastAPI

from app.api.v1.health import router as health_router
from app.core.config import APP_NAME, APP_VERSION


app = FastAPI(
    title=APP_NAME,
    description="HVAC Operations and Monitoring Platform",
    version=APP_VERSION,
)

app.include_router(health_router)
