from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.buildings import router as buildings_router
from app.api.v1.equipment import router as equipment_router
from app.api.v1.health import router as health_router
from app.core.config import settings

app = FastAPI(
    title=settings.app_name,
    description="HVAC Operations and Monitoring Platform",
    version=settings.app_version,
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
