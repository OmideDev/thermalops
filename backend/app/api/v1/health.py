from fastapi import APIRouter

from app.core.config import APP_VERSION


router = APIRouter()


@router.get("/health", tags=["Health"])
def health_check():
    return {
        "status": "healthy",
        "service": "thermalops-api",
        "version": APP_VERSION,
    }