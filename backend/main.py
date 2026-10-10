from fastapi import FastAPI

app = FastAPI(
    title="ThermalOps API",
    description="HVAC Operations and Monitoring Platform",
    version="0.1.0",
)


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "thermalops-api",
        "version": "0.1.0",
    }
