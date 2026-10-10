# ThermalOps

ThermalOps is an HVAC Operations and Monitoring Platform designed to monitor equipment, track operational status, and support maintenance workflows.

The platform is being developed with an API-first architecture, with room for future AI-powered diagnostics and operational assistance.

## Current Technology Stack

- **Backend:** Python, FastAPI
- **Data validation:** Pydantic
- **API documentation:** OpenAPI / Swagger UI
- **Planned database:** PostgreSQL
- **Planned frontend:** React, TypeScript

## Current Features

- Health-check endpoint
- Automatic API documentation
- Initial backend foundation

## Run the Backend Locally

Open PowerShell and navigate to the backend directory:

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
fastapi dev main.py
```

The API will be available at:

- API health check: http://127.0.0.1:8000/health
- Interactive API documentation: http://127.0.0.1:8000/docs

## Development Roadmap

1. Establish a clean backend structure.
2. Design HVAC equipment and operational-status models.
3. Integrate PostgreSQL.
4. Add equipment monitoring and maintenance workflows.
5. Introduce AI capabilities incrementally.

This project is under active development.