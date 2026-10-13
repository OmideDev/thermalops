from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class BuildingResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    organization_id: UUID
    name: str
    location: str | None
    timezone: str
    created_at: datetime
