from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.models import EquipmentStatus, EquipmentType

class EquipmentCreate(BaseModel):
    building_id: UUID
    name: str = Field(min_length=1, max_length=200)
    equipment_type: EquipmentType
    status: EquipmentStatus = EquipmentStatus.OFFLINE

class EquipmentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    building_id: UUID
    name: str
    equipment_type: EquipmentType
    status: EquipmentStatus
    created_at: datetime


class EquipmentUpdate(BaseModel):
    name: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
    )
    equipment_type: EquipmentType | None = None
    status: EquipmentStatus | None = None



    @model_validator(mode="after")
    def reject_explicit_nulls(self):
        for field_name in self.model_fields_set:
            if getattr(self, field_name) is None:
                raise ValueError(
                    f"{field_name} cannot be null"
                )

        return self
