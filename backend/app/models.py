from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum

from sqlalchemy import (
    DateTime,
    Enum as SAEnum,
    ForeignKey,
    String,
    Uuid,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

class EquipmentType(str, Enum):
    AHU = "ahu"
    CHILLER = "chiller"
    BOILER = "boiler"
    PUMP = "pump"
    COOLING_TOWER = "cooling_tower"
    OTHER = "other"


class EquipmentStatus(str, Enum):
    RUNNING = "running"
    STOPPED = "stopped"
    FAULT = "fault"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"


def enum_values(enum_class):
    return [item.value for item in enum_class]


class Organization(Base):
    __tablename__ = "organizations"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid.uuid4,
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    buildings: Mapped[list[Building]] = relationship(
        back_populates="organization",
    )


class Building(Base):
    __tablename__ = "buildings"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid.uuid4,
    )
    organization_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("organizations.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    location: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )
    timezone: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        default="UTC",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    organization: Mapped[Organization] = relationship(
        back_populates="buildings",
    )
    equipment: Mapped[list[Equipment]] = relationship(
        back_populates="building",
    )


class Equipment(Base):
    __tablename__ = "equipment"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid.uuid4,
    )
    building_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("buildings.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)

    equipment_type: Mapped[EquipmentType] = mapped_column(
        SAEnum(
            EquipmentType,
            name="equipment_type_values",
            native_enum=False,
            create_constraint=True,
            values_callable=enum_values,
        ),
        nullable=False,
    )
    status: Mapped[EquipmentStatus] = mapped_column(
        SAEnum(
            EquipmentStatus,
            name="equipment_status_values",
            native_enum=False,
            create_constraint=True,
            values_callable=enum_values,
        ),
        nullable=False,
        default=EquipmentStatus.OFFLINE,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    building: Mapped[Building] = relationship(
        back_populates="equipment",
    )


# Register RAG / admin-session tables on the same declarative Base metadata.
from app import models_rag as _models_rag  # noqa: E402,F401
