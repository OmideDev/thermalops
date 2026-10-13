from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models import Building, Equipment, EquipmentStatus
from app.schemas.equipment import (
    EquipmentCreate,
    EquipmentResponse,
    EquipmentUpdate,
)
router = APIRouter(
    prefix="/api/v1/equipment",
    tags=["Equipment"],
)


@router.get("/", response_model=list[EquipmentResponse])
def list_equipment(
    building_id: UUID | None = None,
    status: EquipmentStatus | None = None,
    db: Session = Depends(get_db),
) -> list[Equipment]:
    statement = select(Equipment)

    if building_id is not None:
        statement = statement.where(
            Equipment.building_id == building_id
        )

    if status is not None:
        statement = statement.where(
            Equipment.status == status
        )

    statement = statement.order_by(
        Equipment.created_at.desc(),
        Equipment.id.desc(),
    )

    return list(db.scalars(statement).all())

@router.get("/{equipment_id}", response_model=EquipmentResponse)
def get_equipment(
    equipment_id: UUID,
    db: Session = Depends(get_db),
) -> Equipment:
    equipment = db.get(Equipment, equipment_id)

    if equipment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Equipment not found",
        )

    return equipment


@router.post(
    "/",
    response_model=EquipmentResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_equipment(
    equipment_data: EquipmentCreate,
    db: Session = Depends(get_db),
) -> Equipment:
    building = db.get(Building, equipment_data.building_id)

    if building is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Building not found",
        )

    equipment = Equipment(**equipment_data.model_dump())

    try:
        db.add(equipment)
        db.commit()
        db.refresh(equipment)
    except Exception:
        db.rollback()
        raise

    return equipment


@router.patch(
    "/{equipment_id}",
    response_model=EquipmentResponse,
)
def update_equipment(
    equipment_id: UUID,
    equipment_data: EquipmentUpdate,
    db: Session = Depends(get_db),
) -> Equipment:
    equipment = db.get(Equipment, equipment_id)

    if equipment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Equipment not found",
        )

    update_data = equipment_data.model_dump(exclude_unset=True)

    if not update_data:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="At least one field must be provided",
        )

    for field, value in update_data.items():
        setattr(equipment, field, value)

    try:
        db.commit()
        db.refresh(equipment)
    except Exception:
        db.rollback()
        raise

    return equipment
