from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models import Building
from app.schemas.building import BuildingResponse

router = APIRouter(
    prefix="/api/v1/buildings",
    tags=["Buildings"],
)


@router.get("/", response_model=list[BuildingResponse])
def list_buildings(db: Session = Depends(get_db)) -> list[Building]:
    statement = select(Building).order_by(
        Building.name.asc(),
        Building.id.asc(),
    )
    return list(db.scalars(statement).all())
