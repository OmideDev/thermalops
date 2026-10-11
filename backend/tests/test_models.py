
from app.db.base import Base
from app.models import (
    Building,
    Equipment,
    EquipmentStatus,
    EquipmentType,
    Organization,
)


def test_models_register_expected_tables():
    assert set(Base.metadata.tables.keys()) == {
        "organizations",
        "buildings",
        "equipment",
    }


def test_building_belongs_to_organization():
    organization = Organization(name="ThermalOps Demo")
    building = Building(
        name="Office Building",
        organization=organization,
    )

    assert building.organization is organization
    assert building in organization.buildings


def test_equipment_belongs_to_building():
    organization = Organization(name="ThermalOps Demo")
    building = Building(
        name="Office Building",
        organization=organization,
    )
    equipment = Equipment(
        name="AHU-01",
        equipment_type=EquipmentType.AHU,
        building=building,
    )

    assert equipment.building is building
    assert equipment in building.equipment


def test_equipment_enums_have_expected_values():
    assert EquipmentType.AHU.value == "ahu"
    assert EquipmentType.CHILLER.value == "chiller"
    assert EquipmentStatus.RUNNING.value == "running"
    assert EquipmentStatus.FAULT.value == "fault"
    assert EquipmentStatus.OFFLINE.value == "offline"
