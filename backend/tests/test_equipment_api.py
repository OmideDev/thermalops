from unittest.mock import MagicMock
from app.api.v1.equipment import update_equipment
from app.models import Equipment
from uuid import uuid4
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.db.session import get_db
from app.models import Building, Organization
from main import app

from app.schemas.equipment import EquipmentUpdate

from app.api.v1.equipment import create_equipment
from app.schemas.equipment import EquipmentCreate
@pytest.fixture
def session_factory():
    test_engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    TestingSessionLocal = sessionmaker(
        bind=test_engine,
        autoflush=False,
        autocommit=False,
    )

    Base.metadata.create_all(bind=test_engine)

    yield TestingSessionLocal

    Base.metadata.drop_all(bind=test_engine)
    test_engine.dispose()


@pytest.fixture
def client(session_factory):
    def override_get_db():
        db = session_factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.clear()


def create_test_building(session_factory):
    with session_factory() as db:
        organization = Organization(name="Test Organization")
        db.add(organization)
        db.flush()

        building = Building(
            organization_id=organization.id,
            name="Test Building",
            location="Test Location",
            timezone="UTC",
        )
        db.add(building)
        db.commit()
        db.refresh(building)

        return building.id


def test_list_equipment_returns_empty_list(client):
    response = client.get("/api/v1/equipment/")

    assert response.status_code == 200
    assert response.json() == []


def test_create_equipment(client, session_factory):
    building_id = create_test_building(session_factory)

    payload = {
        "building_id": str(building_id),
        "name": "AHU-01",
        "equipment_type": "ahu",
        "status": "offline",
    }

    response = client.post(
        "/api/v1/equipment/",
        json=payload,
    )

    assert response.status_code == 201

    data = response.json()

    assert data["building_id"] == str(building_id)
    assert data["name"] == "AHU-01"
    assert data["equipment_type"] == "ahu"
    assert data["status"] == "offline"
    assert "id" in data
    assert "created_at" in data


def test_create_equipment_returns_404_for_unknown_building(client):
    payload = {
        "building_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
        "name": "AHU-UNKNOWN",
        "equipment_type": "ahu",
        "status": "offline",
    }

    response = client.post(
        "/api/v1/equipment/",
        json=payload,
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Building not found",
    }


def test_create_equipment_rejects_invalid_equipment_type(
    client,
    session_factory,
):
    building_id = create_test_building(session_factory)

    payload = {
        "building_id": str(building_id),
        "name": "AHU-INVALID",
        "equipment_type": "invalid_type",
        "status": "offline",
    }

    response = client.post(
        "/api/v1/equipment/",
        json=payload,
    )

    assert response.status_code == 422


def test_get_equipment_by_id(client, session_factory):
    building_id = create_test_building(session_factory)

    payload = {
        "building_id": str(building_id),
        "name": "AHU-01",
        "equipment_type": "ahu",
        "status": "offline",
    }

    create_response = client.post(
        "/api/v1/equipment/",
        json=payload,
    )

    equipment_id = create_response.json()["id"]

    response = client.get(
        f"/api/v1/equipment/{equipment_id}"
    )

    assert response.status_code == 200
    assert response.json()["id"] == equipment_id
    assert response.json()["name"] == "AHU-01"
    assert response.json()["equipment_type"] == "ahu"


def test_get_equipment_returns_404_for_unknown_id(client):
    unknown_id = uuid4()

    response = client.get(
        f"/api/v1/equipment/{unknown_id}"
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Equipment not found",
    }


def test_equipment_update_only_includes_provided_fields():
    equipment_data = EquipmentUpdate(status="running")

    assert equipment_data.model_dump(exclude_unset=True) == {
        "status": "running",
    }


def test_update_equipment_changes_only_provided_fields(
    client,
    session_factory,
):
    building_id = create_test_building(session_factory)

    create_response = client.post(
        "/api/v1/equipment/",
        json={
            "building_id": str(building_id),
            "name": "AHU-01",
            "equipment_type": "ahu",
            "status": "offline",
        },
    )

    equipment_id = create_response.json()["id"]

    response = client.patch(
        f"/api/v1/equipment/{equipment_id}",
        json={"status": "running"},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["id"] == equipment_id
    assert data["name"] == "AHU-01"
    assert data["equipment_type"] == "ahu"
    assert data["status"] == "running"


def test_update_equipment_rejects_null_status(client, session_factory):
    building_id = create_test_building(session_factory)

    create_response = client.post(
        "/api/v1/equipment/",
        json={
            "building_id": str(building_id),
            "name": "AHU-01",
            "equipment_type": "ahu",
            "status": "offline",
        },
    )

    equipment_id = create_response.json()["id"]

    response = client.patch(
        f"/api/v1/equipment/{equipment_id}",
        json={"status": None},
    )

    assert response.status_code == 422
    assert response.json()["detail"][0]["type"] == "value_error"


def test_update_equipment_returns_404_for_unknown_id(client):
    unknown_id = uuid4()

    response = client.patch(
        f"/api/v1/equipment/{unknown_id}",
        json={"status": "running"},
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Equipment not found",
    }


def test_update_equipment_rolls_back_when_commit_fails():
    equipment_id = uuid4()

    equipment = Equipment(
        id=equipment_id,
        building_id=uuid4(),
        name="AHU-01",
        equipment_type="ahu",
        status="offline",
    )

    db = MagicMock()
    db.get.return_value = equipment
    db.commit.side_effect = RuntimeError("Database commit failed")

    equipment_data = EquipmentUpdate(status="running")

    with pytest.raises(
        RuntimeError,
        match="Database commit failed",
    ):
        update_equipment(
            equipment_id=equipment_id,
            equipment_data=equipment_data,
            db=db,
        )

    db.commit.assert_called_once()
    db.rollback.assert_called_once()
    db.refresh.assert_not_called()
def test_create_equipment_rolls_back_when_commit_fails():
    building_id = uuid4()

    db = MagicMock()
    db.get.return_value = MagicMock()
    db.commit.side_effect = RuntimeError("Database commit failed")

    equipment_data = EquipmentCreate(
        building_id=building_id,
        name="AHU-01",
        equipment_type="ahu",
        status="offline",
    )

    with pytest.raises(
        RuntimeError,
        match="Database commit failed",
    ):
        create_equipment(
            equipment_data=equipment_data,
            db=db,
        )

    db.add.assert_called_once()
    db.commit.assert_called_once()
    db.rollback.assert_called_once()
    db.refresh.assert_not_called()



def test_update_equipment_rejects_empty_payload(
    client,
    session_factory,
):
    building_id = create_test_building(session_factory)

    create_response = client.post(
        "/api/v1/equipment/",
        json={
            "building_id": str(building_id),
            "name": "AHU-01",
            "equipment_type": "ahu",
            "status": "offline",
        },
    )

    assert create_response.status_code == 201
    equipment_id = create_response.json()["id"]

    response = client.patch(
        f"/api/v1/equipment/{equipment_id}",
        json={},
    )

    assert response.status_code == 422
    assert response.json() == {
        "detail": "At least one field must be provided",
    }

def test_list_equipment_filters_by_building(
    client,
    session_factory,
):
    building_1 = create_test_building(session_factory)
    building_2 = create_test_building(session_factory)

    for building_id, name in [
        (building_1, "AHU-01"),
        (building_2, "PUMP-01"),
    ]:
        response = client.post(
            "/api/v1/equipment/",
            json={
                "building_id": str(building_id),
                "name": name,
                "equipment_type": "ahu",
                "status": "offline",
            },
        )
        assert response.status_code == 201

    response = client.get(
        "/api/v1/equipment/",
        params={"building_id": str(building_1)},
    )

    assert response.status_code == 200

    data = response.json()

    assert len(data) == 1
    assert data[0]["name"] == "AHU-01"
    assert data[0]["building_id"] == str(building_1)


def test_list_equipment_filters_by_status(
    client,
    session_factory,
):
    building_id = create_test_building(session_factory)

    for name, equipment_status in [
        ("AHU-RUNNING", "running"),
        ("AHU-OFFLINE", "offline"),
    ]:
        response = client.post(
            "/api/v1/equipment/",
            json={
                "building_id": str(building_id),
                "name": name,
                "equipment_type": "ahu",
                "status": equipment_status,
            },
        )

        assert response.status_code == 201

    response = client.get(
        "/api/v1/equipment/",
        params={"status": "running"},
    )

    assert response.status_code == 200

    data = response.json()

    assert len(data) == 1
    assert data[0]["name"] == "AHU-RUNNING"
    assert data[0]["status"] == "running"


def test_list_equipment_filters_by_building_and_status(
    client,
    session_factory,
):
    building_1 = create_test_building(session_factory)
    building_2 = create_test_building(session_factory)

    equipment_data = [
        (building_1, "AHU-01", "running"),
        (building_1, "AHU-02", "offline"),
        (building_2, "AHU-03", "running"),
    ]

    for building_id, name, equipment_status in equipment_data:
        response = client.post(
            "/api/v1/equipment/",
            json={
                "building_id": str(building_id),
                "name": name,
                "equipment_type": "ahu",
                "status": equipment_status,
            },
        )

        assert response.status_code == 201

    response = client.get(
        "/api/v1/equipment/",
        params={
            "building_id": str(building_1),
            "status": "running",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert len(data) == 1
    assert data[0]["name"] == "AHU-01"
    assert data[0]["building_id"] == str(building_1)
    assert data[0]["status"] == "running"