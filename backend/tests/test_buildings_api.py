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


def create_test_building(session_factory, name="Test Building", location="Test Location"):
    with session_factory() as db:
        organization = Organization(name="Test Organization")
        db.add(organization)
        db.flush()

        building = Building(
            organization_id=organization.id,
            name=name,
            location=location,
            timezone="UTC",
        )
        db.add(building)
        db.commit()
        db.refresh(building)

        return {
            "id": building.id,
            "organization_id": building.organization_id,
            "name": building.name,
            "location": building.location,
            "timezone": building.timezone,
        }


def test_list_buildings_returns_empty_list(client):
    response = client.get("/api/v1/buildings/")

    assert response.status_code == 200
    assert response.json() == []


def test_list_buildings_returns_real_ids_and_names(client, session_factory):
    created = create_test_building(
        session_factory,
        name="Riverside Corporate Center",
        location="Austin, TX",
    )

    response = client.get("/api/v1/buildings/")

    assert response.status_code == 200

    data = response.json()
    assert len(data) == 1
    assert data[0]["id"] == str(created["id"])
    assert data[0]["organization_id"] == str(created["organization_id"])
    assert data[0]["name"] == "Riverside Corporate Center"
    assert data[0]["location"] == "Austin, TX"
    assert data[0]["timezone"] == "UTC"
    assert "created_at" in data[0]


def test_list_buildings_orders_by_name(client, session_factory):
    create_test_building(session_factory, name="Zebra Hall")
    create_test_building(session_factory, name="Alpha Tower")

    response = client.get("/api/v1/buildings/")

    assert response.status_code == 200

    names = [item["name"] for item in response.json()]
    assert names == ["Alpha Tower", "Zebra Hall"]


def test_buildings_endpoint_is_read_only_get(client):
    response = client.post("/api/v1/buildings/", json={"name": "Nope"})
    assert response.status_code == 405


def test_buildings_router_does_not_expose_detail_route(client):
    response = client.get(f"/api/v1/buildings/{uuid4()}")
    assert response.status_code == 404
