"""Offline SQLite schema/constraint tests for RAG persistence tables."""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.models import Building, Equipment, EquipmentType, Organization
from app.models_rag import (
    RagActiveIndexPointer,
    RagChunk,
    RagChunkEligibility,
    RagDocument,
    RagDocumentStatus,
    RagIndexRun,
    RagIndexRunStatus,
    RagIndexSnapshot,
    RagIndexSnapshotMember,
    RagSnapshotStatus,
)


@pytest.fixture
def db_session():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    session = Session()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


def test_rag_tables_registered():
    names = set(Base.metadata.tables.keys())
    assert "rag_documents" in names
    assert "rag_index_runs" in names
    assert "rag_chunks" in names
    assert "rag_index_snapshots" in names
    assert "rag_index_snapshot_members" in names
    assert "rag_active_index_pointer" in names
    assert "admin_sessions" in names


def test_document_unique_content_sha256(db_session):
    doc1 = RagDocument(
        display_title="A",
        original_filename="a.pdf",
        storage_key="keys/a",
        content_sha256="a" * 64,
        byte_size=10,
        status=RagDocumentStatus.UPLOADED,
        eligibility_approved=False,
    )
    db_session.add(doc1)
    db_session.commit()

    doc2 = RagDocument(
        display_title="B",
        original_filename="b.pdf",
        storage_key="keys/b",
        content_sha256="a" * 64,
        byte_size=11,
        status=RagDocumentStatus.UPLOADED,
        eligibility_approved=False,
    )
    db_session.add(doc2)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


def test_chunk_unique_per_index_run(db_session):
    doc = RagDocument(
        display_title="Doc",
        original_filename="d.pdf",
        storage_key="keys/d",
        content_sha256="b" * 64,
        byte_size=1,
        status=RagDocumentStatus.UPLOADED,
        eligibility_approved=False,
    )
    db_session.add(doc)
    db_session.flush()
    run = RagIndexRun(
        document_id=doc.id,
        status=RagIndexRunStatus.PENDING,
        pipeline_version="v1",
        embedding_model="minilm",
    )
    db_session.add(run)
    db_session.flush()

    chunk = RagChunk(
        index_run_id=run.id,
        document_id=doc.id,
        chunk_id="doc:p1:c0:bbbbbbbbbbbb:abcd",
        page_number=1,
        text="hello",
        char_count=5,
        eligibility=RagChunkEligibility.NEEDS_REVIEW,
    )
    db_session.add(chunk)
    db_session.commit()

    dup = RagChunk(
        index_run_id=run.id,
        document_id=doc.id,
        chunk_id="doc:p1:c0:bbbbbbbbbbbb:abcd",
        page_number=1,
        text="hello",
        char_count=5,
        eligibility=RagChunkEligibility.NEEDS_REVIEW,
    )
    db_session.add(dup)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


def test_snapshot_members_and_active_pointer(db_session):
    org = Organization(name="Org")
    db_session.add(org)
    db_session.flush()
    building = Building(organization_id=org.id, name="B1", timezone="UTC")
    db_session.add(building)
    db_session.flush()
    equipment = Equipment(
        building_id=building.id,
        name="AHU",
        equipment_type=EquipmentType.AHU,
    )
    db_session.add(equipment)
    db_session.flush()

    doc = RagDocument(
        display_title="Manual",
        original_filename="m.pdf",
        storage_key="keys/m",
        content_sha256="c" * 64,
        byte_size=100,
        status=RagDocumentStatus.READY,
        eligibility_policy_id="greenheck_erv_v1",
        eligibility_approved=True,
        equipment_id=equipment.id,
    )
    db_session.add(doc)
    db_session.flush()
    run = RagIndexRun(
        document_id=doc.id,
        status=RagIndexRunStatus.READY,
        pipeline_version="v1",
        embedding_model="minilm",
        chunk_count=1,
        artifact_ref="artifacts/run1.npz",
        artifact_sha256="d" * 64,
    )
    db_session.add(run)
    db_session.flush()

    snap = RagIndexSnapshot(
        snapshot_version="snap_001",
        status=RagSnapshotStatus.READY,
        artifact_ref="artifacts/snap_001.npz",
        artifact_sha256="e" * 64,
    )
    db_session.add(snap)
    db_session.flush()
    member = RagIndexSnapshotMember(
        snapshot_id=snap.id,
        index_run_id=run.id,
        document_id=doc.id,
    )
    db_session.add(member)
    pointer = RagActiveIndexPointer(id=1, snapshot_id=snap.id)
    db_session.add(pointer)
    db_session.commit()

    loaded = db_session.get(RagActiveIndexPointer, 1)
    assert loaded is not None
    assert loaded.snapshot_id == snap.id
    assert len(snap.members) == 1


def test_active_pointer_singleton_constraint(db_session):
    pointer = RagActiveIndexPointer(id=1, snapshot_id=None)
    db_session.add(pointer)
    db_session.commit()
    bad = RagActiveIndexPointer(id=2, snapshot_id=None)
    db_session.add(bad)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()
