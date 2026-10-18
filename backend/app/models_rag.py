"""
RAG document / index / admin-session persistence (Phase 6B.1).

Schema and invariants only — no upload or embedding execution in this phase.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Enum as SAEnum,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


def _enum_values(enum_class: type[Enum]) -> list[str]:
    return [item.value for item in enum_class]


class RagDocumentStatus(str, Enum):
    UPLOADED = "uploaded"
    VALIDATING = "validating"
    AWAITING_REVIEW = "awaiting_review"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"
    ARCHIVED = "archived"


class RagIndexRunStatus(str, Enum):
    PENDING = "pending"
    BUILDING = "building"
    READY = "ready"
    FAILED = "failed"
    SUPERSEDED = "superseded"


class RagSnapshotStatus(str, Enum):
    BUILDING = "building"
    READY = "ready"
    FAILED = "failed"
    SUPERSEDED = "superseded"


class RagChunkEligibility(str, Enum):
    ELIGIBLE = "eligible"
    EXCLUDED = "excluded"
    NEEDS_REVIEW = "needs_review"


class RagDocument(Base):
    __tablename__ = "rag_documents"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid, primary_key=True, default=uuid.uuid4
    )
    display_title: Mapped[str] = mapped_column(String(200), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    storage_key: Mapped[str] = mapped_column(
        String(512), nullable=False, unique=True
    )
    content_sha256: Mapped[str] = mapped_column(
        String(64), nullable=False, unique=True, index=True
    )
    byte_size: Mapped[int] = mapped_column(BigInteger, nullable=False)
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    indexed_page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[RagDocumentStatus] = mapped_column(
        SAEnum(
            RagDocumentStatus,
            name="rag_document_status_values",
            native_enum=False,
            create_constraint=True,
            values_callable=_enum_values,
        ),
        nullable=False,
        default=RagDocumentStatus.UPLOADED,
    )
    error_summary: Mapped[str | None] = mapped_column(String(500), nullable=True)
    eligibility_policy_id: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        doc="Explicit policy id required before activation (e.g. greenheck_erv_v1).",
    )
    eligibility_approved: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    pipeline_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # Application-managed pointer (no FK) to avoid circular dependency with runs.
    active_index_run_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, nullable=True)
    equipment_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid,
        ForeignKey("equipment.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    archived_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    index_runs: Mapped[list[RagIndexRun]] = relationship(
        back_populates="document",
        foreign_keys="RagIndexRun.document_id",
    )


class RagIndexRun(Base):
    __tablename__ = "rag_index_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid, primary_key=True, default=uuid.uuid4
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("rag_documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    status: Mapped[RagIndexRunStatus] = mapped_column(
        SAEnum(
            RagIndexRunStatus,
            name="rag_index_run_status_values",
            native_enum=False,
            create_constraint=True,
            values_callable=_enum_values,
        ),
        nullable=False,
        default=RagIndexRunStatus.PENDING,
    )
    pipeline_version: Mapped[str] = mapped_column(String(64), nullable=False)
    pipeline_config_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    embedding_model: Mapped[str] = mapped_column(String(128), nullable=False)
    chunk_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    artifact_ref: Mapped[str | None] = mapped_column(String(512), nullable=True)
    artifact_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_summary: Mapped[str | None] = mapped_column(String(500), nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    document: Mapped[RagDocument] = relationship(
        back_populates="index_runs",
        foreign_keys=[document_id],
    )
    chunks: Mapped[list[RagChunk]] = relationship(back_populates="index_run")


class RagChunk(Base):
    __tablename__ = "rag_chunks"
    __table_args__ = (
        UniqueConstraint(
            "index_run_id",
            "chunk_id",
            name="uq_rag_chunks_run_chunk_id",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid, primary_key=True, default=uuid.uuid4
    )
    index_run_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("rag_index_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("rag_documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_id: Mapped[str] = mapped_column(String(256), nullable=False)
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    section_heading: Mapped[str | None] = mapped_column(String(512), nullable=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    char_count: Mapped[int] = mapped_column(Integer, nullable=False)
    eligibility: Mapped[RagChunkEligibility] = mapped_column(
        SAEnum(
            RagChunkEligibility,
            name="rag_chunk_eligibility_values",
            native_enum=False,
            create_constraint=True,
            values_callable=_enum_values,
        ),
        nullable=False,
        default=RagChunkEligibility.NEEDS_REVIEW,
    )
    eligibility_reason: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    index_run: Mapped[RagIndexRun] = relationship(back_populates="chunks")


class RagIndexSnapshot(Base):
    """Immutable global retrieval snapshot (multi-document)."""

    __tablename__ = "rag_index_snapshots"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid, primary_key=True, default=uuid.uuid4
    )
    snapshot_version: Mapped[str] = mapped_column(
        String(64), nullable=False, unique=True
    )
    status: Mapped[RagSnapshotStatus] = mapped_column(
        SAEnum(
            RagSnapshotStatus,
            name="rag_snapshot_status_values",
            native_enum=False,
            create_constraint=True,
            values_callable=_enum_values,
        ),
        nullable=False,
        default=RagSnapshotStatus.BUILDING,
    )
    artifact_ref: Mapped[str | None] = mapped_column(String(512), nullable=True)
    artifact_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    activated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    members: Mapped[list[RagIndexSnapshotMember]] = relationship(
        back_populates="snapshot"
    )


class RagIndexSnapshotMember(Base):
    __tablename__ = "rag_index_snapshot_members"
    __table_args__ = (
        UniqueConstraint(
            "snapshot_id",
            "index_run_id",
            name="uq_rag_snapshot_member_run",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid, primary_key=True, default=uuid.uuid4
    )
    snapshot_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("rag_index_snapshots.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    index_run_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("rag_index_runs.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("rag_documents.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )

    snapshot: Mapped[RagIndexSnapshot] = relationship(back_populates="members")


class RagActiveIndexPointer(Base):
    """
    Singleton pointer to the active global snapshot.

    Only one row (id=1) is allowed. A failed build must never update this
    until a ready snapshot exists.
    """

    __tablename__ = "rag_active_index_pointer"
    __table_args__ = (
        CheckConstraint("id = 1", name="ck_rag_active_index_pointer_singleton"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    snapshot_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid,
        ForeignKey("rag_index_snapshots.id", ondelete="SET NULL"),
        nullable=True,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class AdminSession(Base):
    __tablename__ = "admin_sessions"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid, primary_key=True, default=uuid.uuid4
    )
    session_token_hash: Mapped[str] = mapped_column(
        String(64), nullable=False, unique=True, index=True
    )
    # CSRF synchronizer secret is session-scoped server state (not a password).
    # Returned to the authenticated same-origin client via GET /auth/me after refresh.
    csrf_token: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    revoked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_seen_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
