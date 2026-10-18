"""add rag admin foundation tables

Revision ID: b7e4c2a91f05
Revises: c8625a040993
Create Date: 2026-10-18 08:40:00.000000

Phase 6B.1 — document metadata, index runs/chunks, global snapshot pointer,
and hashed admin sessions. No embedding execution.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "b7e4c2a91f05"
down_revision: Union[str, Sequence[str], None] = "c8625a040993"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "rag_documents",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("display_title", sa.String(length=200), nullable=False),
        sa.Column("original_filename", sa.String(length=255), nullable=False),
        sa.Column("storage_key", sa.String(length=512), nullable=False),
        sa.Column("content_sha256", sa.String(length=64), nullable=False),
        sa.Column("byte_size", sa.BigInteger(), nullable=False),
        sa.Column("page_count", sa.Integer(), nullable=True),
        sa.Column("indexed_page_count", sa.Integer(), nullable=True),
        sa.Column(
            "status",
            sa.Enum(
                "uploaded",
                "validating",
                "awaiting_review",
                "processing",
                "ready",
                "failed",
                "archived",
                name="rag_document_status_values",
                native_enum=False,
                create_constraint=True,
            ),
            nullable=False,
        ),
        sa.Column("error_summary", sa.String(length=500), nullable=True),
        sa.Column("eligibility_policy_id", sa.String(length=128), nullable=True),
        sa.Column(
            "eligibility_approved",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.Column("pipeline_version", sa.String(length=64), nullable=True),
        sa.Column("active_index_run_id", sa.Uuid(), nullable=True),
        sa.Column("equipment_id", sa.Uuid(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("archived_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["equipment_id"], ["equipment.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("storage_key"),
        sa.UniqueConstraint("content_sha256"),
    )
    op.create_index(
        op.f("ix_rag_documents_content_sha256"),
        "rag_documents",
        ["content_sha256"],
        unique=True,
    )
    op.create_index(
        op.f("ix_rag_documents_equipment_id"),
        "rag_documents",
        ["equipment_id"],
        unique=False,
    )

    op.create_table(
        "rag_index_runs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "pending",
                "building",
                "ready",
                "failed",
                "superseded",
                name="rag_index_run_status_values",
                native_enum=False,
                create_constraint=True,
            ),
            nullable=False,
        ),
        sa.Column("pipeline_version", sa.String(length=64), nullable=False),
        sa.Column("pipeline_config_hash", sa.String(length=64), nullable=True),
        sa.Column("embedding_model", sa.String(length=128), nullable=False),
        sa.Column("chunk_count", sa.Integer(), nullable=True),
        sa.Column("artifact_ref", sa.String(length=512), nullable=True),
        sa.Column("artifact_sha256", sa.String(length=64), nullable=True),
        sa.Column("error_summary", sa.String(length=500), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["document_id"], ["rag_documents.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_rag_index_runs_document_id"),
        "rag_index_runs",
        ["document_id"],
        unique=False,
    )

    op.create_table(
        "rag_chunks",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("index_run_id", sa.Uuid(), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("chunk_id", sa.String(length=256), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=False),
        sa.Column("section_heading", sa.String(length=512), nullable=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("char_count", sa.Integer(), nullable=False),
        sa.Column(
            "eligibility",
            sa.Enum(
                "eligible",
                "excluded",
                "needs_review",
                name="rag_chunk_eligibility_values",
                native_enum=False,
                create_constraint=True,
            ),
            nullable=False,
        ),
        sa.Column("eligibility_reason", sa.String(length=255), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["document_id"], ["rag_documents.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["index_run_id"], ["rag_index_runs.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "index_run_id", "chunk_id", name="uq_rag_chunks_run_chunk_id"
        ),
    )
    op.create_index(
        op.f("ix_rag_chunks_document_id"),
        "rag_chunks",
        ["document_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_rag_chunks_index_run_id"),
        "rag_chunks",
        ["index_run_id"],
        unique=False,
    )

    op.create_table(
        "rag_index_snapshots",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("snapshot_version", sa.String(length=64), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "building",
                "ready",
                "failed",
                "superseded",
                name="rag_snapshot_status_values",
                native_enum=False,
                create_constraint=True,
            ),
            nullable=False,
        ),
        sa.Column("artifact_ref", sa.String(length=512), nullable=True),
        sa.Column("artifact_sha256", sa.String(length=64), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("activated_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("snapshot_version"),
    )

    op.create_table(
        "rag_index_snapshot_members",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("snapshot_id", sa.Uuid(), nullable=False),
        sa.Column("index_run_id", sa.Uuid(), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.ForeignKeyConstraint(
            ["document_id"], ["rag_documents.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["index_run_id"], ["rag_index_runs.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["snapshot_id"], ["rag_index_snapshots.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "snapshot_id", "index_run_id", name="uq_rag_snapshot_member_run"
        ),
    )
    op.create_index(
        op.f("ix_rag_index_snapshot_members_document_id"),
        "rag_index_snapshot_members",
        ["document_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_rag_index_snapshot_members_index_run_id"),
        "rag_index_snapshot_members",
        ["index_run_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_rag_index_snapshot_members_snapshot_id"),
        "rag_index_snapshot_members",
        ["snapshot_id"],
        unique=False,
    )

    op.create_table(
        "rag_active_index_pointer",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("snapshot_id", sa.Uuid(), nullable=True),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("id = 1", name="ck_rag_active_index_pointer_singleton"),
        sa.ForeignKeyConstraint(
            ["snapshot_id"], ["rag_index_snapshots.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "admin_sessions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("session_token_hash", sa.String(length=64), nullable=False),
        sa.Column("csrf_token", sa.String(length=128), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("session_token_hash"),
    )
    op.create_index(
        op.f("ix_admin_sessions_session_token_hash"),
        "admin_sessions",
        ["session_token_hash"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_admin_sessions_session_token_hash"), table_name="admin_sessions"
    )
    op.drop_table("admin_sessions")
    op.drop_table("rag_active_index_pointer")
    op.drop_index(
        op.f("ix_rag_index_snapshot_members_snapshot_id"),
        table_name="rag_index_snapshot_members",
    )
    op.drop_index(
        op.f("ix_rag_index_snapshot_members_index_run_id"),
        table_name="rag_index_snapshot_members",
    )
    op.drop_index(
        op.f("ix_rag_index_snapshot_members_document_id"),
        table_name="rag_index_snapshot_members",
    )
    op.drop_table("rag_index_snapshot_members")
    op.drop_table("rag_index_snapshots")
    op.drop_index(op.f("ix_rag_chunks_index_run_id"), table_name="rag_chunks")
    op.drop_index(op.f("ix_rag_chunks_document_id"), table_name="rag_chunks")
    op.drop_table("rag_chunks")
    op.drop_index(op.f("ix_rag_index_runs_document_id"), table_name="rag_index_runs")
    op.drop_table("rag_index_runs")
    op.drop_index(op.f("ix_rag_documents_equipment_id"), table_name="rag_documents")
    op.drop_index(op.f("ix_rag_documents_content_sha256"), table_name="rag_documents")
    op.drop_table("rag_documents")
