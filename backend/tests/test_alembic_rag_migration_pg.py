"""
Opt-in PostgreSQL Alembic migration check for Phase 6B.1.

Requires:
  RUN_ALEMBIC_PG_TEST=1
  A disposable DATABASE_* pointing at an empty Postgres database.

Does not run in the default offline suite.
"""

from __future__ import annotations

import os

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import URL


pytestmark = [
    pytest.mark.alembic_pg,
    pytest.mark.skipif(
        os.environ.get("RUN_ALEMBIC_PG_TEST") != "1",
        reason="opt-in Postgres Alembic test requires RUN_ALEMBIC_PG_TEST=1",
    ),
]


def _database_url() -> URL:
    password = os.environ.get("DATABASE_PASSWORD")
    if not password:
        pytest.skip("DATABASE_PASSWORD required for Postgres Alembic test")
    return URL.create(
        drivername="postgresql+psycopg",
        username=os.environ.get("DATABASE_USER", "thermalops_app"),
        password=password,
        host=os.environ.get("DATABASE_HOST", "127.0.0.1"),
        port=int(os.environ.get("DATABASE_PORT", "5432")),
        database=os.environ.get("DATABASE_NAME", "thermalops"),
    )


def test_alembic_upgrade_creates_rag_tables(tmp_path):
    """
    WARNING: runs alembic upgrade head against the configured database.

    Use only a disposable database — never production.
    """
    url = _database_url()
    backend_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cfg = Config(os.path.join(backend_root, "alembic.ini"))
    cfg.set_main_option("script_location", os.path.join(backend_root, "migrations"))

    # Alembic env.py reads app settings; ensure password is present in env.
    engine = create_engine(url)
    try:
        command.upgrade(cfg, "head")
        insp = inspect(engine)
        tables = set(insp.get_table_names())
        assert "rag_documents" in tables
        assert "admin_sessions" in tables
        assert "rag_active_index_pointer" in tables
        with engine.connect() as conn:
            version = conn.execute(
                text("select version_num from alembic_version")
            ).scalar_one()
            assert version == "b7e4c2a91f05"
    finally:
        engine.dispose()
