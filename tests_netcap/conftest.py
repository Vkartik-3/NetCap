import os
import uuid

import pytest

from netcap import config

HAVE_ABILENE = (config.RAW_ABILENE / "X24.gz").exists()
HAVE_CESNET = (config.RAW_CESNET / "times" / "times_10_minutes.csv").exists()
needs_abilene = pytest.mark.skipif(not HAVE_ABILENE, reason="raw Abilene files not downloaded")
needs_cesnet = pytest.mark.skipif(not HAVE_CESNET, reason="CESNET institutions not downloaded")


def _db_url():
    return os.environ.get("NETCAP_TEST_DATABASE_URL", config.DATABASE_URL)


@pytest.fixture
def pg():
    """A connection whose search_path is a fresh schema with the NetCap tables; dropped afterwards."""
    psycopg = pytest.importorskip("psycopg")
    try:
        admin = psycopg.connect(_db_url(), connect_timeout=3, autocommit=True)
    except psycopg.OperationalError as e:
        pytest.skip(f"no Postgres at NETCAP_TEST_DATABASE_URL: {e}")
    schema = f"test_{uuid.uuid4().hex[:10]}"
    admin.execute(f"CREATE SCHEMA {schema}")
    conn = psycopg.connect(_db_url(), options=f"-c search_path={schema}")
    try:
        from netcap.db import load
        load.apply_schema(conn)
        yield conn
    finally:
        conn.close()
        admin.execute(f"DROP SCHEMA {schema} CASCADE")
        admin.close()


@pytest.fixture
def live_db():
    """The main NetCap database (for checks on data loaded by the ingest CLI)."""
    psycopg = pytest.importorskip("psycopg")
    try:
        conn = psycopg.connect(config.DATABASE_URL, connect_timeout=3)
    except psycopg.OperationalError as e:
        pytest.skip(f"no Postgres at NETCAP_DATABASE_URL: {e}")
    yield conn
    conn.close()
