"""Idempotent Postgres loading: COPY into a temporary staging table, then upsert.

INSERT ... ON CONFLICT (pk) DO UPDATE ... WHERE target IS DISTINCT FROM excluded
inserts new keys, corrects changed rows and touches nothing that is already identical, so an
identical re-run reports rows_changed = 0. ingest_runs is an append-only log of runs and is the
one table that grows on a re-run, by design.
"""
import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import psycopg

from .. import config

SCHEMA_SQL = Path(__file__).with_name("schema.sql")
PKEYS = {"nodes": ["dataset", "name"], "links": ["dataset", "link_id"], "flows": ["dataset", "flow_id"],
         "routing": ["dataset", "link_id", "flow_id"], "time_steps": ["dataset", "step"],
         "traffic_matrix": ["dataset", "step", "flow_id"], "link_load": ["dataset", "step", "link_id"],
         "cesnet_series": ["aggregation", "institution_id", "id_time"], "datasets": ["dataset"],
         "mawi_traces": ["trace_id"], "mawi_protocol_mix": ["trace_id", "proto"],
         "mawi_top_talkers": ["trace_id", "kind", "rank"], "mawi_throughput": ["trace_id", "bin_start"],
         "forecasts": ["dataset", "model", "horizon_steps", "fold", "entity_kind", "entity_id", "target_step"],
         "scenarios": ["scenario_id"],
         "anomalies": ["dataset", "detector", "entity_kind", "entity_id", "target_step", "injected"]}
DATASETS = [
    ("abilene", "Abilene backbone, 12 routers, 144 OD flows, 5-minute traffic matrices, 24 weeks of 2004",
     "https://www.cs.utexas.edu/~yzhang/research/AbileneTM/", "no license stated; research dataset (Y. Zhang)",
     "100 bytes per 5 minutes", "real"),
    ("cesnet_institutions", "CESNET-TimeSeries24, institutions subset: 283 per-institution aggregate traffic series",
     "https://doi.org/10.5281/zenodo.13382427", "CC-BY-4.0", "flows/packets/bytes per aggregation window", "real"),
]


def connect(url=None, **kw):
    return psycopg.connect(url or config.DATABASE_URL, **kw)


def apply_schema(conn):
    with conn.cursor() as cur:
        cur.execute(SCHEMA_SQL.read_text())
        cur.executemany("INSERT INTO datasets VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT (dataset) DO UPDATE SET "
                        "description=excluded.description, source_url=excluded.source_url, license=excluded.license,"
                        " unit=excluded.unit, provenance=excluded.provenance", DATASETS)
    conn.commit()


def _csv_chunks(df, rows=500_000):
    for i in range(0, df.height, rows):
        buf = io.BytesIO()
        df.slice(i, rows).write_csv(buf, include_header=False, null_value="")
        yield buf.getvalue()


def upsert(conn, table, frames, columns):
    """Load an iterable of polars frames (columns in `columns` order) into `table`. Returns rows (seen, changed)."""
    pk = PKEYS[table]
    cols = ", ".join(columns)
    rest = [c for c in columns if c not in pk]
    update = ", ".join(f"{c} = excluded.{c}" for c in rest)
    distinct = (f"({', '.join(f'{table}.{c}' for c in rest)}) IS DISTINCT FROM "
                f"({', '.join(f'excluded.{c}' for c in rest)})")
    conflict = f"DO UPDATE SET {update} WHERE {distinct}" if rest else "DO NOTHING"
    seen = changed = 0
    with conn.cursor() as cur:
        cur.execute(f"CREATE TEMP TABLE IF NOT EXISTS stage_{table} (LIKE {table} INCLUDING DEFAULTS)")
        for df in frames:
            cur.execute(f"TRUNCATE stage_{table}")
            with cur.copy(f"COPY stage_{table} ({cols}) FROM STDIN WITH (FORMAT csv)") as cp:
                for chunk in _csv_chunks(df.select(columns)):
                    cp.write(chunk)
            cur.execute(f"INSERT INTO {table} ({cols}) SELECT {cols} FROM stage_{table} "
                        f"ON CONFLICT ({', '.join(pk)}) {conflict}")
            changed += cur.rowcount
            seen += df.height
    conn.commit()
    return seen, changed


def table_fingerprint(conn, table, where="TRUE"):
    """Row count and an order-independent checksum of every column (md5 of each row, summed)."""
    with conn.cursor() as cur:
        cur.execute(f"SELECT count(*), coalesce(sum(('x' || substr(md5(t::text), 1, 15))::bit(60)::bigint), 0) "
                    f"FROM {table} t WHERE {where}")
        n, s = cur.fetchone()
    return {"rows": int(n), "checksum": str(s)}


def log_run(conn, dataset, target, started, rows_seen, rows_changed, wall_s, details):
    with conn.cursor() as cur:
        cur.execute("INSERT INTO ingest_runs (dataset, target, started_at, finished_at, rows_seen, rows_changed,"
                    " wall_s, details) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                    (dataset, target, started, datetime.now(timezone.utc), rows_seen, rows_changed, wall_s,
                     json.dumps(details, default=str)))
    conn.commit()


def timed_upsert(conn, table, frames, columns):
    t0 = time.perf_counter()
    seen, changed = upsert(conn, table, frames, columns)
    wall = time.perf_counter() - t0
    return {"rows_seen": seen, "rows_changed": changed, "wall_s": wall, "rows_per_s": seen / wall if wall else None}
