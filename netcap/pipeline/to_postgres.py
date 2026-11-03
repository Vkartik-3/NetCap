"""Parquet store -> Postgres, one partition file at a time (bounded memory)."""
from pathlib import Path

import polars as pl

from .. import config
from ..db import load as db
from ..provenance import peak_rss_bytes

ABILENE_COLUMNS = {
    "nodes": ["dataset", "node_id", "name", "city", "lat", "lon"],
    "links": ["dataset", "link_id", "name", "src", "dst", "link_type", "capacity_kbps", "igp_weight"],
    "flows": ["dataset", "flow_id", "src", "dst"],
    "routing": ["dataset", "link_id", "flow_id", "fraction"],
    "time_steps": ["dataset", "step", "week", "step_in_week", "ts_assumed", "keep", "reason", "segment_id"],
    "traffic_matrix": ["dataset", "step", "flow_id", "value_raw"],
    "link_load": ["dataset", "step", "link_id", "load_raw", "load_mbps", "utilization"],
}
CESNET_COLUMNS = ["aggregation", "institution_id", "id_time", "ts", "n_flows", "n_packets", "n_bytes",
                  "sum_n_dest_asn", "average_n_dest_asn", "std_n_dest_asn", "sum_n_dest_ports",
                  "average_n_dest_ports", "std_n_dest_ports", "sum_n_dest_ip", "average_n_dest_ip", "std_n_dest_ip",
                  "tcp_udp_ratio_packets", "tcp_udp_ratio_bytes", "dir_ratio_packets", "dir_ratio_bytes",
                  "avg_duration", "avg_ttl"]


def _files(table_dir):
    files = sorted(Path(table_dir).rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no Parquet files under {table_dir}; run the parquet ingest first")
    return files


def _abilene_frames(table_dir):
    for f in _files(table_dir):
        df = pl.read_parquet(f, hive_partitioning=False).with_columns(dataset=pl.lit("abilene"))
        if "reason" in df.columns:
            df = df.with_columns(pl.when(pl.col("reason") == "").then(None).otherwise(pl.col("reason")).alias("reason"))
        yield df


def abilene_to_postgres(conn, root=None):
    root = (root or config.DATA) / "abilene"
    out = {name: db.timed_upsert(conn, name, _abilene_frames(root / name), cols)
           for name, cols in ABILENE_COLUMNS.items()}
    return {"tables": out, "rows_seen": sum(v["rows_seen"] for v in out.values()),
            "rows_changed": sum(v["rows_changed"] for v in out.values()),
            "wall_s": sum(v["wall_s"] for v in out.values()), "peak_rss_bytes": peak_rss_bytes()}


def cesnet_to_postgres(conn, root=None):
    root = (root or config.DATA) / "cesnet" / "series"
    frames = (pl.read_parquet(f, hive_partitioning=False) for f in _files(root))
    r = db.timed_upsert(conn, "cesnet_series", frames, CESNET_COLUMNS)
    return {**r, "peak_rss_bytes": peak_rss_bytes()}
