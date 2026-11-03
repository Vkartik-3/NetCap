"""CESNET-TimeSeries24, institutions subset (Zenodo 10.5281/zenodo.13382427, CC-BY-4.0).

Per-institution aggregate traffic time series (283 institutions; 10-minute, 1-hour and 1-day
aggregations). This is NOT link-level data and has no topology: NetCap uses it for forecasting
at scale, pipeline throughput and anomaly detection, never for routing scenarios.

Layout: institutions/agg_<agg>/<institution_id>.csv keyed by id_time; times/times_<agg>.csv maps
id_time to a timestamp with a UTC offset (10-minute times carry +01/+02 offsets and irregular
seconds; hourly/daily are UTC). All timestamps are stored as UTC.
"""
from pathlib import Path

import polars as pl

from .. import config

AGGREGATIONS = {"10_minutes": "agg_10_minutes", "1_hour": "agg_1_hour", "1_day": "agg_1_day"}
METRICS = ["n_flows", "n_packets", "n_bytes", "sum_n_dest_asn", "average_n_dest_asn", "std_n_dest_asn",
           "sum_n_dest_ports", "average_n_dest_ports", "std_n_dest_ports", "sum_n_dest_ip",
           "average_n_dest_ip", "std_n_dest_ip", "tcp_udp_ratio_packets", "tcp_udp_ratio_bytes",
           "dir_ratio_packets", "dir_ratio_bytes", "avg_duration", "avg_ttl"]
INT_METRICS = {"n_flows", "n_packets", "n_bytes", "sum_n_dest_asn", "sum_n_dest_ports", "sum_n_dest_ip"}
SCHEMA = {"id_time": pl.Int32, **{m: (pl.Int64 if m in INT_METRICS else pl.Float64) for m in METRICS}}
COLUMNS = ["aggregation", "institution_id", "id_time", "ts", *METRICS]


def times(agg, raw=None):
    raw = Path(raw or config.RAW_CESNET)
    t = pl.read_csv(raw / "times" / f"times_{agg}.csv", schema={"id_time": pl.Int32, "time": pl.String})
    return t.select("id_time", pl.col("time").str.to_datetime("%Y-%m-%d %H:%M:%S%z", time_zone="UTC",
                                                              time_unit="us").alias("ts"))


def institution_files(agg, raw=None):
    raw = Path(raw or config.RAW_CESNET)
    files = sorted((raw / "institutions" / AGGREGATIONS[agg]).glob("*.csv"), key=lambda p: int(p.stem))
    if not files:
        raise FileNotFoundError(f"no CESNET files for {agg} under {raw}")
    return files


def read_batch(files, agg, tmap):
    """Read a batch of institution CSVs into one frame with institution id, UTC timestamp and month."""
    frames = [pl.read_csv(f, schema=SCHEMA).with_columns(pl.lit(int(f.stem), pl.Int32).alias("institution_id"))
              for f in files]
    df = pl.concat(frames).join(tmap, on="id_time", how="left", validate="m:1")
    if df["ts"].null_count():
        raise ValueError(f"{df['ts'].null_count()} rows with an id_time missing from times_{agg}.csv")
    return df.with_columns(pl.lit(agg).alias("aggregation")).select(COLUMNS).sort("institution_id", "id_time")


def iter_batches(agg, batch_size=32, raw=None):
    """Yield (batch_index, frame) over the institutions in id order, batch_size files at a time."""
    tmap = times(agg, raw)
    files = institution_files(agg, raw)
    for b, i in enumerate(range(0, len(files), batch_size)):
        yield b, read_batch(files[i:i + batch_size], agg, tmap)


def source_bytes(raw=None, aggs=tuple(AGGREGATIONS)):
    raw = Path(raw or config.RAW_CESNET)
    return {agg: sum(f.stat().st_size for f in institution_files(agg, raw)) for agg in aggs}
